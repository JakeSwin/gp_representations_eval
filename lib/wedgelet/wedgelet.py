import numpy as np

from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from lib.wedgelet.quadrant import Quadrant
from lib.wedgelet.utils import interpolate
from itertools import chain

MAX_DEPTH = 6
HOMOGENEITY_THRESHOLD = 0.005
COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
COLOUR_PALETTE = list(chain.from_iterable([interpolate(COLOUR_1, COLOUR_2, i / (256 - 1)) for i in range(256)]))

class Wedgelet():
    def __init__(self, points):
        self.height, self.width = points.shape

        # keep track of max depth achieved by recursion
        self.max_depth = 0

        # start compression
        self.start(points)

    def start(self, points):
        # create initial root
        self.root = Quadrant(points, (0, 0, self.width, self.height), 0)

        # build quadtree
        self.build(self.root, points)

    def build(self, root, points):
        # Maybe check for line here and compare if line homogeneity is better when reaching max depth
        if root.depth >= MAX_DEPTH or root.homogeneity <= HOMOGENEITY_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            # assign quadrant to leaf and stop recursing
            root.leaf = True
            return
        
        # Check for lines here
        # Also set max depth and leaf node if line is found
        # Else split quadtrant and build children
        h_mean, p, theta, p1_avg, p2_avg = root.check_lines(points)
        if h_mean <= HOMOGENEITY_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth
            
            root.leaf = True
            root.isline = True
            root.p = p
            root.theta = theta
            root.p1_avg = p1_avg
            root.p2_avg = p2_avg
            return

        # split quadrant if there is too much detail
        root.split_quadrant(points)

        for children in root.children:
            self.build(children, points)

    def draw_line_regions(self, quadrant, draw, show_lines=False):
        width = quadrant.bbox[2] - quadrant.bbox[0]
        height = quadrant.bbox[3] - quadrant.bbox[1]
        # OOB Line points of line equation
        p1 = (-width, (quadrant.p - -width * np.cos(quadrant.theta)) / np.sin(quadrant.theta))
        p2 = (width, (quadrant.p - width * np.cos(quadrant.theta)) / np.sin(quadrant.theta))
        
        x_min, x_max = -np.floor(width/2), np.ceil(width/2)
        y_min, y_max = -np.floor(height/2), np.ceil(height/2)

        top_left = (x_min, y_max)
        top_right = (x_max, y_max)
        bottom_right = (x_max, y_min)
        bottom_left = (x_min, y_min)

        line_segments = [
            [top_left, top_right],
            [top_right, bottom_right],
            [bottom_right, bottom_left],
            [bottom_left, top_left]
        ]
        intersection_points = []
        for p_seg1, p_seg2 in line_segments:
            # x1, y1, x2, y2 are coordinates for box line segment
            x1, y1 = p_seg1
            x2, y2 = p_seg2
            # x3, y3, x4, y4 are coordinates for line equation segment
            x3, y3 = p1
            x4, y4 = p2
            # Calculate t
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            # Stop calculation if no intersection
            if (t < 0 or t > 1):
                continue
            else:
                ix, iy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                intersection_points.append((ix, iy))

        points = [top_left, top_right, bottom_right, bottom_left]

        x_min_y_val = (quadrant.p - (x_min * np.cos(quadrant.theta))) / np.sin(quadrant.theta)
        x_max_y_val = (quadrant.p - (x_max * np.cos(quadrant.theta))) / np.sin(quadrant.theta)

        above_line = [p for p in points if p[0] == x_min and p[1] > x_min_y_val or p[0] == x_max and p[1] > x_max_y_val]
        below_line = [p for p in points if p[0] == x_min and p[1] < x_min_y_val or p[0] == x_max and p[1] < x_max_y_val]

        above_line.extend(intersection_points)
        below_line.extend(intersection_points)

        above_line = np.asarray(above_line, dtype=np.int32)
        below_line = np.asarray(below_line, dtype=np.int32)

        above_line[:,0] = above_line[:,0] + width/2 + quadrant.bbox[0]
        above_line[:,1] = -above_line[:,1] + height/2 + quadrant.bbox[1]
        below_line[:,0] = below_line[:,0] + width/2 + quadrant.bbox[0]
        below_line[:,1] = -below_line[:,1] + height/2 + quadrant.bbox[1]

        above_hull = ConvexHull(above_line)
        below_hull = ConvexHull(below_line)

        if show_lines:
            draw.polygon([(int(x),int(y)) for x,y in above_line[above_hull.vertices]], int(quadrant.p1_avg * 255), outline=2)
            draw.polygon([(int(x),int(y)) for x,y in below_line[below_hull.vertices]], int(quadrant.p2_avg * 255), outline=2)
        else:
            draw.polygon([(int(x),int(y)) for x,y in above_line[above_hull.vertices]], int(quadrant.p1_avg * 255))
            draw.polygon([(int(x),int(y)) for x,y in below_line[below_hull.vertices]], int(quadrant.p2_avg * 255))


    def create_image(self, custom_depth, show_lines=False):
        # create blank image canvas
        image = Image.new('P', (self.width, self.height))
        image.putpalette(COLOUR_PALETTE)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, self.height), 0)

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        # draw rectangle size of quadrant for each leaf quadrant
        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, int(quadrant.avg * 255), outline=2)
                if quadrant.isline:
                    self.draw_line_regions(quadrant, draw, show_lines)
            else:
                draw.rectangle(quadrant.bbox, int(quadrant.avg * 255))
                if quadrant.isline:
                    self.draw_line_regions(quadrant, draw, show_lines)

        return image.convert("RGB")

    def get_leaf_quadrants(self, depth):
        if depth > self.max_depth:
            raise ValueError('A depth larger than the trees depth was given')

        quandrants = []

        # search recursively down the quadtree
        self.recursive_search(self, self.root, depth, quandrants.append)

        return quandrants

    def recursive_search(self, tree, quadrant, max_depth, append_leaf):
        # append if quadrant is a leaf
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        # otherwise keep recursing
        elif quadrant.children != None:
            for child in quadrant.children:
                self.recursive_search(tree, child, max_depth, append_leaf)

    def create_gif(self, file_name, duration=1000, loop=0, show_lines=False):
        gif = []
        end_product_image = self.create_image(self.max_depth, show_lines=show_lines)

        for i in range(self.max_depth):
            image = self.create_image(i, show_lines=show_lines)
            gif.append(image)

        # add extra images at end
        for _ in range(4):
            gif.append(end_product_image)

        gif[0].save(
            file_name,
            save_all=True,
            append_images=gif[1:],
            duration=duration, loop=loop)