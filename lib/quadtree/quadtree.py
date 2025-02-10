from PIL import Image, ImageDraw

from lib.quadtree.quadrant import Quadrant
from lib.quadtree.utils import interpolate
from itertools import chain

MAX_DEPTH = 6
HOMOGENEITY_THRESHOLD = 0.005
COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
COLOUR_PALETTE = list(chain.from_iterable([interpolate(COLOUR_1, COLOUR_2, i / (256 - 1)) for i in range(256)]))

class QuadTree():
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
        if root.depth >= MAX_DEPTH or root.homogeneity <= HOMOGENEITY_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            # assign quadrant to leaf and stop recursing
            root.leaf = True
            return
        
        # Check for lines here
        # Also set max depth and leaf node if line is found
        # Else split quadtrant and build children
        # root.check_lines(image)

        # split quadrant if there is too much detail
        root.split_quadrant(points)

        for children in root.children:
            self.build(children, points)

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
            else:
                draw.rectangle(quadrant.bbox, int(quadrant.avg * 255))

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