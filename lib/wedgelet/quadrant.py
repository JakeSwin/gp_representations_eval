import numpy as np

from lib.wedgelet.utils import get_homogeneity, cut_points

HOMOGENEITY_THRESHOLD = 0.008

class Quadrant():
    # Add coordinate (x,y) input if random sampling
    def __init__(self, points, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False
        self.isline = False
        self.p = None
        self.theta = None
        self.p1_avg = None
        self.p2_avg = None

        # crop image to quadrant size
        # image = image.crop(bbox)
        left, top, width, height = self.bbox
        # points = points[left:left+width, top:top+height]
        points = points[top:height, left:width]
        # hist = image.histogram()

        self.avg = np.mean(points)
        self.homogeneity = get_homogeneity(points, self.avg)
        # self.detail = get_detail(hist)
        # self.colour = average_colour(image)

    def split_quadrant(self, points):
        left, top, width, height = self.bbox

        # get the middle coords of bbox
        middle_x = int(left + (width - left) / 2)
        middle_y = int(top + (height - top) / 2)

        # split root quadrant into 4 new quadrants
        upper_left = Quadrant(points, (left, top, middle_x, middle_y), self.depth+1)
        upper_right = Quadrant(points, (middle_x, top, width, middle_y), self.depth+1)
        bottom_left = Quadrant(points, (left, middle_y, middle_x, height), self.depth+1)
        bottom_right = Quadrant(points, (middle_x, middle_y, width, height), self.depth+1)

        # add new quadrants to root children
        self.children = [upper_left, upper_right, bottom_left, bottom_right]

    def check_lines(self, points):
        # Check detail of each region, if both is below threshold then set line
        # TODO instead of using detail instead construct a new image based off avg pixel value in region
        # Then calculate L2 error with original image, if below a certain threshold then use.

        # Do not use even numbers here or get divide by zero error
        # TODO: Look up geometric alignment from Wedgelet Paper
        num_angles = 21
        num_lens = 10
        left, top, width, height = self.bbox
        points = points[top:height, left:width]

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False)
        tested_len = np.linspace(-points.shape[1]/2, points.shape[1]/2, num_lens, endpoint=False)
        lens, angles = np.meshgrid(tested_len, tested_angles)

        min_h_mean = 1000
        best_len = 0
        best_angle = 0
        best_p1_avg = 0
        best_p2_avg = 0

        for a in range(num_angles):
            for l in range(num_lens):
                p1, p2 = cut_points(points, lens[a][l], angles[a][l])
                p1_len = len(p1)
                p2_len = len(p2)
                if p1_len == 0 or p2_len == 0:
                    continue
                p1_avg = np.mean(p1)
                p2_avg = np.mean(p2)
                p1_h = get_homogeneity(p1, p1_avg)
                p2_h = get_homogeneity(p2, p2_avg)
                h_mean = p1_h * (p1_len / points.size) + p2_h * (p2_len / points.size)
                if h_mean < min_h_mean:
                    min_h_mean = h_mean
                    best_len = lens[a][l]
                    best_angle = angles[a][l]
                    best_p1_avg = p1_avg
                    best_p2_avg = p2_avg
                # print(f"p1 len: {p1_len}, p2 len: {p2_len}, size: {points.size}")
                # print(f"p1 mean: {p1_avg}, p2 mean: {p2_avg}, p1 h: {p1_h}, p2 h: {p2_h}")
                # if p1_h <= HOMOGENEITY_THRESHOLD and p2_h <= HOMOGENEITY_THRESHOLD:
                #     print(f"Cut Found at depth: {self.depth}")

                # if h_mean <= HOMOGENEITY_THRESHOLD:
                #     print(f"Cut Found at depth: {self.depth}, bbox: {self.bbox}")
                # if np.mean([p1_h, p2_h]) <= HOMOGENEITY_THRESHOLD:
                #     print("Cut Found")
        return min_h_mean, best_len, best_angle, best_p1_avg, best_p2_avg
        # if min_h_mean <= HOMOGENEITY_THRESHOLD:
        #     print(f"min h mean: {min_h_mean}, bbox: {self.bbox}, normal h: {self.homogeneity}, p: {best_len}, theta: {best_angle}")