import numpy as np

from lib.quadtree.utils import get_homogeneity

class Quadrant():
    def __init__(self, points, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

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

    # def check_lines(self, image):
    #     # Check detail of each region, if both is below threshold then set line
    #     # TODO instead of using detail instead construct a new image based off avg pixel value in region
    #     # Then calculate L2 error with original image, if below a certain threshold then use.
    #     np_img = np.asarray(image)
    #     t_mask = np.empty(np_img.shape, dtype=np.uint8)
    #     for mask in MASKS:
    #         region1 = np_img[~mask == 255]
    #         col_r1 = np.array(np.average(region1, axis=0), dtype=np.uint8)
    #         t_mask[mask==255] = col_r1
    #         region2 = np_img[~mask == 0]
    #         col_r2 = np.array(np.average(region2, axis=0), dtype=np.uint8)
    #         t_mask[mask==0] = col_r2
    #         L2_error = np.sqrt(np.sum((np_img - t_mask)**2))
    #         TOTAL_L2.append(L2_error)
    #         print(f"Min L2: {np.min(TOTAL_L2)}")