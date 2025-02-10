import numpy as np

from PIL import Image
from pykrige.ok import OrdinaryKriging
from skimage import io

CROP_SIZE = 150
NUM_SAMPLES = 2000
IMG_WIDTH = 0
IMG_HEIGHT = 0

def sample_orthomosaic(image):
    global IMG_WIDTH, IMG_HEIGHT
    x = np.random.randint(low=CROP_SIZE, high=IMG_WIDTH-CROP_SIZE+1, size=NUM_SAMPLES)
    y = np.random.randint(low=CROP_SIZE, high=IMG_HEIGHT-CROP_SIZE+1, size=NUM_SAMPLES)
    weed_chance = np.zeros(NUM_SAMPLES)

    for i in range(NUM_SAMPLES):
        crop = image.crop((x[i], y[i], x[i]+CROP_SIZE, y[i]+CROP_SIZE))
        weed_chance[i] = np.count_nonzero(np.array(crop)[:, :, 0]) / CROP_SIZE**2
    
    return x, y, weed_chance

def test_all_reps(map_name, path):
    global IMG_WIDTH, IMG_HEIGHT
    # im = Image.open("000/000/groundtruth/first000_gt.png")
    im = io.imread(path)
    im = Image.fromarray(im)
    IMG_WIDTH, IMG_HEIGHT = im.size

    x, y, weed_chance = sample_orthomosaic(im)

    OK = OrdinaryKriging(
        x,
        y,
        weed_chance,
        variogram_model='exponential',
        verbose=True,
        enable_statistics=True,
    )
    OK.print_statistics()

NUM_REPEATS = 1

maps = {
    "004": "RedEdge/RedEdge/004/groundtruth/second002_gt.tif",
    "003": "RedEdge/RedEdge/003/groundtruth/second001_gt.tif",
    "002": "RedEdge/RedEdge/002/groundtruth/second000_gt.tif",
    "000": "000/000/groundtruth/first000_gt.png",
    "001": "RedEdge/RedEdge/001/groundtruth/first001_gt.tif",
}

if __name__ == "__main__":
   for i in range(NUM_REPEATS):
        for map_name, path in maps.items():
            test_all_reps(map_name, path)
            print(f"Completed iter: {i}, of map: {map_name}")