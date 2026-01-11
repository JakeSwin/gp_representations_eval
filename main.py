import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import imagehash
import time
import jax
import jax.numpy as jnp
import sys
import csv
matplotlib.use('Qt5Agg')

from PIL import Image
from pykrige.ok import OrdinaryKriging
from skimage.metrics import structural_similarity
from itertools import chain
from skimage import io

from lib.h3.h3 import make_h3
from lib.bsp_lse.bsp_lse import make_bsp_lse
from lib.bsp_region.bsp_region import make_bsp_region
from lib.quadtree.quadtree import QuadTree
from lib.wedgelet.wedgelet import Wedgelet
from lib.voronoi.voronoi_fix import Voronoi, lbg_step, fit_voronoi_lbg

from lib.bsp_lse.utils import recursive_search_size_bsp

def interpolate(color1, color2, fraction):
    return [int(c1 + (c2 - c1) * fraction) for c1, c2 in zip(color1, color2)]

COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
COLOUR_PALETTE = list(chain.from_iterable([interpolate(COLOUR_1, COLOUR_2, i / (256 - 1)) for i in range(256)]))

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

def sample_GP_small(KrigingObj):
    global IMG_WIDTH, IMG_HEIGHT
    gridx = np.arange(0, IMG_WIDTH, 35, dtype='float64')
    gridy = np.arange(0, IMG_HEIGHT, 35, dtype='float64')
    zstar, ss = KrigingObj.execute("grid", gridx, gridy)
    return zstar, ss

def sample_GP(KrigingObj):
    global IMG_WIDTH, IMG_HEIGHT
    batch_rows = 30
    gridx = np.arange(0, IMG_WIDTH, 10, dtype='float64')
    gridy = np.arange(0, IMG_HEIGHT, 10, dtype='float64')
    num_iters = int(len(gridy) / batch_rows)
    zstar  = np.zeros((len(gridy), len(gridx)))

    for i in range(num_iters):
        start_idx = i * batch_rows
        end_idx = (i+1) * batch_rows
        if i == num_iters-1:
            end_idx = len(gridy)

        z, _ = KrigingObj.execute("grid", gridx, gridy[start_idx:end_idx])
        zstar[start_idx:end_idx, :] = z
    return zstar

def get_data_range(img1, img2):
  min_img1 = np.min(img1)
  min_img2 = np.min(img2)
  max_img1 = np.max(img1)
  max_img2 = np.max(img2)
  return np.max([max_img1, max_img2]) - np.min([min_img1, min_img2])

def normalise(img):
  return (img - np.min(img)) / (np.max(img) - np.min(img))

def format_as_255(img):
  normalised = normalise(img)
  return (normalised * 255 / np.max(normalised)).astype('uint8')

def image_mse(img1, img2):
   im1 = np.array(img1)
   im2 = np.array(img2)
   diff = im1 - im2
   squared_diff = diff ** 2
   mse = np.mean(squared_diff)
   return mse

def get_size(obj):
    """Recursively calculate the size of an object"""
    total_size = sys.getsizeof(obj)
    if hasattr(obj, '__dict__'):  # Check if object has attributes
        for key, value in obj.__dict__.items():
            total_size += get_size(value)
    elif hasattr(obj, '__iter__'):  # Check if object is iterable
        if hasattr(obj, 'keys'):  # Check if object is a dictionary
            for key in obj:
                total_size += get_size(obj[key])
        elif not isinstance(obj, str):  # Ignore strings
            for item in obj:
                total_size += get_size(item)
    return total_size

# TODO Write functions for each tree to calculate the size of that structure

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
    )

    start = time.time()
    zstar = sample_GP(OK)
    normalised = (zstar.data - np.min(zstar.data)) / (np.max(zstar.data) - np.min(zstar.data))
    end = time.time()
    print(f"Time to Create Gridmap: {end - start}")
    gridmap_size = get_size(zstar)

    # start = time.time()
    # bsp, bsp_image = make_bsp_lse(zstar, IMG_WIDTH, IMG_HEIGHT)
    # end = time.time()
    # bsp_time = end - start
    # print(f"Time to Create BSP: {bsp_time}")
    # # print(f"Size of BSP: {recursive_search_size_bsp(bsp)}")
    # bsp_size = get_size(bsp)
    # print(f"Size of BSP: {bsp_size}")
    # bsp_image.save("bsp.png")
    # bsp_image = bsp_image.convert('L')

    # zstar_s, _ = sample_GP_small(OK)
    # start = time.time()
    # bsp_r_t, bsp_r_a, bsp_r_image = make_bsp_region(zstar_s.data, zstar.shape[1], zstar.shape[0])
    # end = time.time()
    # bsp_r_time = end - start
    # print(f"Time to Create BSP Region: {bsp_r_time}")
    # bsp_r_size = get_size(bsp_r_t) + get_size(bsp_r_a)
    # print(f"Size of BSP Region {bsp_r_size}")
    # bsp_r_image.save("bsp_r.png")
    # bsp_r_image = bsp_r_image.convert('L')

    # start = time.time()
    # hexs, weed_chance, h3_img = make_h3(np.flip(zstar,axis=0))
    # end = time.time()
    # h3_time = end - start
    # print(f"Time to Create H3: {h3_time}")
    # h3_size = get_size(hexs) + get_size(weed_chance)
    # print(f"Size of H3 Rep: {h3_size}")
    # h3_img.save("h3.png")
    # h3_img = h3_img.convert('L')

    # start = time.time()
    # quadtree = QuadTree(normalised)
    # end = time.time()
    # quadtree_time = end - start
    # print(f"Time to Create Quadtree: {quadtree_time}")
    # quadtree_size = get_size(quadtree)
    # print(f"Size of Quadtree: {quadtree_size}")
    # quadtree_image = quadtree.create_image(6, show_lines=False)
    # quadtree_image.save("quadtree.png")
    # quadtree_image = quadtree_image.convert('L')

    # start = time.time()
    # wedgelet = Wedgelet(normalised)
    # end = time.time()
    # wedgelet_time = end - start
    # print(f"Time to Create Wedgelet: {wedgelet_time}")
    # wedgelet_size = get_size(wedgelet)
    # print(f"Size of Wedgelet: {wedgelet_size}")
    # Wedgelet_image = wedgelet.create_image(6, show_lines=False)
    # Wedgelet_image.save("wedgelet.png")
    # Wedgelet_image = Wedgelet_image.convert('L')

    # Voronoi
    start = time.time()
    key = jax.random.PRNGKey(0)
    num_samples = 500
    with jax.default_device(jax.devices("cpu")[0]):
        seeds = (jax.random.uniform(key, shape=(num_samples, 2))* jnp.array([zstar.shape[1], zstar.shape[0]])).astype(jnp.int32)
    vr = Voronoi(height=zstar.shape[0], width=zstar.shape[1], seeds=seeds)          # CHANGED
    jfa_map = vr.jfa()
    new_seeds, new_num_seeds, jfa_map = fit_voronoi_lbg(seeds, num_samples, vr, normalised)
    end = time.time()
    voronoi_time = end - start
    print(f"Time to Create voronoi: {voronoi_time}")
    new_seeds_arr    = np.array(new_seeds, dtype=np.int32)  # or whatever you use
    fake_avg_arr = np.zeros((new_seeds_arr.shape[0],)).astype(np.float32)
    voronoi_size      = get_size(new_seeds_arr) + get_size(fake_avg_arr)
    index_map = vr.get_index_map(jfa_map, new_seeds)
    colour_palette = vr.create_colour_palette(index_map, zstar)
    colour_map = vr.get_colour_map(index_map, colour_palette)
    npcol = np.array(colour_map).astype(np.uint8)
    voro_im = Image.fromarray(npcol).convert("RGB")
    voro_im.save(f"voro_{map_name}.png")
    voro_im = voro_im.convert("L")

    # struc_similarity_bsp = structural_similarity(normalise(zstar), normalise(bsp_image), data_range=get_data_range(zstar,bsp_image))
    # print(f"Structural Simularity BSP: {1 - struc_similarity_bsp}")
    # struc_similarity_bsp_r = structural_similarity(normalise(zstar), normalise(bsp_r_image), data_range=get_data_range(zstar,bsp_r_image))
    # print(f"Structural Simularity BSP Region: {1 - struc_similarity_bsp_r}")
    # struc_similarity_h3 = structural_similarity(normalise(zstar), normalise(h3_img), data_range=get_data_range(zstar,h3_img))
    # print(f"Structural Simularity H3: {1 - struc_similarity_h3}")
    # struc_similarity_quadtree = structural_similarity(normalise(zstar), normalise(quadtree_image), data_range=get_data_range(zstar,quadtree_image))
    # print(f"Structural Simularity Quadtree: {1 - struc_similarity_quadtree}")
    # struc_similarity_wedgelet = structural_similarity(normalise(zstar), normalise(Wedgelet_image), data_range=get_data_range(zstar,Wedgelet_image))
    # print(f"Structural Simularity Wedgelet: {1 - struc_similarity_wedgelet}")
    # Voronoi
    struc_similarity_voronoi = structural_similarity(normalise(zstar), normalise(voro_im), data_range=get_data_range(zstar,voro_im))
    print(f"Structural Simularity Voronoi: {1 - struc_similarity_voronoi}")

    zstar_img = Image.fromarray(format_as_255(zstar))
    hash1 = imagehash.phash(zstar_img, hash_size=64, highfreq_factor=6)

    # hash2 = imagehash.phash(bsp_image, hash_size=64, highfreq_factor=6)
    # bsp_hd = hash1 - hash2
    # print(f"Hash Hamming Distance BSP: {bsp_hd}")
    # hash2 = imagehash.phash(bsp_r_image, hash_size=64, highfreq_factor=6)
    # bsp_r_hd = hash1 - hash2
    # print(f"Hash Hamming Distance BSP Region: {bsp_r_hd}")
    # hash2 = imagehash.phash(h3_img, hash_size=64, highfreq_factor=6)
    # h3_hd = hash1 - hash2
    # print(f"Hash Hamming Distance H3: {h3_hd}")
    # hash2 = imagehash.phash(quadtree_image, hash_size=64, highfreq_factor=6)
    # quadtree_hd = hash1 - hash2
    # print(f"Hash Hamming Distance Quadtree: {quadtree_hd}")
    # hash2 = imagehash.phash(Wedgelet_image, hash_size=64, highfreq_factor=6)
    # wedgelet_hd = hash1 - hash2
    # print(f"Hash Hamming Distance Wedgelet: {wedgelet_hd}")
    hash2 = imagehash.phash(voro_im, hash_size=64, highfreq_factor=6)
    voronoi_hd = hash1 - hash2
    print(f"Hash Hamming Distance Voronoi: {voronoi_hd}")

    # MSE metric
    # mse_bsp = image_mse(zstar_img, bsp_image)
    # print(f"MSE BSP: {mse_bsp}")
    # mse_r_bsp = image_mse(zstar_img, bsp_r_image)
    # print(f"MSE BSP Region: {mse_r_bsp}")
    # mse_h3 = image_mse(zstar_img, h3_img)
    # print(f"MSE H3: {mse_h3}")
    # mse_quadtree = image_mse(zstar_img, quadtree_image)
    # print(f"MSE Quadtree: {mse_quadtree}")
    # mse_wedgelet = image_mse(zstar_img, Wedgelet_image)
    # print(f"MSE Wedgelet: {mse_wedgelet}")
    mse_voronoi = image_mse(zstar_img, voro_im)
    print(f"MSE voronoi: {mse_voronoi}")

    zstar_img.putpalette(COLOUR_PALETTE)
    zstar_img.convert("RGB").save("zstar.png")

    with open(f'results/{map_name}_results.csv', 'a', newline='') as csvfile:
        # Create a csv.writer object
        writer = csv.writer(csvfile)

        # Append a new row
        writer.writerow([
            struc_similarity_voronoi, voronoi_hd, mse_voronoi
        ])

    # bsp_time,bsp_space,bsp_r_time,bsp_r_space,h3_time,h3_space,quadtree_time,quadtree_space,wedgelet_time,wedgelet_space
    with open(f'results/{map_name}_timespace.csv', 'a', newline='') as csvfile:
        # Create a csv.writer object
        writer = csv.writer(csvfile)

        # Append a new row
        writer.writerow([
            voronoi_time, voronoi_size
        ])

NUM_REPEATS = 5

maps = {
    "004": "RedEdge/004/groundtruth/second002_gt.tif",
    "003": "RedEdge/003/groundtruth/second001_gt.tif",
    "002": "RedEdge/002/groundtruth/second000_gt.tif",
    "000": "RedEdge/000/groundtruth/first000_gt.png",
    "001": "RedEdge/001/groundtruth/first001_gt.tif",
}

if __name__ == "__main__":
   for i in range(NUM_REPEATS):
        for map_name, path in maps.items():
            test_all_reps(map_name, path)
            print(f"Completed iter: {i}, of map: {map_name}")
