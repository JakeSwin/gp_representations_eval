from pykrige.ok import OrdinaryKriging
from scipy.spatial import ConvexHull
from shapely.ops import triangulate
from PIL import Image, ImageDraw
from itertools import chain
from lib.bsp_lse.draw import draw_bsp_lse

import matplotlib.pyplot as plt
import shapely.geometry as sg
import gstools as gs
import numpy as np
import random

width = 0
height = 0
thetas = None
zstar = None

min_x_graph = 0
min_y_graph = 0
max_x_graph = 0
max_y_graph = 0

SAMPLING_STEP = 10

def plot_corner_points(points):
  global min_x_graph, max_x_graph, min_y_graph, max_y_graph, zstar
  points = np.asarray(points)
  ch = ConvexHull(points)

  plt.figure(figsize=(5,5))
  plt.xlim(min_x_graph, max_x_graph)
  plt.ylim(min_y_graph, max_y_graph)

  for simplex in ch.simplices:
    plt.plot(points[simplex,0], points[simplex, 1], linewidth=3)

  plt.imshow(zstar, extent=(min_x_graph, max_x_graph, min_y_graph, max_y_graph), origin='upper')

def plot_min_and_maxes(min_and_maxes, thetas):
  scatter_coords = np.asarray([[(theta, min), (theta, max)] for ((min, max), theta) in zip(min_and_maxes, thetas)]).reshape(len(thetas) * 2,2)
  plt.scatter(scatter_coords[:, 0], scatter_coords[:, 1])

def p_quantiser(theta):
  # Returns the quantisation step for p
  return np.max([np.abs(np.cos(theta)), np.abs(np.sin(theta))])

def get_num_p_steps(p_min, p_max, theta):
  # Gets the quantisation step for p
  min_j = np.ceil(p_min / p_quantiser(theta))
  max_j = np.floor(p_max / p_quantiser(theta))
  return np.int32(max_j - min_j + 1), np.int32(min_j), np.int32(max_j)

def get_all_params(num_p_steps):
  global thetas
  all_params = [[[thetas[i], j * p_quantiser(thetas[i])] for j in np.linspace(min, max, num_steps)] for i, (num_steps, min, max) in enumerate(num_p_steps)]
  param_list_flat = []

  for param_list in all_params:
    for parameters in param_list:
      param_list_flat.append(parameters)

  return np.asarray(param_list_flat)

def get_intersection_points(corner_points, line_params):
  # Note that line_params should be in the format (theta, p)

  # Get all the line segments from combinations of corner points
  line_segments_idx = ConvexHull(corner_points).simplices

  corner_points = np.asarray(corner_points)

  min_x, max_x = np.min(corner_points[:, 0]), np.max(corner_points[:, 0])
  min_y, max_y = np.min(corner_points[:, 1]), np.max(corner_points[:, 1])

  # Gets the two points for our line segment from line equation, sampling from outside the bounds of the polygon
  if np.sin(line_params[0]) < 0.0000001 and np.sin(line_params[0]) > 0.0000001:
    return None

  p1 = (min_x-1, (line_params[1] - (min_x-1) * np.cos(line_params[0])) / np.sin(line_params[0]))
  p2 = (max_x+1, (line_params[1] - (max_x+1) * np.cos(line_params[0])) / np.sin(line_params[0]))

  intersection_points = []
  for p_seg1, p_seg2 in line_segments_idx:
      # x1, y1, x2, y2 are coordinates for box line segment
      x1, y1 = corner_points[p_seg1]
      x2, y2 = corner_points[p_seg2]
      # x3, y3, x4, y4 are coordinates for line equation segment
      x3, y3 = p1
      x4, y4 = p2
      # Calculate t
      # print(f"p_seg1: {p_seg1}, p_seg2: {p_seg2}")
      # print(f"p1: {p1}, p2: {p2}")
      t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
      # print(f"t: {t}")
      # Stop calculation if no intersection
      if (t < 0 or t > 1):
          continue
      else:
          ix, iy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
          intersection_points.append((ix, iy))
  if len(intersection_points) != 2:
    # print(intersection_points)
    return None

  return intersection_points

def calc_y(line_params, x):
  return (line_params[1] - x * np.cos(line_params[0])) / np.sin(line_params[0])

def get_points_to_test_for_lpl(line_params, intersection_points):
  if intersection_points is None:
    return None
  if np.sin(line_params[0]) < 0.0000001 and np.sin(line_params[0]) > -0.0000001:
    return None
  intersection_points = np.asarray(intersection_points)
  x1, x2 = intersection_points[:,0]
  min_y = np.min(intersection_points[:,1])
  max_y = np.max(intersection_points[:,1])

  Xs = np.linspace(x1, x2, SAMPLING_STEP)
  Ys = np.asarray([calc_y(line_params, x) for x in Xs])
  if (Ys < (min_y-1)).any() or (Ys > (max_y+1)).any():
    print(f"Xs: {Xs}")
    print(f"Ys: {Ys}")
    print(f"min_y {min_y}, max_y {max_y}")
    return None

  return Xs, Ys

def get_points_to_test(corner_points, params, intersection_points):
  points_to_test = []
  error_points_idx = []

  min_y, max_y = np.min(np.asarray(corner_points)[:, 1]), np.max(np.asarray(corner_points)[:, 1])

  for i in range(len(params)):
    x, y = get_points_to_test_for_lpl(params[i], intersection_points[i])
    if (y < min_y).any():
      # print(f"Index {i}, is invalid, skipping")
      error_points_idx.append(i)
      continue
    if (y > max_y).any():
      # print(f"Index {i}, is invalid, skipping")
      error_points_idx.append(i)
      continue
    points_to_test.append([x, y])

  return np.asarray(points_to_test), error_points_idx

def get_Zs(points, batch_size=15000):
  points = np.asarray(points)
  Xs = points[:, 0, :].reshape((-1))
  Ys = points[:, 1, :].reshape((-1))
  h, w = zstar.shape
  print(f"min x: {np.min(Xs)}, max x: {np.max(Xs)}, min y: {np.min(Ys)}, max y: {np.max(Ys)}")

  x_transform = np.asarray(Xs) / max_x_graph
  y_transform = np.asarray(Ys) / max_y_graph
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  x_transform = x_transform * ((w / 2) - 1)
  y_transform = y_transform * ((h / 2) - 1)
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  x_transform = x_transform + ((w / 2) - 1)
  y_transform = y_transform + ((h / 2) - 1)
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  x_transform = x_transform.astype(np.int32)
  y_transform = (h - y_transform - 1).astype(np.int32)
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  z = []

  for x, y in zip(x_transform, y_transform):
    z.append(zstar[y, x])

  return z

def get_candidate_mask(points, Zs, threshold):
  global max_x_graph, width
  points = np.asarray(points)
  Xs = (points[:, 0, :].reshape((-1)) / max_x_graph) * (width / 2)
  candidates = []
  for i in range(int(len(Zs) / SAMPLING_STEP)):
    start = i * SAMPLING_STEP
    end = (i+1) * SAMPLING_STEP
    candidates.append(np.abs((np.sum(Zs[start:end] * Xs[start:end]) / np.sum(Zs[start:end])) - ((Xs[end-1] + Xs[start]) / 2)))
  candidates = np.asarray(candidates)
  return candidates < threshold

def get_samples_in_region(corner_points):
  global max_x_graph, max_y_graph
  min_x, max_x = np.min(np.asarray(corner_points)[:, 0]), np.max(np.asarray(corner_points)[:, 0])
  min_y, max_y = np.min(np.asarray(corner_points)[:, 1]), np.max(np.asarray(corner_points)[:, 1])
  h, w = zstar.shape
  width_of_region = max_x - min_x
  height_of_region = max_y - min_y

  # Give more samples for larger regions
  samples_for_region = int((width_of_region * height_of_region) / 3)
  x = np.random.randint(low=min_x, high=max_x, size=samples_for_region)
  y = np.random.randint(low=min_y, high=max_y, size=samples_for_region)

  # Define the polygon
  polygon = sg.Polygon(corner_points)

  # Get Triangles
  triangles = triangulate(polygon)
  print(triangles)

  is_in_triangles = lambda x, y: [t.contains(sg.Point(x,y)) for t in triangles]

  is_point_in_triangles = [is_in_triangles(xi, yi) for xi, yi in zip(x,y)]

  x_filtered = []
  y_filtered = []
  x_removed = []
  y_removed = []

  for i, v in enumerate(is_point_in_triangles):
    # print(v)
    if v == [False] * len(triangles):
      # print(f"filtering out index: {i}, got: {v}")
      x_removed.append(x[i])
      y_removed.append(y[i])
      continue
    x_filtered.append(x[i])
    y_filtered.append(y[i])

  x_transform = np.asarray(x_filtered) / max_x_graph
  y_transform = np.asarray(y_filtered) / max_y_graph
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  x_transform = x_transform * np.floor((w / 2))
  y_transform = y_transform * np.floor((h / 2))
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  x_transform = x_transform + np.floor((w / 2))
  y_transform = y_transform + np.floor((h / 2))
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  x_transform = x_transform.astype(np.int32)
  y_transform = (h - y_transform - 1).astype(np.int32)
  print(f"min x: {np.min(x_transform)}, max x: {np.max(x_transform)}, min y: {np.min(y_transform)}, max y: {np.max(y_transform)}")

  z = []

  for x, y in zip(x_transform, y_transform):
    z.append(zstar[y, x])

  return x_filtered, y_filtered, np.asarray(z)

def get_regions(x_points, y_points, line_params):
  x_points = np.array(x_points)
  y_points = np.array(y_points)

  if len(x_points) != len(y_points):
    return None

  above_idx = np.array([], dtype=np.int32)
  below_idx = np.array([], dtype=np.int32)

  for x in np.unique(x_points):
    y_idxs = np.where(x_points == x)[0]
    ys = y_points[y_idxs]
    y_from_eq = (line_params[1] - (x * np.cos(line_params[0]))) / np.sin(line_params[0])
    above_idx = np.append(above_idx, y_idxs[np.where(ys < y_from_eq)[0]])
    below_idx = np.append(below_idx, y_idxs[np.where(ys >= y_from_eq)[0]])

  if len(above_idx) == 0 or len(below_idx) == 0:
    return None

  return above_idx, below_idx

def get_error(x, y, z, params):
  regions = get_regions(x, y, params)
  if regions == None:
    return np.Inf
  above, below = regions
  avg_above = np.mean(z[above])
  avg_below = np.mean(z[below])

  e_above = np.sum([np.square(zi - avg_above) for zi in z[above]])
  e_below = np.sum([np.square(zi - avg_below) for zi in z[below]])

  return e_above + e_below

def get_attributes(x, y, z, params):
  regions = get_regions(x, y, params)
  if regions == None:
    return np.Inf
  above, below = regions
  avg_above = np.mean(z[above])
  avg_below = np.mean(z[below])

  e_above = np.sum([np.square(zi - avg_above) for zi in z[above]])
  e_below = np.sum([np.square(zi - avg_below) for zi in z[below]])

  return avg_above, avg_below, e_above, e_below

import time

def plot_intersection_line(corner_points, param):
  global min_x_graph, max_x_graph, min_y_graph, max_y_graph, zstar

  bounding_box_corner_points = [
    (min_x_graph, max_y_graph),
    (max_x_graph, max_y_graph),
    (max_x_graph, min_y_graph),
    (min_x_graph, min_y_graph),
  ]
  p1, p2 = get_intersection_points(bounding_box_corner_points, param)
  i1, i2 = get_intersection_points(corner_points, param)
  plot_corner_points(corner_points, zstar)
  plt.plot(np.asarray((p1,p2))[:, 0], np.asarray((p1,p2))[:, 1], "r--", linewidth=2)
  plt.plot(np.asarray((i1,i2))[:, 0], np.asarray((i1,i2))[:, 1], "or")

def plot_all_intersection_lines(corner_points, params):
  global min_x_graph, max_x_graph, min_y_graph, max_y_graph, zstar

  bounding_box_corner_points = [
    (min_x_graph, max_y_graph),
    (max_x_graph, max_y_graph),
    (max_x_graph, min_y_graph),
    (min_x_graph, min_y_graph),
  ]

  plot_corner_points(corner_points, zstar)

  for param in params:
    p1, p2 = get_intersection_points(bounding_box_corner_points, param)
    plt.plot(np.asarray((p1,p2))[:, 0], np.asarray((p1,p2))[:, 1], "r--", linewidth=1)

def divide_best_regions(region):
  global thetas
  # Get range of parameters for provided region
  start = time.time()
  functions = [lambda theta, x=x, y=y: x * np.cos(theta) + y * np.sin(theta) for (x, y) in region]
  min_and_maxes = [(np.min(l), np.max(l)) for l in [[f(theta) for f in functions] for theta in thetas]]
  end = time.time()
  print(f"Time to Get range of parameters for provided region: {end - start}")

  # Get total number of parameters of intersecting lines for region
  start = time.time()
  num_p_steps = np.asarray([get_num_p_steps(*min_and_maxes[i], thetas[i]) for i in range(len(thetas))])
  all_params = get_all_params(num_p_steps)
  end = time.time()
  print(f"Time to Get total number of parameters of intersecting lines for region: {end - start}")

  # Find all intersection points
  start = time.time()
  all_intersection_points = [get_intersection_points(region, line_params) for line_params in all_params]
  end = time.time()
  print(f"Time to Find all intersection points: {end - start}")

  # Remove parameters with invalid intersections
  start = time.time()
  error_params_index = [i for i, v in enumerate(all_intersection_points) if v == None]
  params_pruned = np.delete(all_params, error_params_index, axis=0)
  intersection_points_pruned = [ip for ip in all_intersection_points if ip != None]
  end = time.time()
  print(f"Time to Remove parameters with invalid intersections: {end - start}")

  # Get all the points to test for LPL transform pruning
  start = time.time()
  points_to_test, error_points_idx = get_points_to_test(region, params_pruned, intersection_points_pruned)
  end = time.time()
  print(f"Time to Get all the points to test for LPL transform pruning: {end - start}")

  # Remove parameters with invalid test points
  start = time.time()
  params_pruned = np.delete(params_pruned, error_points_idx, axis=0)
  intersection_points_pruned = np.delete(intersection_points_pruned, error_points_idx, axis=0)
  end = time.time()
  print(f"Time to Remove parameters with invalid test points: {end - start}")

  # Perform GP inference for every point to test, Batched
  start = time.time()
  Zs = get_Zs(points_to_test, 25000)
  end = time.time()
  print(f"Time to Perform GP inference for every point to test, Batched: {end - start}")

  # Get candidate parameters based on LPL transform
  start = time.time()
  candidate_mask = get_candidate_mask(points_to_test, Zs, 30)
  candidate_params = params_pruned[candidate_mask]
  end = time.time()
  print(f"Time to Get candidate parameters based on LPL transform: {end - start}")

  # Get samples in region
  start = time.time()
  x_region_samples, y_region_samples, z_region_samples = get_samples_in_region(region)
  end = time.time()
  print(f"Time to Get samples in region: {end - start}")

  # Get error for candidate parameters based on dividing samples in region
  start = time.time()
  errors = [get_error(x_region_samples, y_region_samples, z_region_samples, params) for params in candidate_params]
  end = time.time()
  print(f"Time to Get error for candidate parameters based on dividing samples in region: {end - start}")

  # Choose parameter with LSE
  start = time.time()
  best_param_index = np.argsort(errors)[0]
  best_param = candidate_params[best_param_index]
  end = time.time()
  print(f"Time to Choose parameter with LSE: {end - start}")

  # Get attiributes
  start = time.time()
  avg_above, avg_below, e_above, e_below = get_attributes(x_region_samples, y_region_samples, z_region_samples, best_param)
  end = time.time()
  print(f"Time to Get attiributes: {end - start}")

  # Get the next polygon regions based on the chosen dividing line
  start = time.time()
  above_region_idx, below_region_idx = get_regions(np.asarray(region)[:,0], np.asarray(region)[:,1], best_param)
  p1, p2 = get_intersection_points(region, candidate_params[best_param_index])
  above_polygon = np.asarray(region)[above_region_idx]
  below_polygon = np.asarray(region)[below_region_idx]
  above_polygon = np.append(above_polygon, [p1, p2], axis=0)
  below_polygon = np.append(below_polygon, [p1, p2], axis=0)
  end = time.time()
  print(f"Time to Get the next polygon regions based on the chosen dividing line: {end - start}")

  return best_param, candidate_params, avg_above, avg_below, e_above, e_below, above_polygon, below_polygon

STOPPING_THRESHOLD = 0.0001
MAX_DEPTH = 9

class BSPNode:
  def __init__(self, region, depth=0, error=np.Inf, avg_value=0):
    self.region = region

    self.above_child = None
    self.below_child = None

    self.error = error
    self.avg_value = avg_value

    self.depth = depth
    self.leaf = False
    self.param = None

    self.divide()

  def divide(self):
    if self.error < STOPPING_THRESHOLD or self.depth >= MAX_DEPTH:
      self.leaf = True
    else:
      try:
        best_param, _, avg_above, avg_below, e_above, e_below, above_polygon, below_polygon = divide_best_regions(self.region)
        self.param = best_param
        
        self.above_child = BSPNode(above_polygon, self.depth+1, e_above, avg_above)
        self.below_child = BSPNode(below_polygon, self.depth+1, e_below, avg_below)
      except:
        self.leaf = True

def make_bsp_lse(z, w, h):
  global min_x_graph, max_x_graph, min_y_graph, max_y_graph, zstar, thetas, width, height

  zstar = z
  width = w
  height = h
  image_ratio = width / height
  resolution_factor = 100
  r_width, r_height = image_ratio * resolution_factor, resolution_factor
  quantisation_step = np.arctan(2/r_width)
  num_of_thetas = int(np.pi / quantisation_step)
  thetas = np.linspace(np.pi-quantisation_step, 0, num_of_thetas, endpoint=False)[::-1]
  min_x_graph, max_x_graph = -r_width/2, r_width/2
  min_y_graph, max_y_graph = -r_height/2, r_height/2
  
  corner_points = [
      (min_x_graph, max_y_graph),
      (max_x_graph, max_y_graph),
      (max_x_graph, min_y_graph),
      (min_x_graph, min_y_graph),
  ]

  BSP = BSPNode(corner_points)
  BSP_image = draw_bsp_lse(BSP, z.shape[1], z.shape[0])
  return BSP, BSP_image