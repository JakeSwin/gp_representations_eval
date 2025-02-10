from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from itertools import chain

import numpy as np

def interpolate(color1, color2, fraction):
    return [int(c1 + (c2 - c1) * fraction) for c1, c2 in zip(color1, color2)]

COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
COLOUR_PALETTE = list(chain.from_iterable([interpolate(COLOUR_1, COLOUR_2, i / (256 - 1)) for i in range(256)]))

def recursive_search(node, append_leaf):
  if node.leaf:
    append_leaf(node)
  else:
    recursive_search(node.above_child, append_leaf)
    recursive_search(node.below_child, append_leaf)

def get_leaf_nodes(bsp_tree):
  nodes = []
  recursive_search(bsp_tree, nodes.append)
  return nodes

def draw_bsp_lse(BSP, width, height):
  image = Image.new('P', (width, height))
  image.putpalette(COLOUR_PALETTE)
  draw = ImageDraw.Draw(image)
  leaf_nodes = get_leaf_nodes(BSP)
  root_region = np.asarray(BSP.region)
  max_x_graph = np.max(root_region[:, 0])
  max_y_graph = np.max(root_region[:, 1])

  avg_values = [leaf.avg_value for leaf in leaf_nodes]
  
  for leaf in leaf_nodes:
    region = np.asarray(leaf.region)
    x_region_transform = np.int32(((region[:,0] / max_x_graph) * (width / 2)) + (width / 2))
    y_region_transform = height - np.int32(((region[:,1] / max_y_graph) * (height / 2)) + (height / 2)) - 1
    
    draw_points = np.transpose([x_region_transform, y_region_transform])
    
    ch = ConvexHull(draw_points)
    
    norm_value = (leaf.avg_value - np.min(avg_values)) / (np.max(avg_values) - np.min(avg_values))
    
    draw.polygon([(x,y) for x,y in draw_points[ch.vertices]], int(norm_value * 255))

  im = image.convert("RGB")
  return im