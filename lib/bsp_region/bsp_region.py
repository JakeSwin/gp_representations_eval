import PIL.Image as Image
import numpy as np
import higra as hg

from itertools import chain

def interpolate(color1, color2, fraction):
    return [int(c1 + (c2 - c1) * fraction) for c1, c2 in zip(color1, color2)]

COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
COLOUR_PALETTE = list(chain.from_iterable([interpolate(COLOUR_1, COLOUR_2, i / (256 - 1)) for i in range(256)]))

def is_non_relevant(tree, altitudes):
    area = hg.attribute_area(tree)
    min_area_children = hg.accumulate_parallel(tree, area, hg.Accumulators.min)
    return min_area_children <= 10

def normalise(img):
  return (img - np.min(img)) / (np.max(img) - np.min(img))

def format_as_255(img):
  normalised = normalise(img)
  return (normalised * 255 / np.max(normalised)).astype('uint8')

def make_bsp_region(zstar, width, height):
    graph = hg.get_4_adjacency_graph(zstar.shape)
    edge_weights = hg.weight_graph(graph, zstar, hg.WeightFunction.L2)
    tree, altitudes = hg.bpt_canonical(graph, edge_weights)
    filtered_tree, filtered_altitudes = hg.filter_non_relevant_node_from_tree(tree, altitudes, is_non_relevant)
    mean_color = hg.attribute_mean_vertex_weights(filtered_tree, zstar)
    explorer = hg.HorizontalCutExplorer(filtered_tree, filtered_altitudes)
    cut_nodes = explorer.horizontal_cut_from_num_regions(250, at_least=False)
    im = cut_nodes.reconstruct_leaf_data(filtered_tree, mean_color)
    img = Image.fromarray(format_as_255(im))
    img = img.resize((width, height))
    img.putpalette(COLOUR_PALETTE)
    img = img.convert("RGB")
    return filtered_tree, filtered_altitudes, img
