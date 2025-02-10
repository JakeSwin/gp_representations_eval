import sys

def recursive_search_size_bsp(node):
  if node.leaf:
    node_size = 0
    node_size += sys.getsizeof(node.region)
    node_size += sys.getsizeof(node.above_child)
    node_size += sys.getsizeof(node.below_child)
    node_size += sys.getsizeof(node.error)
    node_size += sys.getsizeof(node.avg_value)
    node_size += sys.getsizeof(node.depth)
    node_size += sys.getsizeof(node.leaf)
    node_size += sys.getsizeof(node.param)

    return node_size
  else:
    left_size = recursive_search_size_bsp(node.above_child)
    right_size = recursive_search_size_bsp(node.below_child)
    return left_size + right_size