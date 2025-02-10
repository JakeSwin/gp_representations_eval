import numpy as np

def get_mask(shape, dist, angle):
    # Get image dimensions
    height, width = shape

    # Create a mask of the same size as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    x_vals = np.append(np.flip(-np.arange(width/2)), np.arange(width/2))
    y_vals = np.append(np.flip(np.arange(height/2)), -np.arange(height/2))
    # Create a grid of x and y coordinates
    x, y = np.meshgrid(x_vals, y_vals)

    # Create the mask based on the normal line equation
    mask[x*np.cos(angle) + y*np.sin(angle) < dist] = 255

    return mask

def cut_points(points, dist, angle):
    # Get dimensions
    height, width = points.shape

    x_vals = np.append(np.flip(-np.arange(np.floor(width/2))), np.arange(np.ceil(width/2)))
    y_vals = np.append(np.flip(np.arange(np.floor(height/2))), -np.arange(np.ceil(height/2)))
    # Create a grid of x and y coordinates
    x, y = np.meshgrid(x_vals, y_vals)

    # Create the mask based on the normal line equation
    p1 = points[x*np.cos(angle) + y*np.sin(angle) < dist]
    p2 = points[x*np.cos(angle) + y*np.sin(angle) > dist]

    return p1, p2

def average_colour(image):
    # convert image to np array
    image_arr = np.asarray(image)

    # get average of whole image
    avg_color_per_row = np.average(image_arr, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

def weighted_average(hist):
    total = sum(hist)
    error = value = 0

    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5

    return error

def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    # IDK what these magic numbers are
    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140

    return detail_intensity

def get_homogeneity(points, avg):
    rmse = np.sqrt(np.mean((points - avg)**2))
    return rmse

def interpolate(color1, color2, fraction):
    return [int(c1 + (c2 - c1) * fraction) for c1, c2 in zip(color1, color2)]