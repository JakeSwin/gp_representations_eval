import pandas as pd
import shapely
import geopandas
import geopandas as gpd
import xarray as xr
from geopandas import GeoDataFrame
from shapely.geometry import mapping
from shapely.ops import cascaded_union
from pykrige.ok import OrdinaryKriging
from shapely.geometry import box
from PIL import Image
from latlon import LatLon
from functools import partial
from PIL import Image, ImageDraw
from itertools import chain
import h3
import base64
import urllib
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import random

def interpolate(color1, color2, fraction):
    return [int(c1 + (c2 - c1) * fraction) for c1, c2 in zip(color1, color2)]

COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
COLOUR_PALETTE = list(chain.from_iterable([interpolate(COLOUR_1, COLOUR_2, i / (256 - 1)) for i in range(256)]))

APERTURE_SIZE = 15

lat = 37.537215
lon = 15.068894

coord = LatLon(lat, lon)

def make_h3(zstar):
    df = xr.DataArray(zstar, dims=("lat", "lng")).to_pandas().stack().reset_index().rename(columns={0: "weed_chance"})
    df["lng"] = df["lng"].apply(lambda x: np.float64(coord.offset(90, 0.0001 * x).lon))
    df["lat"] = df["lat"].apply(lambda x: np.float64(coord.offset(0, 0.0001 * x).lat))

    hex_col = 'hex'+str(APERTURE_SIZE)

    # find hexs containing the points
    df[hex_col] = df.apply(lambda x: h3.latlng_to_cell(x.lat,x.lng,APERTURE_SIZE),1)

    # calculate weed_chance average per hex
    df_dem = df.groupby(hex_col)['weed_chance'].mean().to_frame('weed_chance').reset_index()

    #find center of hex for visualization
    df_dem['lat'] = df_dem[hex_col].apply(lambda x: h3.cell_to_latlng(x)[0])
    df_dem['lng'] = df_dem[hex_col].apply(lambda x: h3.cell_to_latlng(x)[1])

    def get_mse(res_name, row):
        parent = row[res_name]
        mean = row["weed_chance"]
        children_rows = df_dem.loc[df_dem[res_name] == parent]
        mse = ((children_rows["weed_chance"] - mean)**2).sum()
        return mse

    parent_dfs = {}

    res_to_check = np.arange(APERTURE_SIZE-1, APERTURE_SIZE-5, -1, )
    for res in res_to_check:
        res_name = hex_col + "_p" + str(res)
        df_dem[res_name] = [h3.cell_to_parent(cell, res) for cell in df_dem["hex15"]]
        parent_weed_chance = df_dem.groupby(res_name)['weed_chance'].mean().to_frame('weed_chance').reset_index()
        get_mse_for_res = partial(get_mse, res_name)
        parent_weed_chance["MSE"] = parent_weed_chance.apply(get_mse_for_res, axis=1)
        parent_dfs[res_name] = parent_weed_chance

    hexs = []
    weed_chance = []
    checking_order = [hex_col + "_p" + str(res) for res in res_to_check][::-1] + [hex_col]
    threshold = 0.001

    for index, row in df_dem.iterrows():
        for col in checking_order:
            if col == hex_col:
                hexs.append(row[col])
                weed_chance.append(row["weed_chance"])
                break

            hex_to_check = row[col]
            hex_info = parent_dfs[col].loc[parent_dfs[col][col] == hex_to_check].reset_index()
            mse = hex_info.loc[0, "MSE"]

            if mse < threshold:
                if hex_to_check not in hexs:
                    hexs.append(hex_to_check)
                    weed_chance.append(hex_info.loc[0, "weed_chance"])
                break
    
    gdf = gpd.GeoDataFrame(geometry=[h3.cells_to_h3shape([cell]) for cell in hexs])
    # fig = plt.gcf()
    # ax = plt.gca()
    # ax.set_xlim(df["lng"].min(), df["lng"].max())
    # ax.set_ylim(df["lat"].min(), df["lat"].max())
    # fig.set_size_inches(10,10)
    # gdf.plot(column=pd.Series(weed_chance, name="weed_chance"), figsize=(10, 10), ax=ax)
    # plt.show()

    gdf.crs = "EPSG:4326"
    roi_minx, roi_miny, roi_maxx, roi_maxy = df["lng"].min(), df["lat"].min(), df["lng"].max(), df["lat"].max()
    roi_box = box(roi_minx, roi_miny, roi_maxx, roi_maxy)
    clipped_gdf = gdf.clip(roi_box, sort=True)
    gdf_cartesian = clipped_gdf.to_crs(epsg=3857)
    minx, miny, maxx, maxy = gdf_cartesian.total_bounds
    print(minx, miny, maxx, maxy)
    img_h, img_w = zstar.shape
    scale_x = img_w / (maxx - minx)
    scale_y = img_h / (maxy - miny)
    print(scale_x, scale_y)
    polygon_coords = [list(poly.exterior.coords) for poly in gdf_cartesian.geometry]
    scaled_polygon_coords = []
    for poly_coords in polygon_coords:
        scaled_coords = [(int((x - minx) * scale_x), int(img_h - (y - miny) * scale_y)) for x, y in poly_coords]
        scaled_polygon_coords.append(scaled_coords)

    img = Image.new('P', (img_w, img_h))
    img.putpalette(COLOUR_PALETTE)
    d = ImageDraw.Draw(img)
    min_weed_chance = np.min(weed_chance)
    max_weed_chance = np.max(weed_chance)
    for i, coords in enumerate(scaled_polygon_coords):
        norm_value = (weed_chance[i] - min_weed_chance) / (max_weed_chance - min_weed_chance)
        d.polygon(coords, fill=int(norm_value * 255))
    img = img.convert("RGB")

    return hexs, weed_chance, img