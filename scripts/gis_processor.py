# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "geopandas",
#     "shapely",
#     "rasterio",
#     "scipy",
#     "matplotlib",
#     "pillow",
# ]
# ///

# -*- coding: utf-8 -*-
"""
14th National GIS Application Skills Contest (Morning Session A) Automation Pipeline Tool
This script provides automated data cleaning, spatial modeling, and imagery registration
for the morning session contest tasks.
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from scipy.signal import correlate
from scipy.ndimage import label
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

# Default Coordinate Reference Systems
DEFAULT_UTM_CRS = "EPSG:32610"
DEFAULT_WGS_CRS = "EPSG:4326"

def parse_dms(dms_str):
    """
    Parses Degree-Minute-Second (DMS) string to Decimal Degrees (DD).
    """
    if pd.isna(dms_str) or not isinstance(dms_str, str):
        return None
    dms_str = dms_str.strip()
    match = re.search(r"(\d+)\s*[^\d'\w\s]?\s*(\d+)\s*'\s*([\d.]+)\s*[^\d'\w\s]?\s*([NSEWnsew])", dms_str)
    if not match:
        match = re.search(r"(\d+)\s*\D+\s*(\d+)\s*'\s*([\d.]+)\s*\D+\s*([NSEWnsew])", dms_str)
        if not match:
            return None
    deg = float(match.group(1))
    mn = float(match.group(2))
    sec = float(match.group(3))
    dir_char = match.group(4).upper()
    decimal = deg + mn / 60.0 + sec / 3600.0
    if dir_char in ['S', 'W']:
        decimal = -decimal
    return decimal

def parse_met_coords(loc_str):
    """
    Parses weather station coordinate pair, e.g. "125°12'30\"W, 41°5'10\"N"
    """
    if pd.isna(loc_str) or not isinstance(loc_str, str):
        return None, None
    loc_str = loc_str.strip().strip('"')
    parts = loc_str.split(',')
    if len(parts) != 2:
        return None, None
    lon = parse_dms(parts[0])
    lat = parse_dms(parts[1])
    return lon, lat

def clean_animals(args):
    """
    Task 1: Clean animal records and generate study area boundary
    """
    print(f"\n>>> Starting wild animal data cleaning [File: {args.input}]")
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input, sep='\t', encoding='gbk')
    orig_len = len(df)
    
    # 1. Filter out null values
    df_clean = df.dropna(subset=['Year', 'Count', 'Name', 'Latitude', 'Longitude']).copy()
    nulls_dropped = orig_len - len(df_clean)
    
    # 2. Filter out duplicate records
    dups_dropped = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates().copy()
    
    # 3. Coordinate parsing
    df_clean['lat_deg'] = df_clean['Latitude'].apply(parse_dms)
    df_clean['lon_deg'] = df_clean['Longitude'].apply(parse_dms)
    
    # 4. Filter spatial outliers
    spatial_outliers = (
        (df_clean['lat_deg'] < args.min_lat) | (df_clean['lat_deg'] > args.max_lat) |
        (df_clean['lon_deg'] < args.min_lon) | (df_clean['lon_deg'] > args.max_lon)
    )
    outliers_dropped = spatial_outliers.sum()
    df_clean = df_clean[~spatial_outliers].copy()
    
    print("Animal Data Cleaning Summary:")
    print(f"  - Original Records: {orig_len}")
    print(f"  - Dropped due to null values: {nulls_dropped}")
    print(f"  - Dropped due to duplicates: {dups_dropped}")
    print(f"  - Dropped due to spatial outliers: {outliers_dropped}")
    print(f"  - Cleaned Records remaining: {len(df_clean)}")
    
    # Build geometry
    geometry = [Point(xy) for xy in zip(df_clean['lon_deg'], df_clean['lat_deg'])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs=DEFAULT_WGS_CRS)
    
    # Project to UTM Zone 10N
    gdf_proj = gdf.to_crs(args.crs)
    
    os.makedirs(args.output_gdb, exist_ok=True)
    out_points = os.path.join(args.output_gdb, "animal_cleaned.shp")
    gdf_proj.to_file(out_points, encoding='gbk')
    print(f"  [SUCCESS] Cleaned and projected animal points saved to: {out_points}")
    
    # 5. Extract bounding box buffered by specified distance
    bounds = gdf_proj.total_bounds
    envelope = box(bounds[0], bounds[1], bounds[2], bounds[3])
    study_area = envelope.buffer(args.buffer)
    study_area_gdf = gpd.GeoDataFrame(geometry=[study_area], crs=args.crs)
    
    out_sa = os.path.join(args.output_gdb, "study_area.shp")
    study_area_gdf.to_file(out_sa, encoding='gbk')
    print(f"  [SUCCESS] Study area boundary buffer saved to: {out_sa}")
    print(f"  Study Area projected bounds: MinX={bounds[0]:.1f}, MinY={bounds[1]:.1f}, MaxX={bounds[2]:.1f}, MaxY={bounds[3]:.1f}")
    
    return out_points, out_sa

def clean_weather(args):
    """
    Task 2: Clean climate stations, run 10% validation and interpolate grids
    """
    print(f"\n>>> Starting climate station cleaning & spatial interpolation [File: {args.input}]")
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)
        
    df = pd.read_csv(args.input, sep='\t', encoding='gbk')
    orig_len = len(df)
    
    # 1. Filter out duplicate records
    df_clean = df.drop_duplicates().copy()
    dups_dropped = orig_len - len(df_clean)
    
    # 2. Parse coordinates
    lons, lats = [], []
    for idx, row in df_clean.iterrows():
        lon, lat = parse_met_coords(row['Latitude and Longitude'])
        lons.append(lon)
        lats.append(lat)
    df_clean['lon_deg'] = lons
    df_clean['lat_deg'] = lats
    
    # 3. Clean temperature physical anomalies and spatial outliers
    temp_outliers = (df_clean['tmp_k'] < 0) | (df_clean['tmp_k'] > 350)
    temp_dropped = temp_outliers.sum()
    df_clean = df_clean[~temp_outliers].copy()
    
    spatial_outliers = (
        (df_clean['lat_deg'] < args.min_lat) | (df_clean['lat_deg'] > args.max_lat) |
        (df_clean['lon_deg'] < args.min_lon) | (df_clean['lon_deg'] > args.max_lon)
    )
    spatial_dropped = spatial_outliers.sum()
    df_clean = df_clean[~spatial_outliers].copy()
    
    # Kelvin to Celsius conversion
    df_clean['tmp_c'] = df_clean['tmp_k'] - 273.15
    
    print("Climate Data Cleaning Summary:")
    print(f"  - Original Records: {orig_len}")
    print(f"  - Dropped duplicates: {dups_dropped}")
    print(f"  - Dropped temperature anomalies: {temp_dropped}")
    print(f"  - Dropped spatial outliers: {spatial_dropped}")
    print(f"  - Cleaned Stations remaining: {len(df_clean)}")
    
    # Build projected stations shapefile
    geometry = [Point(xy) for xy in zip(df_clean['lon_deg'], df_clean['lat_deg'])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs=DEFAULT_WGS_CRS)
    gdf_proj = gdf.to_crs(args.crs)
    
    os.makedirs(args.output_gdb, exist_ok=True)
    out_points = os.path.join(args.output_gdb, "qixiang_cleaned.shp")
    gdf_proj.to_file(out_points, encoding='gbk')
    print(f"  [SUCCESS] Climate stations saved to: {out_points}")
    
    # 4. Accuracy validation (10% subset)
    gdf_proj['x'] = gdf_proj.geometry.x
    gdf_proj['y'] = gdf_proj.geometry.y
    
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(gdf_proj))
    val_sz = int(len(gdf_proj) * 0.1)
    val_idx = shuffled_idx[:val_sz]
    train_idx = shuffled_idx[val_sz:]
    
    train_pts = gdf_proj.iloc[train_idx]
    val_pts = gdf_proj.iloc[val_idx]
    
    train_xy = train_pts[['x', 'y']].values
    val_xy = val_pts[['x', 'y']].values
    
    pred_temp = griddata(train_xy, train_pts['tmp_c'].values, val_xy, method='linear')
    pred_prec = griddata(train_xy, train_pts['pre_mm'].values, val_xy, method='linear')
    
    # Nan handling (fill boundary with nearest neighbor)
    nan_temp = np.isnan(pred_temp)
    if nan_temp.any():
        pred_temp[nan_temp] = griddata(train_xy, train_pts['tmp_c'].values, val_xy, method='nearest')[nan_temp]
    nan_prec = np.isnan(pred_prec)
    if nan_prec.any():
        pred_prec[nan_prec] = griddata(train_xy, train_pts['pre_mm'].values, val_xy, method='nearest')[nan_prec]
        
    rmse_temp = np.sqrt(np.mean((val_pts['tmp_c'] - pred_temp)**2))
    rmse_prec = np.sqrt(np.mean((val_pts['pre_mm'] - pred_prec)**2))
    
    print("Interpolation Grid Accuracy Validation (10% Cross-Validation):")
    print(f"  - Temperature RMSE: {rmse_temp:.4f} °C (Limit <= 0.5 °C) -> {'[PASS]' if rmse_temp <= 0.5 else '[FAIL]'}")
    print(f"  - Precipitation RMSE: {rmse_prec:.4f} mm (Limit <= 0.5 mm) -> {'[PASS]' if rmse_prec <= 0.5 else '[FAIL]'}")
    
    # Plotting and saving error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(val_pts['tmp_c'] - pred_temp, bins=15, color='#4A90E2', edgecolor='black', alpha=0.8)
    axes[0].set_title(f'Temperature Interpolation Error\n(RMSE = {rmse_temp:.4f} °C)')
    axes[0].set_xlabel('Error (Observed - Predicted) [°C]')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    axes[1].hist(val_pts['pre_mm'] - pred_prec, bins=15, color='#50E3C2', edgecolor='black', alpha=0.8)
    axes[1].set_title(f'Precipitation Interpolation Error\n(RMSE = {rmse_prec:.4f} mm)')
    axes[1].set_xlabel('Error (Observed - Predicted) [mm]')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    out_plot = os.path.join(args.output_gdb, "error_statistics.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"  [SUCCESS] Error statistics plot saved to: {out_plot}")
    
    # 5. Full grid interpolation matching the study area (30m)
    if not args.study_area or not os.path.exists(args.study_area):
        print("[ERROR] Climate grid interpolation requires the study area shapefile (--study-area)!")
        sys.exit(1)
        
    sa_gdf = gpd.read_file(args.study_area)
    sa_poly = sa_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = sa_poly.bounds
    res = args.resolution
    
    cols = int(np.ceil((maxx - minx) / res))
    rows = int(np.ceil((maxy - miny) / res))
    
    grid_x = np.linspace(minx, maxx, cols)
    grid_y = np.linspace(maxy, miny, rows)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    all_xy = gdf_proj[['x', 'y']].values
    
    grid_temp = griddata(all_xy, gdf_proj['tmp_c'].values, (grid_X, grid_Y), method='linear')
    grid_prec = griddata(all_xy, gdf_proj['pre_mm'].values, (grid_X, grid_Y), method='linear')
    
    # Fill boundary NaNs
    n_temp = np.isnan(grid_temp)
    if n_temp.any():
        grid_temp[n_temp] = griddata(all_xy, gdf_proj['tmp_c'].values, (grid_X, grid_Y), method='nearest')[n_temp]
    n_prec = np.isnan(grid_prec)
    if n_prec.any():
        grid_prec[n_prec] = griddata(all_xy, gdf_proj['pre_mm'].values, (grid_X, grid_Y), method='nearest')[n_prec]
        
    transform = rasterio.transform.from_origin(minx, maxy, res, res)
    
    def save_raster(data, path):
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=rows, width=cols,
            count=1, dtype='float32',
            crs=args.crs,
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(data.astype('float32'), 1)
            
    temp_raster = os.path.join(args.output_gdb, "temp_c.tif")
    prec_raster = os.path.join(args.output_gdb, "precip_mm.tif")
    save_raster(grid_temp, temp_raster)
    save_raster(grid_prec, prec_raster)
    print(f"  [SUCCESS] 30m Temperature raster exported to: {temp_raster}")
    print(f"  [SUCCESS] 30m Precipitation raster exported to: {prec_raster}")
    
    return temp_raster, prec_raster

def process_dem(args):
    """
    Task 2: Merge raw DEM tiles, reproject to UTM 10N and clip to study area
    """
    print(f"\n>>> Starting DEM mosaicing, reprojection & clipping [Dir: {args.dem_dir}]")
    if not os.path.exists(args.dem_dir):
        print(f"[ERROR] DEM directory does not exist: {args.dem_dir}")
        sys.exit(1)
        
    dem_files = [os.path.join(args.dem_dir, f) for f in os.listdir(args.dem_dir) if f.endswith('.tif')]
    if len(dem_files) == 0:
        print("[ERROR] No DEM .tif files found in the directory!")
        sys.exit(1)
        
    print(f"  Found {len(dem_files)} DEM tiles to merge: {[os.path.basename(f) for f in dem_files]}")
    
    # 1. Mosaic
    src_list = [rasterio.open(f) for f in dem_files]
    mosaic_arr, mosaic_trans = merge(src_list)
    src_meta = src_list[0].meta.copy()
    
    os.makedirs(args.output_gdb, exist_ok=True)
    temp_merged = os.path.join(args.output_gdb, "dem_merged_temp.tif")
    
    src_meta.update({
        "height": mosaic_arr.shape[1],
        "width": mosaic_arr.shape[2],
        "transform": mosaic_trans,
        "crs": DEFAULT_WGS_CRS
    })
    
    with rasterio.open(temp_merged, 'w', **src_meta) as dst:
        dst.write(mosaic_arr)
        
    for src in src_list:
        src.close()
    print("  Mosaic merging completed.")
    
    # 2. Reproject to UTM Zone 10N with bilinear resampling (30m)
    print("  Reprojecting merged DEM to UTM Zone 10N (30m resolution)...")
    with rasterio.open(temp_merged) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, args.crs, src.width, src.height, *src.bounds, resolution=(args.resolution, args.resolution)
        )
        
        temp_reproj = os.path.join(args.output_gdb, "dem_reproj_temp.tif")
        reproj_meta = src.meta.copy()
        reproj_meta.update({
            'crs': args.crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'dtype': 'float32'
        })
        
        with rasterio.open(temp_reproj, 'w', **reproj_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=args.crs,
                resampling=Resampling.bilinear
            )
            
    # 3. Clip Precisely to study area boundary
    if not args.study_area or not os.path.exists(args.study_area):
        print("[ERROR] DEM clipping requires the study area shapefile (--study-area)!")
        sys.exit(1)
        
    print("  Clipping projected DEM to study area boundary...")
    sa_gdf = gpd.read_file(args.study_area)
    geoms = [sa_gdf.geometry.iloc[0].__geo_interface__]
    
    with rasterio.open(temp_reproj) as src:
        clipped_arr, clipped_trans = mask(src, geoms, crop=True)
        clipped_meta = src.meta.copy()
        
    clipped_meta.update({
        'height': clipped_arr.shape[1],
        'width': clipped_arr.shape[2],
        'transform': clipped_trans
    })
    
    final_dem = os.path.join(args.output_gdb, "dem_clipped.tif")
    with rasterio.open(final_dem, 'w', **clipped_meta) as dst:
        dst.write(clipped_arr)
        
    # Clean up temporary files
    if os.path.exists(temp_merged):
        os.remove(temp_merged)
    if os.path.exists(temp_reproj):
        os.remove(temp_reproj)
        
    print(f"  [SUCCESS] Clipped high-precision DEM grid saved to: {final_dem}")
    return final_dem

def register_landsat(args):
    """
    Task 3: CCW Landsat rotation & FFT-based automatic spatial auto-registration
    """
    print(f"\n>>> Starting Landsat georeferencing and spatial auto-alignment [File: {args.input}]")
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)
    if not args.landcover or not os.path.exists(args.landcover):
        print("[ERROR] Automatic Landsat registration requires the cloud-repaired landcover map (--landcover)!")
        sys.exit(1)
        
    # 1. Read landcover water mask at 30m resolution for fast FFT correlation
    print("  Extracting landcover water mask for shoreline matching...")
    with rasterio.open(args.landcover) as src:
        lc_bounds = src.bounds
        minx, miny, maxx, maxy = lc_bounds.left, lc_bounds.bottom, lc_bounds.right, lc_bounds.top
        ref_h, ref_w = 2109, 1122
        water_ref_raw = src.read(1, out_shape=(ref_h, ref_w), resampling=Resampling.nearest)
        water_lc = (water_ref_raw == 1).astype(np.float32)
        
    print(f"    Reference water mask pixel count: {np.sum(water_lc)}")
    
    # 2. Read Landsat Green & SWIR2 to detect water via MNDWI
    print("  Loading Landsat bands for MNDWI water body index calculation...")
    with rasterio.open(args.input) as src:
        green_raw = src.read(2, out_shape=(1094, 2285), resampling=Resampling.bilinear).astype(np.float32)
        swir2_raw = src.read(6, out_shape=(1094, 2285), resampling=Resampling.bilinear).astype(np.float32)
        
    valid = (green_raw > 0) & (swir2_raw > 0)
    g_val = np.where(valid, green_raw, np.nan)
    s_val = np.where(valid, swir2_raw, np.nan)
    
    # Normalize bands for index calculation
    g_min, g_max = np.nanmin(g_val), np.nanmax(g_val)
    s_min, s_max = np.nanmin(s_val), np.nanmax(s_val)
    g_norm = (g_val - g_min) / (g_max - g_min + 1e-6)
    s_norm = (s_val - s_min) / (s_max - s_min + 1e-6)
    mndwi_30m = (g_norm - s_norm) / (g_norm + s_norm + 1e-6)
    mndwi_30m = np.where(np.isnan(mndwi_30m), -1.0, mndwi_30m)
    
    water_ls = (mndwi_30m > 0.4).astype(np.float32)
    print(f"    Landsat MNDWI water body pixel count (30m): {np.sum(water_ls)}")
    
    # 3. FFT-based 2D Cross-Correlation with orientation search
    print("  Running 2D FFT Cross-Correlation sweep across orientations...")
    orientations = {
        "Rot90 CCW": np.rot90(water_ls, 1),
        "Rot270 CCW": np.rot90(water_ls, 3),
        "Transpose": water_ls.T,
        "Transverse": np.rot90(water_ls, 1).T
    }
    
    best_score = -1
    best_name = None
    best_dy, best_dx = 0, 0
    
    for name, grid in orientations.items():
        corr = correlate(grid, water_lc, mode='same', method='fft')
        max_idx = np.unravel_index(np.argmax(corr), corr.shape)
        score = corr[max_idx]
        dy = max_idx[0] - corr.shape[0] // 2
        dx = max_idx[1] - corr.shape[1] // 2
        
        print(f"    - Direction [{name:12s}] -> Max Overlap Score: {score:.2f} (dx={dx}, dy={dy})")
        if score > best_score:
            best_score = score
            best_name = name
            best_dy = dy
            best_dx = dx
            
    dx_meters = best_dx * 30.0
    dy_meters = best_dy * 30.0
    print(f"  [ALIGNMENT SUCCESS] Optimal match direction: {best_name} (Max Correlation Score: {best_score:.2f})")
    print(f"  [ALIGNMENT OFFSETS] Calculated translation offsets: dx = {dx_meters:.1f} m, dy = {dy_meters:.1f} m")
    
    # 4. Apply offset to high-res (15m) Landsat bands and crop to study area
    print("  Reprojecting Landsat bands to 15m target grid aligned with study area...")
    ls_res = 15.0
    ls_cols = int(np.ceil((maxx - minx) / ls_res))
    ls_rows = int(np.ceil((maxy - miny) / ls_res))
    ls_transform = from_bounds(minx, miny, maxx, maxy, ls_cols, ls_rows)
    
    bands_data = {}
    with rasterio.open(args.input) as src:
        meta = src.meta.copy()
        for b in range(1, 7):
            print(f"    Rotating Band {b}...")
            bands_data[b] = np.rot90(src.read(b), 1)
            
    h_ls, w_ls = bands_data[1].shape
    physical_w = w_ls * 15.0
    physical_h = h_ls * 15.0
    
    sa_center_x = (minx + maxx) / 2.0
    sa_center_y = (miny + maxy) / 2.0
    
    c_x = sa_center_x + dx_meters
    c_y = sa_center_y + dy_meters
    
    left = c_x - physical_w / 2.0
    right = c_x + physical_w / 2.0
    bottom = c_y - physical_h / 2.0
    top = c_y + physical_h / 2.0
    
    landsat_src_transform = from_bounds(left, bottom, right, top, w_ls, h_ls)
    
    os.makedirs(args.output_gdb, exist_ok=True)
    out_landsat = os.path.join(args.output_gdb, "landsat_clipped.tif")
    
    meta.update({
        "driver": "GTiff",
        "crs": args.crs,
        "transform": ls_transform,
        "width": ls_cols,
        "height": ls_rows,
        "count": 6,
        "dtype": "float32",
        "nodata": 0.0
    })
    
    with rasterio.open(out_landsat, "w", **meta) as dst:
        for b in range(1, 7):
            print(f"    Warping High-Res Band {b}...")
            reproj_band = np.zeros((ls_rows, ls_cols), dtype=np.float32)
            reproject(
                source=bands_data[b].astype(np.float32),
                destination=reproj_band,
                src_transform=landsat_src_transform,
                src_crs=args.crs,
                src_nodata=0.0,
                dst_transform=ls_transform,
                dst_crs=args.crs,
                dst_nodata=0.0,
                resampling=Resampling.bilinear
            )
            dst.write(reproj_band, b)
            
    print(f"  [SUCCESS] Georeferenced and clipped multi-band Landsat raster saved to: {out_landsat}")
    return out_landsat

def update_landcover(args):
    """
    Task 3: Landcover cloud major focal filling, spectral extraction of water/built-up, updating categories
    """
    print(f"\n>>> Starting landcover cloud repair, index extraction, and updating [File: {args.input}]")
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)
    if not args.dem or not os.path.exists(args.dem):
        print("[ERROR] Landcover update process requires the clipped DEM raster (--dem)!")
        sys.exit(1)
    if not args.landsat or not os.path.exists(args.landsat):
        print("[ERROR] Landcover update process requires the registered Landsat raster (--landsat)!")
        sys.exit(1)
        
    with rasterio.open(args.dem) as dem_src:
        dem_bounds = dem_src.bounds
    minx, miny, maxx, maxy = dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top
    
    # 1. Warp unaligned landcover to 10m target UTM grid
    print("  1. Reprojecting raw landcover.tif to UTM 10m standard grid...")
    lc_res = 10.0
    lc_cols = int(np.ceil((maxx - minx) / lc_res))
    lc_rows = int(np.ceil((maxy - miny) / lc_res))
    lc_transform = from_bounds(minx, miny, maxx, maxy, lc_cols, lc_rows)
    
    with rasterio.open(args.input) as src:
        lc_data = np.full((lc_rows, lc_cols), 255, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=lc_data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else 255,
            dst_transform=lc_transform,
            dst_crs=args.crs,
            dst_nodata=255,
            resampling=Resampling.nearest
        )
        
    # 2. Local window focal mode repair for cloud pixels (value 10)
    print("  2. Repairing cloud-obstructed pixels (value 10) using Focal Majority Mode...")
    repaired_lc = lc_data.copy()
    cloud_rows, cloud_cols = np.where(lc_data == 10)
    repaired_count = 0
    
    for r, c in zip(cloud_rows, cloud_cols):
        found = False
        for half_w in range(1, 50):
            r_min = max(0, r - half_w)
            r_max = min(lc_rows - 1, r + half_w)
            c_min = max(0, c - half_w)
            c_max = min(lc_cols - 1, c + half_w)
            
            window = lc_data[r_min:r_max+1, c_min:c_max+1]
            flat_win = window.ravel()
            valid = flat_win[(flat_win != 10) & (flat_win != 255)]
            
            if len(valid) > 0:
                c_counter = Counter(valid)
                mode_val = c_counter.most_common(1)[0][0]
                repaired_lc[r, c] = mode_val
                repaired_count += 1
                found = True
                break
                
    print(f"    Focal majority filling completed: repaired {repaired_count} cloud pixels.")
    print(f"    Remaining un-filled cloud pixels: {np.sum(repaired_lc == 10)}")
    
    # 3. Read Landsat, calculate indexes, and extract water / built-up land covers
    print("  3. Calculating indices (MNDWI, BU) and extracting high-res land features...")
    with rasterio.open(args.landsat) as ls_src:
        ls_bounds = ls_src.bounds
        ls_trans = ls_src.transform
        
        green_15m = ls_src.read(2)
        red_15m = ls_src.read(3)
        nir_15m = ls_src.read(4)
        swir1_15m = ls_src.read(5)
        swir2_15m = ls_src.read(6)
        
    norm_bands = {}
    for b_idx, arr in enumerate([green_15m, red_15m, nir_15m, swir1_15m, swir2_15m], start=2):
        valid_m = arr > 0
        b_min, b_max = arr[valid_m].min(), arr[valid_m].max()
        norm_bands[b_idx] = np.zeros_like(arr)
        norm_bands[b_idx][valid_m] = (arr[valid_m] - b_min) / (b_max - b_min + 1e-6)
        
    # A. Water: MNDWI = (Green - SWIR2) / (Green + SWIR2) > 0.4
    mndwi = (norm_bands[2] - norm_bands[6]) / (norm_bands[2] + norm_bands[6] + 1e-6)
    water_raw = (mndwi > 0.4) & (green_15m > 0)
    
    # Connectivity filter: size > 0.1 km2 (15m pixel = 225m2, 0.1km2 = 100,000m2 = 444.4 pixels)
    labeled_w, num_w = label(water_raw)
    water_final = np.zeros_like(water_raw, dtype=np.uint8)
    for f in range(1, num_w + 1):
        p_cnt = np.sum(labeled_w == f)
        if (p_cnt * 225.0) > 100000.0:
            water_final[labeled_w == f] = 1
            
    # B. Built-up: BU = (SWIR1 + SWIR2) / (Red + NIR) > 3.0
    bu = (norm_bands[5] + norm_bands[6]) / (norm_bands[3] + norm_bands[4] + 1e-6)
    built_raw = (bu > 3.0) & (green_15m > 0)
    
    # Connectivity filter: size > 1.0 km2 (1.0km2 = 1,000,000m2 = 4444.4 pixels)
    labeled_b, num_b = label(built_raw)
    built_final = np.zeros_like(built_raw, dtype=np.uint8)
    for f in range(1, num_b + 1):
        p_cnt = np.sum(labeled_b == f)
        if (p_cnt * 225.0) > 1000000.0:
            built_final[labeled_b == f] = 1
            
    print(f"    Extracted Water Area: {np.sum(water_final) * 225.0 / 1e6:.4f} km2")
    print(f"    Extracted Built-up Area: {np.sum(built_final) * 225.0 / 1e6:.4f} km2")
    
    # 4. Resample features to 10m grid and burn updates
    print("  4. Burning updated features to the final 10m landcover grid...")
    water_10m = np.zeros((lc_rows, lc_cols), dtype=np.uint8)
    built_10m = np.zeros((lc_rows, lc_cols), dtype=np.uint8)
    
    reproject(
        source=water_final, destination=water_10m,
        src_transform=ls_trans, src_crs=args.crs, src_nodata=0,
        dst_transform=lc_transform, dst_crs=args.crs, dst_nodata=0,
        resampling=Resampling.nearest
    )
    reproject(
        source=built_final, destination=built_10m,
        src_transform=ls_trans, src_crs=args.crs, src_nodata=0,
        dst_transform=lc_transform, dst_crs=args.crs, dst_nodata=0,
        resampling=Resampling.nearest
    )
    
    updated_lc = repaired_lc.copy()
    updated_lc = np.where(water_10m == 1, 1, updated_lc)
    updated_lc = np.where(built_10m == 1, 7, updated_lc)
    
    os.makedirs(args.output_gdb, exist_ok=True)
    out_lc = os.path.join(args.output_gdb, "Updated_LC.tif")
    
    with rasterio.open(
        out_lc, 'w',
        driver='GTiff',
        height=lc_rows, width=lc_cols,
        count=1, dtype='uint8',
        crs=args.crs,
        transform=lc_transform,
        nodata=255
    ) as dst:
        dst.write(updated_lc, 1)
    print(f"  [SUCCESS] Updated landcover raster saved to: {out_lc}")
    
    # 5. Class Areal Statistics
    print("  5. Generating final class areal statistics table...")
    class_names = {
        1: "Water Areas (水域)",
        2: "Trees (林地)",
        4: "Flooded Vegetation (淹没植被)",
        5: "Crops (耕地)",
        7: "Built Areas (建筑区)",
        8: "Bare Ground (裸地)",
        9: "Snow/Ice (雪/冰)",
        11: "Rangelands (牧场)"
    }
    
    unique_vals, counts = np.unique(updated_lc, return_counts=True)
    stats_data = []
    for val, count in zip(unique_vals, counts):
        if val == 255:
            continue
        c_name = class_names.get(val, "Unknown")
        # 10m pixel = 100m2
        area_km2 = count * 100.0 / 1e6
        stats_data.append({
            "Code": val,
            "Class Name": c_name,
            "Pixels": count,
            "Area (km2)": round(area_km2, 4)
        })
        
    df_stats = pd.DataFrame(stats_data)
    total_sa = df_stats["Area (km2)"].sum()
    df_stats["Percentage (%)"] = round(df_stats["Area (km2)"] / total_sa * 100, 2)
    
    out_csv = os.path.join(args.output_gdb, "landcover_area_statistics.csv")
    df_stats.to_csv(out_csv, index=False, encoding='gbk')
    
    print("\nLand Cover Areal Statistics Table:")
    print("=" * 65)
    print(df_stats.to_string(index=False))
    print("=" * 65)
    print(f"Total Study Area: {total_sa:.4f} km2\n")
    print(f"  [SUCCESS] Stats table exported as CSV to: {out_csv}")
    
    return out_lc, out_csv

def run_all(args):
    """
    Runs the entire pipeline (Tasks 1, 2, and 3) end-to-end.
    """
    print("\n======================================================================")
    print("      ★ GIS Contest Morning A Session Automation Pipeline ★")
    print("======================================================================")
    
    result_gdb = os.path.join(args.output_dir, "result_data.gdb")
    os.makedirs(result_gdb, exist_ok=True)
    
    # 1. Animal point cleaning
    anim_txt = os.path.join(args.base_dir, "dongwu", "animal.txt")
    args.input = anim_txt
    args.output_gdb = result_gdb
    args.buffer = 2000.0
    _, study_area_shp = clean_animals(args)
    
    # 2. Climate station interpolation
    met_txt = os.path.join(args.base_dir, "qixiang", "Meteorological_sampling.txt")
    args.input = met_txt
    args.study_area = study_area_shp
    args.resolution = 30.0
    clean_weather(args)
    
    # 3. DEM mosaicing
    dem_folder = os.path.join(args.base_dir, "dem")
    args.dem_dir = dem_folder
    args.resolution = 30.0
    clipped_dem = process_dem(args)
    
    # 4. Cloud pre-filling for Landsat registration
    print("\n>>> Preparing temporary landcover base layer for auto-registration...")
    raw_lc_tif = os.path.join(args.base_dir, "tudi", "landcover.tif")
    landsat_tif = os.path.join(args.base_dir, "yingx", "Landsat.tif")
    
    temp_dir = os.path.join(args.output_dir, "temp_data.gdb")
    os.makedirs(temp_dir, exist_ok=True)
    
    with rasterio.open(clipped_dem) as dem_src:
        dem_bounds = dem_src.bounds
    minx, miny, maxx, maxy = dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top
    
    lc_res = 10.0
    lc_cols = int(np.ceil((maxx - minx) / lc_res))
    lc_rows = int(np.ceil((maxy - miny) / lc_res))
    lc_transform = from_bounds(minx, miny, maxx, maxy, lc_cols, lc_rows)
    
    with rasterio.open(raw_lc_tif) as src:
        lc_data = np.full((lc_rows, lc_cols), 255, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1), destination=lc_data,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=lc_transform, dst_crs=args.crs, dst_nodata=255,
            resampling=Resampling.nearest
        )
    
    repaired_lc = lc_data.copy()
    cloud_rows, cloud_cols = np.where(lc_data == 10)
    for r, c in zip(cloud_rows, cloud_cols):
        for half_w in range(1, 20):
            r_min = max(0, r - half_w)
            r_max = min(lc_rows - 1, r + half_w)
            c_min = max(0, c - half_w)
            c_max = min(lc_cols - 1, c + half_w)
            window = lc_data[r_min:r_max+1, c_min:c_max+1]
            flat_win = window.ravel()
            valid = flat_win[(flat_win != 10) & (flat_win != 255)]
            if len(valid) > 0:
                repaired_lc[r, c] = Counter(valid).most_common(1)[0][0]
                break
                
    temp_repaired_lc = os.path.join(temp_dir, "landcover_repaired_10m_temp.tif")
    with rasterio.open(
        temp_repaired_lc, 'w', driver='GTiff', height=lc_rows, width=lc_cols,
        count=1, dtype='uint8', crs=args.crs, transform=lc_transform, nodata=255
    ) as dst:
        dst.write(repaired_lc, 1)
        
    # 5. Landsat auto-registration
    args.input = landsat_tif
    args.landcover = temp_repaired_lc
    args.output_gdb = result_gdb
    registered_landsat = register_landsat(args)
    
    # 6. Update landcover and output statistics
    args.input = raw_lc_tif
    args.dem = clipped_dem
    args.landsat = registered_landsat
    args.output_gdb = result_gdb
    update_landcover(args)
    
    # Cleanup temporary repaired base
    if os.path.exists(temp_repaired_lc):
        os.remove(temp_repaired_lc)
        
    print("\n======================================================================")
    print("      ★ Pipeline completed successfully! Outputs are in result_data.gdb ★")
    print("======================================================================")

def main():
    parser = argparse.ArgumentParser(description="GIS Morning Session A Automation Geoprocessing Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    def add_common_args(sp):
        sp.add_argument("--crs", default=DEFAULT_UTM_CRS, help="Target UTM Coordinate Reference System (default: EPSG:32610)")
        sp.add_argument("--min-lat", type=float, default=39.5, help="Valid latitude lower boundary")
        sp.add_argument("--max-lat", type=float, default=42.5, help="Valid latitude upper boundary")
        sp.add_argument("--min-lon", type=float, default=-127.0, help="Valid longitude lower boundary")
        sp.add_argument("--max-lon", type=float, default=-123.0, help="Valid longitude upper boundary")
        sp.add_argument("--output-gdb", required=True, help="Output directory / Geodatabase path")

    # 1. clean-animals
    sp_anim = subparsers.add_parser("clean-animals", help="Task 1: Clean animal point records and generate study area boundary")
    sp_anim.add_argument("--input", required=True, help="Path to raw animal.txt file")
    sp_anim.add_argument("--buffer", type=float, default=2000.0, help="Study area buffer outer distance in meters")
    add_common_args(sp_anim)
    
    # 2. clean-weather
    sp_met = subparsers.add_parser("clean-weather", help="Task 2: Clean weather station data and interpolate climate grids")
    sp_met.add_argument("--input", required=True, help="Path to raw Meteorological_sampling.txt file")
    sp_met.add_argument("--study-area", required=True, help="Path to study_area.shp boundary")
    sp_met.add_argument("--resolution", type=float, default=30.0, help="Climate interpolation grid resolution in meters")
    add_common_args(sp_met)
    
    # 3. process-dem
    sp_dem = subparsers.add_parser("process-dem", help="Task 2: Merge raw DEM tiles, reproject to UTM and mask clip")
    sp_dem.add_argument("--dem-dir", required=True, help="Directory containing raw demA/B/C/D .tif tiles")
    sp_dem.add_argument("--study-area", required=True, help="Path to study_area.shp boundary")
    sp_dem.add_argument("--resolution", type=float, default=30.0, help="DEM target grid resolution in meters")
    sp_dem.add_argument("--crs", default=DEFAULT_UTM_CRS, help="Target coordinate reference system")
    sp_dem.add_argument("--output-gdb", required=True, help="Output directory")
    
    # 4. register-landsat
    sp_ls = subparsers.add_parser("register-landsat", help="Task 3: Rotate Landsat bands and execute fast FFT-based auto-alignment")
    sp_ls.add_argument("--input", required=True, help="Path to raw Landsat.tif file")
    sp_ls.add_argument("--landcover", required=True, help="Path to cloud-repaired landcover map (shoreline base)")
    sp_ls.add_argument("--crs", default=DEFAULT_UTM_CRS, help="Target coordinate reference system")
    sp_ls.add_argument("--output-gdb", required=True, help="Output directory")
    
    # 5. update-landcover
    sp_lc = subparsers.add_parser("update-landcover", help="Task 3: Cloud pixel repair, MNDWI/BU index extraction, category updating and statistics")
    sp_lc.add_argument("--input", required=True, help="Path to raw landcover.tif file")
    sp_lc.add_argument("--dem", required=True, help="Path to clipped DEM raster")
    sp_lc.add_argument("--landsat", required=True, help="Path to registered Landsat raster")
    sp_lc.add_argument("--crs", default=DEFAULT_UTM_CRS, help="Target coordinate reference system")
    sp_lc.add_argument("--output-gdb", required=True, help="Output directory")
    
    # 6. run-all (One-click pipeline)
    sp_all = subparsers.add_parser("run-all", help="Runs Tasks 1, 2, and 3 sequentially in a single command")
    sp_all.add_argument("--base-dir", required=True, help="Root base directory containing contest raw '数据' folder")
    sp_all.add_argument("--output-dir", required=True, help="Root destination directory for results")
    sp_all.add_argument("--crs", default=DEFAULT_UTM_CRS, help="Target coordinate reference system")
    sp_all.add_argument("--min-lat", type=float, default=39.5)
    sp_all.add_argument("--max-lat", type=float, default=42.5)
    sp_all.add_argument("--min-lon", type=float, default=-127.0)
    sp_all.add_argument("--max-lon", type=float, default=-123.0)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
        
    if args.command == "clean-animals":
        clean_animals(args)
    elif args.command == "clean-weather":
        clean_weather(args)
    elif args.command == "process-dem":
        process_dem(args)
    elif args.command == "register-landsat":
        register_landsat(args)
    elif args.command == "update-landcover":
        update_landcover(args)
    elif args.command == "run-all":
        run_all(args)

if __name__ == "__main__":
    main()
