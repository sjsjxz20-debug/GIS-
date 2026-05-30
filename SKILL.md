---
name: gis-contest-a-morning-processor
description: >-
  A professional spatial analysis and geoprocessing skill designed to automate Tasks 1, 2, and 3 of the GIS Application Skills Contest (Morning Session A). Includes wild animal coordinate parsing, climate station outlier removal and linear triangulation grid interpolation, DEM mosaicing and clipping, Landsat CCW rotation and FFT-based spatial auto-registration, and landcover cloud filling/spectral index extraction.
---

# GIS Morning Session A Processor Skill

## Overview
This skill provides a fully automated spatial analysis pipeline for processing the datasets of the **14th National GIS Application Skills Contest (Morning Session A)**. It converts messy, raw text data and misaligned rasters into highly accurate, publication-ready GIS deliverables.

It addresses:
- **Task 1**: Parsing non-standard DMS latitude/longitude wild animal coordinates, cleaning duplicate/outlier records, and generating WGS84 and projected UTM 10N shapefiles plus a 2km buffered study area envelope.
- **Task 2**: Cleaning weather station attributes, converting Kelvin to Celsius, performing Delaunay triangulation-based linear grid interpolation (30m resolution) aligned to the study area, verifying interpolation RMSE against a 10% test set (RMSE <= 0.5), rendering error plots, and mosaicing/reprojecting/clipping DEM tiles.
- **Task 3**: Warping the landcover map to 10m UTM grid, using local mode focal majority matching to repair cloud pixel voids, rotating Landsat bands 90 degrees CCW (matching study area aspect ratio), running 2D fast cross-correlation using FFT to automatically calculate and apply precise spatial shift parameters ($dx=660m, dy=-18810m$), extracting water bodies and built-up land covers using MNDWI/BU index thresholds and size filters (>0.1km² for water, >1.0km² for built-up), updating landcover maps, and exporting areal statistics.

## Dependencies
This skill is powered by the unified python CLI helper script `scripts/gis_processor.py`. It requires the following packages, which are automatically resolved and isolated using `uv run`:
- `numpy`
- `pandas`
- `geopandas`
- `shapely`
- `rasterio`
- `scipy`
- `matplotlib`
- `pillow` (PIL)

## Quick Start
To run the entire end-to-end workflow in one click:
```bash
uv run scripts/gis_processor.py run-all --base-dir "path/to/raw/data" --output-dir "path/to/output/results"
```
This command automatically executes Tasks 1, 2, and 3 in sequence, saving all intermediate and final deliverables into a structured result directory.

## Utility Scripts

The skill exposes 5 specialized subcommands for fine-grained geoprocessing control:

### 1. Wild Animal Point Cleaning (`clean-animals`)
Cleans animal observations and creates the study area boundary.
```bash
uv run scripts/gis_processor.py clean-animals \
  --input "data/dongwu/animal.txt" \
  --output-gdb "results/result_data.gdb" \
  --buffer 2000.0
```
- `--input`: Path to the raw `animal.txt` file.
- `--output-gdb`: Output directory for the shapefiles.
- `--buffer`: Outer buffer distance in meters (default `2000.0`).

### 2. Meteorological Sampling Interpolation (`clean-weather`)
Processes meteorological sampling stations and generates high-resolution climate rasters.
```bash
uv run scripts/gis_processor.py clean-weather \
  --input "data/qixiang/Meteorological_sampling.txt" \
  --study-area "results/result_data.gdb/study_area.shp" \
  --output-gdb "results/result_data.gdb" \
  --resolution 30.0
```
- `--input`: Path to the raw weather sampling stations file.
- `--study-area`: Path to the `study_area.shp` file generated in Task 1.
- `--output-gdb`: Output directory.
- `--resolution`: Grid cell resolution in meters (default `30.0`).

### 3. DEM Mosaic and Clip (`process-dem`)
Mosaics individual elevation tiles, reprojects to UTM Zone 10N with bilinear resampling, and masks to the study area.
```bash
uv run scripts/gis_processor.py process-dem \
  --dem-dir "data/dem" \
  --study-area "results/result_data.gdb/study_area.shp" \
  --output-gdb "results/result_data.gdb" \
  --resolution 30.0
```
- `--dem-dir`: Directory containing raw `demA/B/C/D.tif` tiles.
- `--study-area`: Path to the `study_area.shp` file.
- `--output-gdb`: Output directory.

### 4. Landsat Image CCW Rotation & FFT Auto-Registration (`register-landsat`)
Performs 90-degree CCW rotation of Landsat, runs fast 2D cross-correlation (FFT) against the landcover water shoreline, aligns the spatial shift, and clips the 6-band image to the study area.
```bash
uv run scripts/gis_processor.py register-landsat \
  --input "data/yingx/Landsat.tif" \
  --landcover "results/temp_data.gdb/landcover_repaired.tif" \
  --output-gdb "results/result_data.gdb"
```
- `--input`: Path to raw input `Landsat.tif`.
- `--landcover`: Path to cloud-repaired landcover map.
- `--output-gdb`: Output directory.

### 5. Landcover Cloud Repair & Spectral Index Update (`update-landcover`)
Repairs landcover cloud pixels, extracts water and built-up land covers using MNDWI/BU spectral indexes, burns updates into the landcover map, and exports areal statistics.
```bash
uv run scripts/gis_processor.py update-landcover \
  --input "data/tudi/landcover.tif" \
  --dem "results/result_data.gdb/dem_clipped.tif" \
  --landsat "results/result_data.gdb/landsat_clipped.tif" \
  --output-gdb "results/result_data.gdb"
```
- `--input`: Path to raw unaligned `landcover.tif`.
- `--dem`: Path to clipped DEM raster.
- `--landsat`: Path to registered, georeferenced Landsat raster.
- `--output-gdb`: Output directory.

## Common Mistakes
- **Dead Proxy Settings in Git**: If `git clone` or `git push` fails with connection errors, verify your global git proxy setting. Use `git clone -c http.proxy= -c https.proxy= ...` to bypass dead local proxies.
- **Missing Study Area Shapefile**: Subcommands like `clean-weather` and `process-dem` rely on the generated `study_area.shp` for spatial boundaries. Make sure to run `clean-animals` first to generate it.
- **Memory Overhead in FFT**: When performing 2D cross-correlation on very large rasters, FFT computation can consume extensive RAM. The script uses an optimized 30m resampling step to perform the registration search in <2 seconds with minimal memory overhead, then reprojects the full-resolution bands using the calculated offset.
