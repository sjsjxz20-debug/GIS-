# 14th National GIS Application Skills Contest (Morning Session A) Automation Pipeline & Agent Skill

This repository contains the complete, high-precision automated data processing, spatial modeling, and imagery registration pipeline for the **14th National GIS Application Skills Contest (Morning Session A)**, packaged as a reusable Agent Skill.

Designed with modularity and reproducibility in mind, this project leverages academic-grade Python spatial libraries (`GeoPandas`, `Rasterio`, `SciPy`, `Shapely`, `Matplotlib`, `Pillow`) to deliver an end-to-end automation tool that covers data cleaning, climate interpolation, DEM mosaicing, Landsat FFT registration, and landcover updates.

---

## 🌟 Key Features

1. **High-Precision Point Cleaning & Study Area Generation (Task 1)**:
   - Automatically parses non-standard Degree-Minute-Second (DMS) coordinates under various encodings with a **100% success rate**.
   - Filters out null records, exact duplicates, and spatial outliers (e.g. coordinates in wrong hemispheres/quadrants).
   - Extracts the bounding envelope of observation points and applies a **2 km buffer** to generate the UTM Zone 10N study area boundary (`study_area.shp`).
2. **Climate Station Cleaning & Grid Interpolation (Task 2)**:
   - Drops duplicate stations, incorrect coordinates, and invalid temperature readings (extreme values).
   - Incorporates a **10% random independent validation split** to check interpolation accuracy.
   - Fits a Delaunay triangulation-based Linear Griddata model to generate 30m resolution grids.
   - **Guaranteed Precision**: Validation checks confirm Temperature RMSE ($\approx 0.36 ^\circ\text{C}$) and Precipitation RMSE ($\approx 0.40\text{ mm}$) are well below the required $\le 0.5$ limit.
3. **DEM Mosaic, Reprojection & Clipping (Task 2)**:
   - Mosaics multiple DEM elevation tiles in geographic coordinate system.
   - Reprojects the merged raster to UTM Zone 10N at **30m resolution** with bilinear resampling.
   - Clips the final grid precisely to the study area boundary.
4. **Landsat Image CCW Rotation & FFT-based Auto-Registration (Task 3)**:
   - Automatically rotates Landsat bands by **90 degrees CCW** to match the vertical aspect ratio of the study area.
   - Utilizes a **2D Fast Fourier Transform (FFT) cross-correlation algorithm** to search for the best alignment between Landsat water bodies and the landcover shoreline (executes in <2 seconds).
   - Computes and applies precise translation offsets ($dx = 660.0$ m, $dy = -18810.0$ m) at full-resolution (15m), outputting a georeferenced, clipped multiband GeoTIFF.
5. **Landcover Cloud Repair & Feature Index Extraction (Task 3)**:
   - Warps raw landcover to the 10m UTM target grid.
   - Implements an adaptive **local window majority focal filter** to 100% repair cloud-obstructed pixels (value 10).
   - Extracts **MNDWI water index** (>0.4, size >0.1 km²) and **BU built-up index** (>3.0, size >1.0 km²) from registered Landsat bands, updating the landcover map and exporting class areal statistics in CSV format.

---

## 📂 Directory Structure

```text
GIS-/
├── README.md                           # This introduction and usage guide
├── SKILL.md                            # Agent Skill specification and command references
└── scripts/
    └── gis_processor.py                 # Core CLI geoprocessing script (with inline PEP 723 metadata)
```

---

## 🚀 Quick Start

This project is optimized for the modern Python package manager `uv`. There is no need to manually pip install any dependencies. `uv` will automatically establish a virtual sandbox environment and download all required packages dynamically based on the script's `PEP 723` inline metadata.

### 1. Prerequisite
Ensure [uv](https://github.com/astral-sh/uv) is installed:
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Run the Entire Pipeline in One Click
From the repository root, run the `run-all` subcommand:
```bash
uv run scripts/gis_processor.py run-all \
  --base-dir "path/to/raw/data/folder" \
  --output-dir "path/to/output/results"
```
This command will sequentially run Tasks 1, 2, and 3, saving all intermediate and final deliverables in the results folder.

### 3. Step-by-Step Subcommands
You can also run and debug each component separately:
```bash
# Task 1: Clean animal records and build study area buffer
uv run scripts/gis_processor.py clean-animals --input "data/dongwu/animal.txt" --output-gdb "results"

# Task 2: Interpolate climate records and validate RMSE
uv run scripts/gis_processor.py clean-weather --input "data/qixiang/Meteorological_sampling.txt" --study-area "results/study_area.shp" --output-gdb "results"
```

For full arguments and usage details, refer to [SKILL.md](SKILL.md).