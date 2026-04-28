# Southeast Asia 10m Plantation Forest Extraction

A remote sensing-based project for extracting and analyzing plantation forests in Southeast Asia using multi-temporal satellite imagery and change detection algorithms.

## Project Structure

```
.
├── 01_data_preprocessing/      # Data download, preprocessing, and quality control
├── 02_gee_classification/      # GEE-based classification and change detection
├── 03_analysis/                # Statistical analysis and area calculations
├── 04_writing/                 # Figure creation for publications
├── logs/                       # Processing logs
└── output/                     # Output results
```

## Directory Descriptions

### 01_data_preprocessing
Data preprocessing pipeline including satellite data download, quality control, and format conversion.

| Subdirectory | Description |
|--------------|-------------|
| `02_ccdc_calculation/` | CCDC (Continuous Change Detection and Classification) and COLD algorithm calculations |
| `03_ccdc_data_1999_2019/` | CCDC data mosaic and processing (1999-2019) |
| `04_gadm_boundaries/` | GADM administrative boundary data for Southeast Asia countries |
| `05_sdpt_download/` | SDPT (StylerDPA ProdTitle) data download scripts |
| `06_sdpt_filtering/` | SDPT vector data filtering by region and country |
| `07_plantation_product_data/` | Plantation forest product data clipping and mosaicking |
| `08_sample_balancing/` | Sample balancing for classification grid data |
| `09_land_cover_data/` | ESRI land cover data filtering and processing |
| `10_filter_sdpt_polygons/` | SDPT polygon attribute filtering (betel palm, fruit, etc.) |
| `11_validation_data/` | Validation point data |

### 02_gee_classification
Google Earth Engine based classification and change detection.

| Subdirectory | Description |
|--------------|-------------|
| `01_iterative_sample_refinement/` | Iterative sample refinement for classification |
| `02_change_detection/` | Change detection using GEE |
| `03_data_statistics/` | Classification result merging and statistics |
| `04_result_statistics/` | Plantation/natural forest change analysis, land use transition matrices |
| `05_calculate_accuracy/` | Accuracy assessment calculations |
| `06_markov_model/` | Markov model for land cover transition probability estimation |

### 03_analysis
Post-classification statistical analysis.

| Subdirectory | Description |
|--------------|-------------|
| `01_fao_area_mapping/` | FAO area data mapping and comparison |
| `02_plantation_increase_stats/` | Plantation forest increase statistics |
| `03_country_area_stats/` | Country-level area statistics |
| `04_calculate_classification_accuracy/` | Classification accuracy calculations |
| `05_annual_area_error_bars/` | Annual area error bar calculations |
| `06_output_results/` | Analysis result outputs |

## Key Methods

- **CCDC (Continuous Change Detection and Classification)**: Temporal segmentation and change detection for Landsat imagery
- **COLD (Continuous Land Dynamics Detection)**: For gap-free land cover monitoring
- **GEE (Google Earth Engine)**: Cloud-based large-scale image processing
- **Markov Model**: Land cover transition probability estimation
- **Sankey/Alluvial/Chord Diagrams**: Land use conversion visualization

## Requirements

- Python 3.8+
- GDAL, rasterio, geopandas
- Google Earth Engine API (for GEE scripts)
- CCDC/pyxccd package (included in `01_data_preprocessing/02_ccdc_calculation/pyxccd/`)

## Usage

1. **Data Preprocessing**: Run scripts in `01_data_preprocessing/` to download and preprocess satellite data
2. **Classification**: Use GEE scripts in `02_gee_classification/` for classification
3. **Change Detection**: Run `02_gee_classification/02_change_detection/` scripts
4. **Analysis**: Use `03_analysis/` scripts for statistical analysis and visualization

## Output

Results are stored in `output/` directory, including:
- Classification maps
- Change detection results
- Statistical summaries
- Visualization figures

## License

For research and educational purposes.

## Citation

If you use this code or methodology, please cite appropriately.
