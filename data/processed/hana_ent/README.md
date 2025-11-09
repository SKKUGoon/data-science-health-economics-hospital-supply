# Processed Data for Hana ENT Hospital

This directory contains processed data files generated from the Hana ENT hospital dataset. The data has been transformed through embedding generation, dimensionality reduction, and merging operations.

## File Naming Conventions

- **f2**: Data reduced to 2 factors using UMAP (2D dimensionality reduction)
- **f3**: Data reduced to 3 factors using UMAP (3D dimensionality reduction)
- **mdf**: Merged dataframe combining demographic information from the original data with UMAP-reduced features
- **meta**: Meta dataframe containing the real data behind the embeddings
- **supply**: Supply data derived from the ATC.csv file in `data/external/`

## File Descriptions

### UMAP-Reduced Embeddings

#### 2D Reductions (f2_*)
- `f2_dep.parquet`: Department embeddings reduced to 2 dimensions using UMAP
- `f2_diag.parquet`: Diagnosis embeddings reduced to 2 dimensions using UMAP
- `f2_pres.parquet`: Prescription embeddings reduced to 2 dimensions using UMAP

Each file contains columns: `umap1`, `umap2` (indexed by `id` and `date`)

#### 3D Reductions (f3_*)
- `f3_dep.parquet`: Department embeddings reduced to 3 dimensions using UMAP
- `f3_diag.parquet`: Diagnosis embeddings reduced to 3 dimensions using UMAP
- `f3_pres.parquet`: Prescription embeddings reduced to 3 dimensions using UMAP

Each file contains columns: `umap1`, `umap2`, `umap3` (indexed by `id` and `date`)

### Metadata Dataframes

- `meta_dep.parquet`: Real data behind the department embeddings
- `meta_diag.parquet`: Real data behind the diagnosis embeddings
- `meta_pres.parquet`: Real data behind the prescription embeddings

These files contain the original structured data that corresponds to the embedding vectors.

### Merged Dataframes

- `mdf_2d.parquet`: Merged dataframe combining:
  - Demographic information (id, date, sex, age) from the original dataset
  - 2D UMAP features from department, diagnosis, and prescription data
  
  Indexed by `['id', 'date']` with columns for demographic fields and UMAP features from all three categories.

- `mdf_3d.parquet`: Merged dataframe combining:
  - Demographic information (id, date, sex, age) from the original dataset
  - 3D UMAP features from department, diagnosis, and prescription data
  
  Indexed by `['id', 'date']` with columns for demographic fields and UMAP features from all three categories.

### Supply Data

- `supply.parquet`: Supply data extracted from the ATC.csv file located in `data/external/ATC_20250912_110516.csv`

## Data Processing Pipeline

1. **Embedding Generation**: Original hospital data is processed through `ChartElementEmbedding` to generate embeddings for departments, diagnoses, and prescriptions
2. **UMAP Reduction**: Embeddings are reduced to 2D and 3D using UMAP (see `pipeline/twpstep/hana_ent/prep.py`)
3. **Metadata Extraction**: Real data corresponding to embeddings is extracted and saved separately
4. **Merging**: Demographic data is merged with UMAP-reduced features (see `pipeline/twpstep/hana_ent/step1.py`)

## Usage

These processed files are used for:
- Clustering analysis (HDBSCAN)
- Dimensionality visualization (PCA)
- Patient pattern analysis
- Supply-demand modeling

