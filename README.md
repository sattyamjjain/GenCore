# GenCore

GenCore is a data-driven AI project built to process and analyze large-scale health data from sources like Fitbit. This
project leverages advanced techniques such as FAISS-based vector retrieval, clustering of embeddings, and statistical
analysis to extract meaningful insights from activity data. It focuses on using machine learning for physical health,
providing tools for embedding generation, clustering, and retrieval.

## Features

- Data Ingestion: Handles importing and preprocessing of raw Fitbit datasets.
- Embedding Generation: Generates activity embeddings using a custom embedding model.
- Clustering: Performs clustering on the generated embeddings to find meaningful patterns in activity data.
- FAISS-Based Retrieval: Uses FAISS (Facebook AI Similarity Search) for fast retrieval of similar embeddings on GPU.
- Cluster Analysis: Provides statistical analysis of the clusters formed from embeddings.
- Visualization: Generates plots of the cluster distributions and embeddings.

## Prerequisites

To run this project, you'll need:

- Python 3.7+
- Anaconda or virtualenv for environment management
- NVIDIA GPU with CUDA support for FAISS GPU acceleration (Optional but recommended)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sattyamjjain/GenCore.git
   cd GenCore

2. Create a virtual environment and activate it:

   ```bash
    conda create -n GenCore python=3.11
    conda activate GenCore

3. Install the required dependencies:

   ```bash
    pip install -r requirements.txt

For GPU acceleration with FAISS, ensure your system has the correct CUDA toolkit installed.

## Key Files:

- `data/fitbit`: Contains data files such as activity embeddings, calories, and sleep data.
- `scripts/`: Contains Python scripts for FAISS embeddings, clustering analysis, and visualization.

## Usage

### Data Ingestion

The data ingestion step involves loading the Fitbit activity data. This data is stored in various CSV and JSON files
under the data/fitbit/ directory. The data_ingestion.py script handles reading and processing the raw data files into
usable formats for further processing.

    python3 scripts/data_ingestion.py

### Generate Activity Embeddings

Once the data is ingested, we generate embeddings that represent the activity data in a vectorized form. This is
achieved using machine learning models to process the data into a meaningful embedding space.

    python3 scripts/generate_activity_embeddings.py

### FAISS GPU Embedding Search

We apply clustering techniques such as KMeans to the activity embeddings to group similar activities. These clusters are
saved for further analysis.

    python3 scripts/faiss_gpu_embeddings.py

### Cluster Activity Embeddings

We apply clustering techniques such as KMeans to the activity embeddings to group similar activities. These clusters are
saved for further analysis.

    python3 scripts/cluster_activity_embeddings.py

### Cluster Analysis

The analyze_clusters.py script allows us to analyze the generated clusters. It calculates statistics like mean and
standard deviation of activity metrics for each cluster and provides visualizations of the cluster distributions.

The visualize_embeddings.py script is used to visualize the activity embeddings in a lower-dimensional space, typically
using t-SNE or PCA for dimensionality reduction. This allows us to inspect how well the embeddings represent different
activities and their clustering.

    python3 scripts/analyze_clusters.py
    python3 scripts/visualize_embeddings.py

### Retrieval Using FAISS

Once the embeddings are loaded into FAISS, we can retrieve similar embeddings based on a query vector. The
faiss_retrieval.py script enables querying by either an index or a custom embedding vector.

    python3 scripts/faiss_retrieval.py

### FAISS GPU Embeddings

The faiss_gpu_embeddings.py script transfers the generated activity embeddings to GPU memory using FAISS for fast
similarity searches. This script enables us to search for embeddings similar to a query efficiently.

    python3 scripts/faiss_gpu_embeddings.py

## Project Workflow

- Data Ingestion: Fitbit data is loaded from JSON and CSV files.
- Embedding Generation: Activity embeddings are generated based on the input data.
- Clustering: Activity embeddings are clustered to group similar activities.
- Analysis: Cluster statistics are computed, and cluster results are visualized.
- Similarity Search: FAISS is used to perform fast similarity searches on the embeddings.

## Datasets Used

### Fitbit Dataset:

Activity, sleep, heart rate, and calorie data sourced from the Fitbit dataset on Kaggle.

## Future Work

- Model Deployment: Add deployment to AWS Lambda using AWS API Gateway.
- Real-Time Data Processing: Enable real-time ingestion and analysis of health data.
- Advanced Retrieval Techniques: Implement additional retrieval methods such as semantic similarity with sentence embeddings.
- Additional Clustering Methods: Explore other clustering algorithms like DBSCAN or hierarchical clustering.

#### Feel free to contribute, raise issues, or provide suggestions for further improvement.