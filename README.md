# Beyond the Beat

## Big Data Analytics of Spotify Tracks

**Beyond the Beat** is an end-to-end Big Data Analytics project that analyzes Spotify track data to identify patterns in musical characteristics, genres, and track popularity. The project combines **Apache Spark, PySpark, Spark SQL, Machine Learning, FastAPI, and React** to build a scalable analytics pipeline and an interactive dashboard for presenting the results.

The system demonstrates how large-scale music datasets can be processed, analyzed, modeled, and transformed into meaningful insights using modern Big Data technologies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Project Workflow](#project-workflow)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Data Processing and Analytics](#data-processing-and-analytics)
- [Machine Learning](#machine-learning)
- [Model Evaluation](#model-evaluation)
- [Data Visualization](#data-visualization)
- [Interactive Dashboard](#interactive-dashboard)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)

---

## Project Overview

Music streaming platforms generate large volumes of data containing information about tracks, artists, genres, popularity, and audio characteristics. Analyzing such datasets requires efficient processing frameworks capable of handling large-scale structured data.

**Beyond the Beat** uses **Apache Spark** as the primary distributed data processing framework. Spotify track data is processed using **PySpark DataFrames and Spark SQL**, followed by exploratory analysis, feature engineering, statistical analysis, and machine learning.

The project investigates several audio characteristics, including:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Track duration

These features are analyzed against **track popularity** to identify relationships and broader patterns within the dataset.

The project further implements multiple regression models for popularity prediction and presents the analytical results through a **FastAPI backend and React-based dashboard**.

---

## Problem Statement

Spotify contains a large and diverse collection of tracks with varying musical characteristics and popularity levels. Understanding the factors associated with track popularity requires processing and analyzing multiple attributes across a large dataset.

The project addresses the following analytical questions:

- How is track popularity distributed across the dataset?
- Which genres have the highest average popularity?
- How do different audio features relate to track popularity?
- Does track duration show any relationship with popularity?
- Which audio features contribute most to popularity prediction?
- How effectively can machine learning models estimate track popularity?
- Which regression model provides the best predictive performance?
- How can Apache Spark be used to efficiently process and analyze music data at scale?

---

## Objectives

The primary objectives of **Beyond the Beat** are to:

1. Build a scalable data analytics pipeline using **Apache Spark and PySpark**.
2. Clean and preprocess raw Spotify track data.
3. Perform exploratory and descriptive analysis using **Spark DataFrames and Spark SQL**.
4. Analyze genre-wise, popularity-wise, and duration-wise patterns.
5. Examine relationships between audio characteristics and track popularity.
6. Engineer relevant features for analytical and machine learning tasks.
7. Develop regression models for predicting track popularity.
8. Compare model performance using standard regression metrics.
9. Analyze feature importance to identify influential audio characteristics.
10. Generate meaningful visualizations from the analytical results.
11. Expose processed results through a **FastAPI REST API**.
12. Present insights through an interactive **React dashboard**.

---

## System Architecture

The project follows a layered architecture that separates data processing, analytics, machine learning, API services, and visualization.

```text
                        Spotify Dataset
                               |
                               v
                    Apache Spark / PySpark
                               |
                               v
                   Data Cleaning & Validation
                               |
                               v
                      Feature Engineering
                               |
                               v
                 Spark DataFrame Processing
                               |
                +--------------+--------------+
                |                             |
                v                             v
           Spark SQL                    ML Pipeline
                |                             |
                v                             v
      Descriptive Analytics        Regression Models
                |                             |
                v                             v
        Analytical Results           Model Evaluation
                |                             |
                +--------------+--------------+
                               |
                               v
                     Processed Outputs
                               |
                               v
                         FastAPI API
                               |
                               v
                    React + Vite Dashboard
                               |
                               v
                  Interactive Visualizations
```

This architecture allows the analytical pipeline and presentation layer to remain independent while maintaining a clear flow from raw data to actionable insights.

---

## Project Workflow

The project follows an end-to-end Big Data Analytics workflow.

### 1. Data Ingestion

The Spotify dataset is loaded into **Apache Spark** using PySpark and represented as distributed Spark DataFrames.

### 2. Data Cleaning

The raw dataset is inspected and processed to handle:

- Missing values
- Duplicate records
- Invalid data
- Inconsistent data types
- Unnecessary attributes

### 3. Feature Engineering

Relevant attributes are transformed or derived to support further analysis. This includes preparing numerical audio features and creating categories required for analytical tasks.

### 4. Distributed Data Processing

PySpark DataFrame operations are used to perform:

- Filtering
- Aggregation
- Grouping
- Sorting
- Statistical computations
- Feature transformations

### 5. Spark SQL Analytics

Spark SQL is used to execute structured analytical queries over the processed dataset and extract meaningful statistics and patterns.

### 6. Exploratory Data Analysis

The processed data is analyzed to study:

- Popularity distribution
- Genre-level popularity
- Audio feature distributions
- Duration patterns
- Feature correlations
- Relationships between audio characteristics and popularity

### 7. Machine Learning

The processed dataset is transformed into an ML-ready feature representation and used to train multiple regression algorithms.

### 8. Model Evaluation

The trained models are evaluated and compared using standard regression metrics.

### 9. Visualization

Analytical outputs and machine learning results are converted into charts and visual representations for easier interpretation.

### 10. Dashboard Integration

Processed results are exposed through a **FastAPI backend** and consumed by a **React + Vite frontend** to provide an interactive analytical dashboard.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python |
| Big Data Processing | Apache Spark |
| Distributed Analytics | PySpark |
| Query Processing | Spark SQL |
| Machine Learning | Spark MLlib |
| Data Processing | Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Backend | FastAPI |
| API Architecture | REST |
| Frontend | React |
| Build Tool | Vite |
| Styling | CSS |
| Data Format | CSV / JSON |

---

## Dataset

The project uses a Spotify tracks dataset containing metadata and audio characteristics for a large collection of songs.

### Key Attributes

| Attribute | Description |
|---|---|
| `track_name` | Name of the track |
| `artists` | Artist or artists associated with the track |
| `track_genre` | Genre classification |
| `popularity` | Spotify popularity score |
| `duration_ms` | Track duration in milliseconds |
| `danceability` | Suitability of the track for dancing |
| `energy` | Perceptual measure of intensity and activity |
| `loudness` | Overall loudness of the track |
| `speechiness` | Presence of spoken words |
| `acousticness` | Confidence that the track is acoustic |
| `instrumentalness` | Likelihood that the track contains no vocals |
| `liveness` | Presence of an audience or live performance |
| `valence` | Musical positivity of the track |
| `tempo` | Estimated tempo in beats per minute |

The dataset is processed through Spark before being used for analytics and machine learning.

---

## Data Processing and Analytics

Apache Spark forms the core of the analytical pipeline.

### PySpark DataFrame Processing

PySpark is used for distributed operations including:

- Schema inspection
- Data type conversion
- Null-value handling
- Duplicate removal
- Filtering
- Aggregation
- Group-based analysis
- Feature transformation
- Statistical computation

### Spark SQL

The processed Spark DataFrame is registered as a temporary SQL view, allowing analytical queries to be executed using Spark SQL.

Spark SQL is used for analyses such as:

- Average popularity by genre
- Track count by genre
- Popularity category distribution
- Duration-based analysis
- Feature-level aggregation
- Top-performing genres and tracks

Using both **PySpark DataFrames and Spark SQL** demonstrates two complementary approaches to distributed Big Data analytics.

---

## Machine Learning

The project uses **Spark MLlib** to develop regression models for predicting track popularity from audio characteristics.

### Input Features

The machine learning pipeline considers numerical audio attributes such as:

```text
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
duration_ms
```

The selected attributes are assembled into a feature vector using Spark's machine learning pipeline.

### Regression Models

The following algorithms are evaluated:

**Linear Regression**

Provides a baseline model and helps determine whether track popularity can be represented using linear relationships between audio features.

**Random Forest Regression**

Uses an ensemble of decision trees to capture nonlinear relationships and interactions between features.

**Gradient-Boosted Tree Regression**

Builds decision trees sequentially to progressively reduce prediction error and model more complex relationships within the data.

---

## Model Evaluation

Model performance is compared using the following regression metrics:

| Metric | Purpose |
|---|---|
| **RMSE** | Measures the magnitude of prediction errors while penalizing larger errors |
| **MAE** | Measures the average absolute prediction error |
| **R² Score** | Measures the proportion of variance explained by the model |

The comparison helps determine which model provides the strongest predictive performance for the available Spotify features.

Feature importance analysis is additionally performed for tree-based models to understand which audio attributes contribute most strongly to their predictions.

---

## Data Visualization

Visualizations are generated to transform analytical outputs into interpretable insights.

The project includes visual analysis of:

- Popularity distribution
- Genre-wise average popularity
- Audio feature correlations
- Track duration distribution
- Popularity categories
- Audio feature relationships
- Machine learning model performance
- Feature importance

These visualizations provide a concise representation of patterns identified during the Spark-based analysis.

---

## Interactive Dashboard

The project includes a web-based dashboard for presenting the analytical results.

### Backend

The backend is developed using **FastAPI** and provides REST endpoints for accessing processed analytical outputs.

Responsibilities include:

- Serving analytics results
- Providing visualization data
- Exposing model evaluation results
- Providing structured JSON responses to the frontend

### Frontend

The frontend is developed using **React and Vite**.

The dashboard provides a centralized interface for exploring:

- Dataset statistics
- Popularity analysis
- Genre-level insights
- Audio feature relationships
- Machine learning results
- Feature importance
- Analytical visualizations

The dashboard separates the computational analytics layer from the presentation layer, allowing results generated by the Spark pipeline to be consumed through a modern web interface.

---

## Project Structure

```text
Beyond-the-Beat/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- notebooks/
|   `-- spotify_analysis.ipynb
|
|-- src/
|   |-- data_processing/
|   |-- analytics/
|   |-- machine_learning/
|   `-- visualization/
|
|-- backend/
|   |-- main.py
|   `-- requirements.txt
|
|-- frontend/
|   |-- src/
|   |-- public/
|   |-- package.json
|   `-- vite.config.js
|
|-- outputs/
|   |-- analytics/
|   |-- models/
|   `-- visualizations/
|
|-- README.md
`-- requirements.txt
```

> **Note:** The directory structure may vary depending on the deployment or development environment.

---

## Installation and Setup

### Prerequisites

Ensure the following are installed:

- Python 3.x
- Java Development Kit (JDK)
- Apache Spark
- Node.js
- npm

### Clone the Repository

```bash
git clone <repository-url>
cd Beyond-the-Beat
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## Running the Project

### Run the Analytics Pipeline

Execute the primary PySpark analysis script or notebook:

```bash
python <analysis-script>.py
```

Alternatively, run the project notebook in the configured Spark environment.

### Start the FastAPI Backend

```bash
cd backend
uvicorn main:app --reload
```

The API will start on the configured local development server.

### Start the React Dashboard

Open a separate terminal:

```bash
cd frontend
npm run dev
```

The Vite development server will provide the local URL for accessing the dashboard.

---

## Key Findings

The analysis is designed to identify patterns across Spotify tracks rather than assume that popularity can be explained by a single musical characteristic.

Key analytical observations include:

- Popularity varies considerably across tracks and genres.
- Individual audio characteristics may show limited direct correlation with popularity.
- Relationships between musical attributes and popularity can be nonlinear.
- Genre-level aggregation provides additional context beyond individual feature correlations.
- Tree-based regression models can capture feature interactions that are not represented by simple linear models.
- Feature importance analysis helps identify which attributes contribute most to model predictions.
- Audio characteristics alone may not fully explain track popularity, suggesting that external factors also play an important role.

> Exact findings and model-performance values should be interpreted from the outputs generated by the final dataset and execution of the analytical pipeline.

---

## Limitations

The project has several limitations:

- Spotify popularity is influenced by factors beyond audio characteristics.
- Artist popularity, marketing, playlist placement, release timing, and social trends are not fully represented by audio features.
- Dataset quality and representativeness directly affect analytical results.
- Correlation between a feature and popularity does not imply causation.
- Machine learning predictions are limited by the features available in the dataset.
- Results represent patterns in the analyzed dataset and should not automatically be generalized to the entire music industry.

---

## Future Scope

The project can be extended through:

- Integration with live music-platform APIs
- Analysis of artist-level and album-level characteristics
- Incorporation of release dates and temporal popularity trends
- Addition of playlist and streaming statistics
- Advanced feature engineering
- Hyperparameter optimization
- Additional ensemble and deep learning models
- Recommendation-system development
- Sentiment analysis of lyrics
- Real-time or streaming analytics using Spark Structured Streaming
- Cloud-based distributed Spark deployment
- Interactive prediction capabilities through the dashboard

---

## Conclusion

**Beyond the Beat** demonstrates an end-to-end Big Data Analytics workflow for analyzing Spotify track data.

The project combines **Apache Spark, PySpark, Spark SQL, Spark MLlib, FastAPI, and React** to cover the complete lifecycle from raw data processing to interactive presentation. Distributed data processing enables scalable analysis, while machine learning provides a framework for examining the predictive relationship between audio characteristics and track popularity.

Rather than treating popularity as the result of a single musical attribute, the project uses descriptive analytics, feature analysis, and predictive modeling to examine how multiple characteristics interact within the dataset.

Overall, the project demonstrates the practical integration of **Big Data processing, analytical querying, machine learning, data visualization, API development, and frontend technologies** within a unified music analytics system.
