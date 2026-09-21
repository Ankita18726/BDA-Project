import os
import pandas as pd
import matplotlib.pyplot as plt
def plot_popularity_distribution(df):

    print(
        "\nCreating popularity distribution chart..."
    )

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    popularity_counts = (
        df.groupBy("popularity_category")
        .count()
        .toPandas()
    )

    desired_order = [
        "Low",
        "Medium",
        "High"
    ]

    popularity_counts[
        "popularity_category"
    ] = pd.Categorical(
        popularity_counts[
            "popularity_category"
        ],
        categories=desired_order,
        ordered=True
    )

    popularity_counts = (
        popularity_counts
        .sort_values(
            "popularity_category"
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        popularity_counts[
            "popularity_category"
        ],
        popularity_counts["count"]
    )

    ax.set_title(
        "Distribution of Spotify Tracks "
        "by Popularity Category"
    )

    ax.set_xlabel(
        "Popularity Category"
    )

    ax.set_ylabel(
        "Number of Tracks"
    )

    # Put count above each bar
    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    output_path = (
        "outputs/charts/"
        "popularity_distribution.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Popularity distribution chart saved: "
        f"{output_path}"
    )
def plot_correlation_heatmap(df):

    print("\nCreating correlation heatmap...")

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    features = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_minutes"
    ]

    # Select only required numerical columns
    selected_df = df.select(features)

    # Convert to Pandas
    # 113k rows × 11 columns is manageable for this project,
    # but sampling also keeps visualization memory-friendly.
    pandas_df = (
        selected_df
        .sample(
            withReplacement=False,
            fraction=0.25,
            seed=42
        )
        .toPandas()
    )

    correlation_matrix = pandas_df.corr(
        numeric_only=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    image = ax.imshow(
        correlation_matrix,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(correlation_matrix.columns))
    )

    ax.set_yticks(
        range(len(correlation_matrix.columns))
    )

    ax.set_xticklabels(
        correlation_matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation_matrix.columns
    )

    # Add correlation values inside cells
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):

            value = correlation_matrix.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson Correlation"
    )

    ax.set_title(
        "Correlation Heatmap of Spotify Audio Features"
    )

    fig.tight_layout()

    output_path = (
        "outputs/charts/"
        "correlation_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Correlation heatmap saved: "
        f"{output_path}"
    )
def create_visualizations(
    analytics_results,
    correlation_results
):

    output_folder = "outputs/charts"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # ================================================
    # 1. Popularity Distribution
    # ================================================

    pdf = (
        analytics_results[
            "popularity_distribution"
        ]
        .toPandas()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        pdf["popularity_category"],
        pdf["count"]
    )

    plt.title(
        "Distribution of Spotify Track Popularity"
    )

    plt.xlabel(
        "Popularity Category"
    )

    plt.ylabel(
        "Number of Tracks"
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/01_popularity_distribution.png",
        dpi=300
    )

    plt.close()

    # ================================================
    # 2. Top Genres
    # ================================================

    pdf = (
        analytics_results[
            "top_genres"
        ]
        .toPandas()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        pdf["track_genre"],
        pdf["count"]
    )

    plt.title(
        "Top 10 Genres by Number of Tracks"
    )

    plt.xlabel(
        "Number of Tracks"
    )

    plt.ylabel(
        "Genre"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/02_top_genres.png",
        dpi=300
    )

    plt.close()

    # ================================================
    # 3. Genre Popularity
    # ================================================

    pdf = (
        analytics_results[
            "genre_popularity"
        ]
        .toPandas()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        pdf["track_genre"],
        pdf["average_popularity"]
    )

    plt.title(
        "Top Genres by Average Popularity"
    )

    plt.xlabel(
        "Average Popularity"
    )

    plt.ylabel(
        "Genre"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/03_genre_popularity.png",
        dpi=300
    )

    plt.close()

    # ================================================
    # 4. Average Energy
    # ================================================

    pdf = (
        analytics_results[
            "energy_analysis"
        ]
        .toPandas()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        pdf["popularity_category"],
        pdf["average_energy"]
    )

    plt.title(
        "Average Energy by Popularity Category"
    )

    plt.xlabel(
        "Popularity Category"
    )

    plt.ylabel(
        "Average Energy"
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/04_energy_vs_popularity.png",
        dpi=300
    )

    plt.close()

    # ================================================
    # 5. Danceability
    # ================================================

    pdf = (
        analytics_results[
            "dance_analysis"
        ]
        .toPandas()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        pdf["popularity_category"],
        pdf["average_danceability"]
    )

    plt.title(
        "Average Danceability by Popularity"
    )

    plt.xlabel(
        "Popularity Category"
    )

    plt.ylabel(
        "Average Danceability"
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/05_danceability_vs_popularity.png",
        dpi=300
    )

    plt.close()

    # ================================================
    # 6. Acousticness
    # ================================================

    pdf = (
        analytics_results[
            "acoustic_analysis"
        ]
        .toPandas()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        pdf["popularity_category"],
        pdf["average_acousticness"]
    )

    plt.title(
        "Average Acousticness by Popularity"
    )

    plt.xlabel(
        "Popularity Category"
    )

    plt.ylabel(
        "Average Acousticness"
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/06_acousticness_vs_popularity.png",
        dpi=300
    )

    plt.close()

    # ================================================
    # 7. Correlation
    # ================================================

    features = [
        item[0]
        for item in correlation_results
    ]

    correlations = [
        item[1]
        for item in correlation_results
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        features,
        correlations
    )

    plt.title(
        "Correlation of Audio Features with Popularity"
    )

    plt.xlabel(
        "Pearson Correlation"
    )

    plt.ylabel(
        "Audio Feature"
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/07_feature_correlation.png",
        dpi=300
    )

    plt.close()

    print(
        "\nCharts successfully saved in:",
        output_folder
    )

def plot_feature_importance(
    feature_importance
):

    import os
    import matplotlib.pyplot as plt

    print(
        "\nCreating feature importance chart..."
    )

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    features = [
        item[0]
        for item in feature_importance
    ]

    importances = [
        item[1]
        for item in feature_importance
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.barh(
        features,
        importances
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.invert_yaxis()

    for bar in bars:

        width = bar.get_width()

        ax.text(
            width,
            bar.get_y()
            + bar.get_height() / 2,
            f"{width:.3f}",
            va="center"
        )

    fig.tight_layout()

    output_path = (
        "outputs/charts/"
        "feature_importance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Feature importance chart saved: "
        f"{output_path}"
    )