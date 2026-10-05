import os
from src.dashboard_export import export_dashboard_data
from src.output_verification import (
    verify_project_outputs
)
from src.spark_session import (
    create_spark_session
)

from src.data_cleaning import (
    clean_spotify_data
)

from src.analytics import (
    run_analytics,
    calculate_correlations
)

from src.spark_sql_analysis import (
    run_spark_sql_analysis
)

from src.visualization import (
    create_visualizations,
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_popularity_distribution
)

from src.ml_models import (
    train_models,
    save_model_results,
    
)


def main():

    print("=" * 70)
    print("BEYOND THE BEAT")
    print("BIG DATA ANALYTICS OF SPOTIFY TRACKS")
    print("=" * 70)

    # =================================================
    # STEP 1
    # Create Spark Session
    # =================================================

    spark = create_spark_session()

    print(
        "\nSpark Session successfully created."
    )

    # =================================================
    # STEP 2
    # Dataset Path
    # =================================================

    dataset_path = (
        "data/raw/spotify_tracks.csv"
    )

    if not os.path.exists(
        dataset_path
    ):

        print(
            "\nERROR: Dataset not found."
        )

        print(
            "Place your CSV file at:"
        )

        print(
            dataset_path
        )

        spark.stop()

        return

    # =================================================
    # STEP 3
    # Read Dataset
    # =================================================

    print(
        "\nLoading Spotify dataset..."
    )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(dataset_path)
    )
    # Remove unnecessary index column created in the CSV
    if "_c0" in df.columns:
        df = df.drop("_c0")

    print(
        "\nDataset loaded successfully."
    )

    # =================================================
    # STEP 4
    # Display Schema
    # =================================================

    print(
        "\nDATASET SCHEMA"
    )

    df.printSchema()

    print(
        "\nFIRST 5 RECORDS"
    )

    df.show(
        5,
        truncate=False
    )

    print(
        "\nDataset rows:",
        df.count()
    )

    print(
        "Dataset columns:",
        len(df.columns)
    )

    # =================================================
    # STEP 5
    # Cleaning
    # =================================================

    cleaned_df = (
        clean_spotify_data(df)
    )

    # Cache because data will be reused
    cleaned_df.cache()

    # =================================================
    # STEP 6
    # Save Cleaned Dataset
    # =================================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # (
    #     cleaned_df
    #     .coalesce(1)
    #     .write
    #     .mode("overwrite")
    #     .option("header", True)
    #     .csv(
    #         "data/processed/cleaned_spotify"
    #     )
    # )

    print(
        "\nCleaned data saved."
    )

    # =================================================
    # STEP 7
    # Descriptive Analytics
    # =================================================

    analytics_results = (
        run_analytics(
            cleaned_df
        )
    )

    # =================================================
    # STEP 8
    # Correlation Analysis
    # =================================================

    correlation_results = (
        calculate_correlations(
            cleaned_df
        )
    )

    # =================================================
    # STEP 9
    # Spark SQL
    # =================================================

    run_spark_sql_analysis(
        spark, cleaned_df
    )

    # =================================================
    # STEP 10
    # Visualizations
    # =================================================

    create_visualizations(
        analytics_results,
        correlation_results
    )
    plot_correlation_heatmap(
    cleaned_df
)
    plot_popularity_distribution(
    cleaned_df
)

    # =================================================
    # STEP 11
    # Machine Learning
    # =================================================

    ml_results = (
        train_models(
            cleaned_df
        )
    )

    # =================================================
    # STEP 12
    # Feature Importance Chart
    # =================================================

    plot_feature_importance(
        ml_results[
            "feature_importance"
        ]
    )
    
    # =================================================
    # STEP 13
    # Save Model Results
    # =================================================

    save_model_results(
        ml_results
    )
    outputs_ok = verify_project_outputs()
    export_dashboard_data(cleaned_df)
    print(
    "\n" + "=" * 70
)

    if outputs_ok:

        print(
        "PROJECT EXECUTION COMPLETED "
        "SUCCESSFULLY"
    )

    else:

        print(
        "PROJECT EXECUTION COMPLETED "
        "WITH MISSING OUTPUTS"
    )

    print(
    "=" * 70
)

    spark.stop()
    


if __name__ == "__main__":
    main()