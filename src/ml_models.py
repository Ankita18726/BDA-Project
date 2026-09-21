import os
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.feature import (
    VectorAssembler
)

from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)

from pyspark.ml.evaluation import (
    RegressionEvaluator
)

def prepare_ml_data(df):

    feature_columns = [
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

    # Remove repeated Spotify tracks
    unique_tracks_df = (
        df.dropDuplicates(
            ["track_id"]
        )
    )

    print(
        "\nUnique tracks used for ML:",
        unique_tracks_df.count()
    )

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="skip"
    )

    ml_data = assembler.transform(
        unique_tracks_df
    )

    ml_data = ml_data.select(
        "features",
        "popularity"
    )

    return (
        ml_data,
        feature_columns
    )

def evaluate_model(
    predictions,
    model_name
):

    rmse_evaluator = RegressionEvaluator(
        labelCol="popularity",
        predictionCol="prediction",
        metricName="rmse"
    )

    mae_evaluator = RegressionEvaluator(
        labelCol="popularity",
        predictionCol="prediction",
        metricName="mae"
    )

    r2_evaluator = RegressionEvaluator(
        labelCol="popularity",
        predictionCol="prediction",
        metricName="r2"
    )

    rmse = rmse_evaluator.evaluate(
        predictions
    )

    mae = mae_evaluator.evaluate(
        predictions
    )

    r2 = r2_evaluator.evaluate(
        predictions
    )

    print(
        f"\n{model_name}"
    )

    print("-" * 50)

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def train_models(df):

    print("\n")
    print("=" * 70)
    print("SPARK MLLIB POPULARITY PREDICTION")
    print("=" * 70)

    ml_data, feature_columns = (
        prepare_ml_data(df)
    )

    # ================================================
    # Train Test Split
    # ================================================

    train_data, test_data = (
        ml_data.randomSplit(
            [0.8, 0.2],
            seed=42
        )
    )

    print(
        "\nTraining records:",
        train_data.count()
    )

    print(
        "Testing records:",
        test_data.count()
    )

    # ================================================
    # LINEAR REGRESSION
    # ================================================

    print(
        "\nTraining Linear Regression..."
    )

    linear_regression = (
        LinearRegression(
            featuresCol="features",
            labelCol="popularity",
            maxIter=100
        )
    )

    lr_model = (
        linear_regression.fit(
            train_data
        )
    )

    lr_predictions = (
        lr_model.transform(
            test_data
        )
    )

    print(
        "\nLinear Regression Predictions:"
    )

    lr_predictions.select(
        "popularity",
        "prediction"
    ).show(10)

    lr_metrics = evaluate_model(
        lr_predictions,
        "Linear Regression"
    )

    # ================================================
    # RANDOM FOREST
    # ================================================

    print(
        "\nTraining Random Forest Regression..."
    )

    random_forest = (
        RandomForestRegressor(
            featuresCol="features",
            labelCol="popularity",
            numTrees=100,
            maxDepth=8,
            seed=42
        )
    )

    rf_model = (
        random_forest.fit(
            train_data
        )
    )

    rf_predictions = (
        rf_model.transform(
            test_data
        )
    )

    print(
        "\nRandom Forest Predictions:"
    )

    rf_predictions.select(
        "popularity",
        "prediction"
    ).show(10)

    rf_metrics = evaluate_model(
        rf_predictions,
        "Random Forest Regression"
    )

    # ================================================
    # Feature importance
    # ================================================

    print(
        "\nRANDOM FOREST FEATURE IMPORTANCE"
    )

    print("-" * 60)

    importances = (
        rf_model.featureImportances.toArray()
    )

    feature_importance = list(
        zip(
            feature_columns,
            importances
        )
    )

    feature_importance.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in feature_importance:

        print(
            f"{feature:20} {importance:.4f}"
        )
        # ================================================
    # GRADIENT BOOSTED TREES
    # ================================================

    print(
        "\nTraining Gradient Boosted Trees Regression..."
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="popularity",
        predictionCol="prediction",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )

    gbt_model = gbt.fit(
        train_data
    )

    gbt_predictions = gbt_model.transform(
        test_data
    )

    print(
        "\nGradient Boosted Trees Predictions:"
    )

    gbt_predictions.select(
        "popularity",
        "prediction"
    ).show(
        10,
        truncate=False
    )

    gbt_metrics = evaluate_model(
        gbt_predictions,
        "Gradient Boosted Trees"
    )

    # ================================================
    # GBT FEATURE IMPORTANCE
    # ================================================

    print(
        "\nGRADIENT BOOSTED TREES FEATURE IMPORTANCE"
    )

    print("-" * 60)

    gbt_importances = (
        gbt_model.featureImportances.toArray()
    )

    gbt_feature_importance = list(
        zip(
            feature_columns,
            gbt_importances
        )
    )

    gbt_feature_importance.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in gbt_feature_importance:

        print(
            f"{feature:20} {importance:.4f}"
        )

    return {
        "linear_regression":
            lr_metrics,

        "random_forest":
            rf_metrics,

        "gbt":
            gbt_metrics,

        "feature_importance":
            feature_importance,

        "gbt_feature_importance":
            gbt_feature_importance,

        "rf_predictions":
            rf_predictions,

        "gbt_predictions":
            gbt_predictions
    }
def save_model_results(results):

    os.makedirs(
        "outputs/results",
        exist_ok=True
    )

    metrics = [
        results["linear_regression"],
        results["random_forest"],
        results["gbt"]
    ]

    comparison_df = pd.DataFrame(
        metrics
    )

    comparison_df.to_csv(
        "outputs/results/model_comparison.csv",
        index=False
    )

    create_model_comparison_charts(
        comparison_df
    )

    print(
        "\nModel comparison saved to "
        "outputs/results/model_comparison.csv"
    )

def create_model_comparison_charts(
    comparison_df
):

    os.makedirs(
        "outputs/charts",
        exist_ok=True
    )

    # -----------------------------
    # RMSE
    # -----------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df["RMSE"]
    )

    ax.set_title(
        "RMSE Comparison of "
        "Popularity Prediction Models"
    )

    ax.set_ylabel("RMSE")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for bar in bars:

        value = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        "outputs/charts/"
        "model_rmse_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -----------------------------
    # MAE
    # -----------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df["MAE"]
    )

    ax.set_title(
        "MAE Comparison of "
        "Popularity Prediction Models"
    )

    ax.set_ylabel("MAE")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for bar in bars:

        value = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        "outputs/charts/"
        "model_mae_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -----------------------------
    # R2
    # -----------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df["R2"]
    )

    ax.set_title(
        "R² Comparison of "
        "Popularity Prediction Models"
    )

    ax.set_ylabel("R²")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for bar in bars:

        value = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        "outputs/charts/"
        "model_r2_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Model comparison charts saved."
    )

