from pyspark.sql.functions import (
    col,
    when,
    round,
    trim
)


def clean_spotify_data(df):

    print("\n========== INITIAL DATASET ==========")

    print("Rows:", df.count())
    print("Columns:", len(df.columns))

    # =================================================
    # 1. Remove unwanted CSV index column
    # =================================================

    if "_c0" in df.columns:
        df = df.drop("_c0")

    # =================================================
    # 2. Convert numerical columns to proper data types
    # =================================================

    numeric_columns = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature"
    ]

    for column_name in numeric_columns:

        if column_name in df.columns:

            df = df.withColumn(
                column_name,
                col(column_name).cast("double")
            )

    # =================================================
    # 3. Remove duplicate rows
    # =================================================

    df = df.dropDuplicates()

    # =================================================
    # 4. Remove NULL values from important columns
    # =================================================

    important_columns = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "track_genre"
    ]

    existing_columns = [
        column_name
        for column_name in important_columns
        if column_name in df.columns
    ]

    df = df.dropna(
        subset=existing_columns
    )

    # =================================================
    # 5. Validate popularity
    # =================================================

    df = df.filter(
        (col("popularity") >= 0) &
        (col("popularity") <= 100)
    )

    # =================================================
    # 6. Validate 0-1 audio features
    # =================================================

    zero_to_one_features = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence"
    ]

    for feature in zero_to_one_features:

        if feature in df.columns:

            df = df.filter(
                (col(feature) >= 0) &
                (col(feature) <= 1)
            )

    # =================================================
    # 7. Validate tempo
    # =================================================

    if "tempo" in df.columns:

        df = df.filter(
            (col("tempo") > 0) &
            (col("tempo") < 300)
        )

    # =================================================
    # 8. Validate duration
    # =================================================

    df = df.filter(
        col("duration_ms") > 0
    )

    # =================================================
    # 9. Clean genre text
    # =================================================

    if "track_genre" in df.columns:

        df = df.withColumn(
            "track_genre",
            trim(col("track_genre"))
        )

    # =================================================
    # 10. Convert duration to minutes
    # =================================================

    df = df.withColumn(
        "duration_minutes",
        round(
            col("duration_ms") / 60000,
            2
        )
    )

    # =================================================
    # 11. Popularity category
    # =================================================

    df = df.withColumn(
        "popularity_category",

        when(
            col("popularity") >= 70,
            "High"
        )

        .when(
            col("popularity") >= 40,
            "Medium"
        )

        .otherwise("Low")
    )

    # =================================================
    # 12. Duration category
    # =================================================

    df = df.withColumn(
        "duration_category",

        when(
            col("duration_minutes") < 3,
            "Short"
        )

        .when(
            col("duration_minutes") <= 5,
            "Medium"
        )

        .otherwise("Long")
    )

    print("\n========== CLEANED DATASET ==========")

    print("Rows:", df.count())
    print("Columns:", len(df.columns))

    print("\n========== CLEANED SCHEMA ==========")

    df.printSchema()

    return df