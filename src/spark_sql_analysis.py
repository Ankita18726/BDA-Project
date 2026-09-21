from pyspark.sql import functions as F

import os

def run_spark_sql_analysis(spark, df):

    print("\n" + "=" * 70)
    print("SPARK SQL ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Create Spark SQL views
    # ---------------------------------------------------------

    # Keep the complete dataset for genre-level analysis
    df.createOrReplaceTempView("spotify_tracks")

    # Remove repeated occurrences of the same Spotify track
    # for track-level analysis
    unique_tracks_df = df.dropDuplicates(["track_id"])
    total_rows = df.count()

    unique_track_count = unique_tracks_df.count()

    duplicate_rows = (
        total_rows - unique_track_count
)

    print("\nTRACK DUPLICATE ANALYSIS")
    print("-" * 50)

    print(
    f"Total dataset rows      : {total_rows}"
)

    print(
    f"Unique Spotify tracks   : {unique_track_count}"
)

    print(
    f"Genre duplicate rows    : {duplicate_rows}"
)

    unique_tracks_df.createOrReplaceTempView(
        "unique_spotify_tracks"
    )

    print(
        "\nTotal records:",
        df.count()
    )

    print(
        "Unique tracks:",
        unique_tracks_df.count()
    )

    # ---------------------------------------------------------
    # QUERY 1 - Top Genres
    # ---------------------------------------------------------

    print("\nQUERY 1: TOP GENRES")

    top_genres = spark.sql("""
        SELECT
            track_genre,
            COUNT(*) AS total_tracks
        FROM spotify_tracks
        GROUP BY track_genre
        ORDER BY total_tracks DESC
        LIMIT 10
    """)

    top_genres.show(
        n=10,
        truncate=False
    )

    # ---------------------------------------------------------
    # QUERY 2 - Top Genres by Average Popularity
    # ---------------------------------------------------------

    print(
        "\nQUERY 2: "
        "TOP GENRES BY AVERAGE POPULARITY"
    )

    popular_genres = spark.sql("""
        SELECT
            track_genre,
            ROUND(
                AVG(popularity),
                2
            ) AS average_popularity,
            COUNT(*) AS track_count
        FROM spotify_tracks
        GROUP BY track_genre
        ORDER BY average_popularity DESC
        LIMIT 10
    """)

    popular_genres.show(
        n=10,
        truncate=False
    )

    # ---------------------------------------------------------
    # QUERY 3 - Audio Features by Popularity
    # ---------------------------------------------------------

    print(
        "\nQUERY 3: "
        "AUDIO FEATURES BY POPULARITY CATEGORY"
    )

    audio_features = spark.sql("""
        SELECT
            popularity_category,

            ROUND(
                AVG(danceability),
                3
            ) AS avg_danceability,

            ROUND(
                AVG(energy),
                3
            ) AS avg_energy,

            ROUND(
                AVG(valence),
                3
            ) AS avg_valence,

            ROUND(
                AVG(acousticness),
                3
            ) AS avg_acousticness,

            ROUND(
                AVG(tempo),
                2
            ) AS avg_tempo

        FROM spotify_tracks

        GROUP BY popularity_category

        ORDER BY
            CASE popularity_category
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                WHEN 'Low' THEN 3
            END
    """)

    audio_features.show(
        truncate=False
    )

    # ---------------------------------------------------------
    # QUERY 4 - Most Popular Unique Tracks
    # ---------------------------------------------------------

    print(
        "\nQUERY 4: "
        "MOST POPULAR UNIQUE TRACKS"
    )

    popular_tracks = spark.sql("""
        SELECT
            track_name,
            artists,
            popularity
        FROM unique_spotify_tracks
        ORDER BY popularity DESC
        LIMIT 10
    """)

    popular_tracks.show(
        n=10,
        truncate=False
    )
    os.makedirs(
    "outputs/results",
    exist_ok=True
)

    popular_tracks.toPandas().to_csv(
        "outputs/results/top_popular_tracks.csv",
        index=False
)

    # ---------------------------------------------------------
    # QUERY 5 - Popularity by Duration
    # ---------------------------------------------------------

    print(
        "\nQUERY 5: "
        "POPULARITY BY TRACK DURATION"
    )

    duration_analysis = spark.sql("""
        SELECT
            duration_category,
            COUNT(*) AS total_tracks,
            ROUND(
                AVG(popularity),
                2
            ) AS average_popularity
        FROM unique_spotify_tracks
        GROUP BY duration_category
        ORDER BY average_popularity DESC
    """)

    duration_analysis.show(
        truncate=False
    )

    return {
        "top_genres": top_genres,
        "popular_genres": popular_genres,
        "audio_features": audio_features,
        "popular_tracks": popular_tracks,
        "duration_analysis": duration_analysis,
        "unique_tracks": unique_tracks_df
    }