from pyspark.sql.functions import (
    col,
    avg,
    count,
    round
)


def run_analytics(df):

    print("\n")
    print("=" * 70)
    print("SPOTIFY BIG DATA ANALYTICS")
    print("=" * 70)

    # =================================================
    # ANALYSIS 1
    # Popularity Distribution
    # =================================================

    print("\n1. POPULARITY CATEGORY DISTRIBUTION")

    popularity_distribution = (
        df.groupBy("popularity_category")
        .count()
        .orderBy(col("count").desc())
    )

    popularity_distribution.show()

    # =================================================
    # ANALYSIS 2
    # Most Common Genres
    # =================================================

    print("\n2. TOP 10 GENRES BY NUMBER OF TRACKS")

    top_genres = (
        df.groupBy("track_genre")
        .count()
        .orderBy(col("count").desc())
        .limit(10)
    )

    top_genres.show(truncate=False)

    # =================================================
    # ANALYSIS 3
    # Average Popularity by Genre
    # =================================================

    print("\n3. TOP 10 GENRES BY AVERAGE POPULARITY")

    genre_popularity = (
        df.groupBy("track_genre")
        .agg(
            round(
                avg("popularity"),
                2
            ).alias("average_popularity"),

            count("*").alias("track_count")
        )
        .filter(col("track_count") >= 20)
        .orderBy(
            col("average_popularity").desc()
        )
        .limit(10)
    )

    genre_popularity.show(truncate=False)

    # =================================================
    # ANALYSIS 4
    # Energy vs Popularity
    # =================================================

    print("\n4. ENERGY VS POPULARITY")

    energy_analysis = (
        df.groupBy("popularity_category")
        .agg(
            round(
                avg("energy"),
                3
            ).alias("average_energy")
        )
    )

    energy_analysis.show()

    # =================================================
    # ANALYSIS 5
    # Danceability vs Popularity
    # =================================================

    print("\n5. DANCEABILITY VS POPULARITY")

    dance_analysis = (
        df.groupBy("popularity_category")
        .agg(
            round(
                avg("danceability"),
                3
            ).alias("average_danceability")
        )
    )

    dance_analysis.show()

    # =================================================
    # ANALYSIS 6
    # Tempo vs Popularity
    # =================================================

    print("\n6. TEMPO VS POPULARITY")

    tempo_analysis = (
        df.groupBy("popularity_category")
        .agg(
            round(
                avg("tempo"),
                2
            ).alias("average_tempo")
        )
    )

    tempo_analysis.show()

    # =================================================
    # ANALYSIS 7
    # Valence vs Popularity
    # =================================================

    print("\n7. VALENCE VS POPULARITY")

    valence_analysis = (
        df.groupBy("popularity_category")
        .agg(
            round(
                avg("valence"),
                3
            ).alias("average_valence")
        )
    )

    valence_analysis.show()

    # =================================================
    # ANALYSIS 8
    # Acousticness vs Popularity
    # =================================================

    print("\n8. ACOUSTICNESS VS POPULARITY")

    acoustic_analysis = (
        df.groupBy("popularity_category")
        .agg(
            round(
                avg("acousticness"),
                3
            ).alias("average_acousticness")
        )
    )

    acoustic_analysis.show()

    # =================================================
    # ANALYSIS 9
    # Explicit vs Non Explicit
    # =================================================

    if "explicit" in df.columns:

        print("\n9. EXPLICIT VS NON-EXPLICIT TRACKS")

        explicit_analysis = (
            df.groupBy("explicit")
            .agg(
                round(
                    avg("popularity"),
                    2
                ).alias("average_popularity"),

                count("*").alias("number_of_tracks")
            )
        )

        explicit_analysis.show()

    # =================================================
    # ANALYSIS 10
    # Duration vs Popularity
    # =================================================

    print("\n10. DURATION VS POPULARITY")

    duration_analysis = (
        df.groupBy("duration_category")
        .agg(
            round(
                avg("popularity"),
                2
            ).alias("average_popularity")
        )
    )

    duration_analysis.show()

    return {
        "popularity_distribution":
            popularity_distribution,

        "top_genres":
            top_genres,

        "genre_popularity":
            genre_popularity,

        "energy_analysis":
            energy_analysis,

        "dance_analysis":
            dance_analysis,

        "tempo_analysis":
            tempo_analysis,

        "valence_analysis":
            valence_analysis,

        "acoustic_analysis":
            acoustic_analysis,

        "duration_analysis":
            duration_analysis
    }
def calculate_correlations(df):

    print("\n")
    print("=" * 70)
    print("FEATURE CORRELATION WITH POPULARITY")
    print("=" * 70)

    features = [
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

    correlation_results = []

    print("\nFeature Correlations:")
    print("-" * 50)

    for feature in features:

        if feature not in df.columns:
            print(f"Skipping {feature}: column not found")
            continue

        try:

            correlation = df.stat.corr(
                feature,
                "popularity"
            )

            correlation_results.append(
                (
                    feature,
                    correlation
                )
            )

            print(
                f"{feature:20} {correlation:.4f}"
            )

        except Exception as error:

            print(
                f"Could not calculate {feature}: {error}"
            )

    correlation_results.sort(
        key=lambda x: abs(x[1])
        if x[1] is not None
        else 0,
        reverse=True
    )

    print("\n")
    print(
        "CORRELATIONS SORTED BY STRENGTH"
    )

    print("-" * 50)

    for feature, value in correlation_results:

        print(
            f"{feature:20} {value:.4f}"
        )

    return correlation_results