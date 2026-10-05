"""Export real PySpark pipeline results to a JSON snapshot for the React dashboard.
Call export_dashboard_data(cleaned_df) in main.py after the models have finished.
"""
import json
import os
from datetime import datetime, timezone
from pyspark.sql import functions as F


def export_dashboard_data(cleaned_df, output_path='outputs/dashboard/dashboard_data.json'):
    # Genre assignments are meaningful duplicates for genre analysis; unique
    # tracks are necessary for track-level measures and artist rankings.
    unique = cleaned_df.dropDuplicates(['track_id']).cache()
    total_rows = cleaned_df.count()
    unique_count = unique.count()

    category_rows = (unique.groupBy('popularity_category').count()
                     .collect())
    distribution = {r['popularity_category']: int(r['count']) for r in category_rows}

    genre_rows = (cleaned_df.groupBy('track_genre')
                  .agg(F.count('*').alias('tracks'),
                       F.avg('popularity').alias('avg_popularity'))
                  .where(F.col('tracks') >= 25)
                  .orderBy(F.desc('avg_popularity'), F.asc('track_genre'))
                  .limit(12).collect())
    genres = [{'name': r['track_genre'], 'tracks': int(r['tracks']),
               'avgPopularity': round(float(r['avg_popularity']), 2)}
              for r in genre_rows]

    # Tracks may list collaborators separated by ';'. Count each artist once
    # per unique track, rather than considering the entire credits string one artist.
    exploded = (unique.select('track_id', 'popularity',
                              F.explode(F.split(F.col('artists'), ';')).alias('artist'))
                .withColumn('artist', F.trim('artist'))
                .where((F.col('artist').isNotNull()) & (F.col('artist') != '')))
    artist_rows = (exploded.groupBy('artist')
                   .agg(F.countDistinct('track_id').alias('tracks'),
                        F.avg('popularity').alias('avg_popularity'))
                   .where(F.col('tracks') >= 5)
                   .orderBy(F.desc('avg_popularity'), F.desc('tracks'))
                   .limit(10).collect())
    artists = [{'name': r['artist'], 'tracks': int(r['tracks']),
                'avgPopularity': round(float(r['avg_popularity']), 2)}
               for r in artist_rows]

    track_rows = (unique.select('track_id', 'track_name', 'artists', 'popularity')
                  .orderBy(F.desc('popularity'), F.asc('track_name'))
                  .limit(10).collect())
    tracks = [{'id': r['track_id'], 'name': r['track_name'],
               'artists': r['artists'], 'popularity': float(r['popularity'])}
              for r in track_rows]

    bucket = (F.when(F.col('popularity') >= 100, F.lit(90))
              .otherwise(F.floor(F.col('popularity') / 10) * 10)
              .cast('int'))
    hist_rows = (unique.withColumn('bucket', bucket)
                 .groupBy('bucket').count().collect())
    hist_counts = {int(r['bucket']): int(r['count']) for r in hist_rows}
    histogram = [{'range': f'{n}-{99 if n == 90 else n+9}',
                  'count': hist_counts.get(n, 0)} for n in range(0, 100, 10)]

    # Correlations computed in Spark, on unique tracks, not a sampled Pandas frame.
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_minutes']
    correlations = []
    for col in features:
        value = unique.stat.corr(col, 'popularity')
        if value is not None:
            correlations.append({'feature': col, 'value': round(float(value), 4)})
    correlations.sort(key=lambda item: abs(item['value']), reverse=True)

    snapshot = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'dataset': {'cleanedRows': total_rows, 'uniqueTracks': unique_count,
                    'genreDuplicateRows': total_rows - unique_count,
                    'genreCount': cleaned_df.select('track_genre').distinct().count()},
        'popularityDistribution': [{'category': category,
                                    'count': distribution.get(category, 0)}
                                   for category in ['Low', 'Medium', 'High']],
        'genres': genres, 'artists': artists, 'topTracks': tracks,
        'histogram': histogram, 'correlations': correlations,
        'methodology': {'genreAnalysis': 'Cleaned genre-assignment rows',
                        'trackAnalysis': 'Unique Spotify track IDs',
                        'modelResults': 'Loaded separately from outputs/results/model_comparison.csv'}
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2, allow_nan=False)
    unique.unpersist()
    print(f'Dashboard snapshot exported: {output_path}')
    return output_path
