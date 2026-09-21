import os
import sys

from pyspark.sql import SparkSession


def create_spark_session():

    # Tell PySpark to use the same Python
    # interpreter currently running this project.
    python_executable = sys.executable

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    print("Python used by PySpark:")
    print(python_executable)

    spark = (
        SparkSession.builder
        .appName("SpotifyBigDataAnalytics")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark