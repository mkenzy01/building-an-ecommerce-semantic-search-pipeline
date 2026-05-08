# import libraries
import os
import sys

# Hadoop config for Windows
hadoop_home = "C:/hadoop"

os.environ["HADOOP_HOME"] = hadoop_home
os.environ["hadoop.home.dir"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + os.environ["PATH"]

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit

# spark config
spark = (
    SparkSession.builder
    .appName("VectorEngine")
    .config("spark.sql.shuffle.partitions", "20")
    .config("spark.sql.caseSensitive", "true")
    .config("spark.driver.memory", "4g")
    .master("local[*]")
    .getOrCreate()
)

# set the log level
spark.sparkContext.setLogLevel("ERROR")

# ingest the metadata files
print("Reading the metadata file...")
meta_df = spark.read.json("./data/bronze/metadata/*.jsonl.gz")

# select the needed columns
meta_clean = (
    meta_df.select(
        col("parent_asin").alias("metadata_parent_asin"),
        col("title"),
        col("main_category"),
        coalesce(col("price"), lit(0.0)).alias("price")
    )
    .drop_duplicates(["metadata_parent_asin"])
)

print(f"Processed {meta_clean.count()} metadata data points")

# ingest reviews data
print("Reading the reviews file...")
reviews_df = spark.read.json("./data/bronze/review/*.jsonl.gz")

# select the needed columns
reviews_clean = (
    reviews_df.select(
        col("parent_asin"),
        col("user_id"),
        col("rating"),
        col("text"),
        col("timestamp")
    )
    .filter("text IS NOT NULL")
)

print(f"Processed {reviews_clean.count()} review data points")

# join
silver_join = (
    reviews_clean
    .join(
        meta_clean,
        reviews_clean.parent_asin == meta_clean.metadata_parent_asin,
        "inner"
    )
    .drop("metadata_parent_asin")
)

# write into the silver layer
SILVER_PATH = "./data/silver"

os.makedirs(SILVER_PATH, exist_ok=True)

silver_join.coalesce(1).write.mode("overwrite").parquet(
    f"{SILVER_PATH}/complete_reviews"
)

print(f"Silver layer is now completed, processed {silver_join.count()} rows")