import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Dat, TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Gets or Creates spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
        Loads song_data from S3 and processes it by extracting the songs and artist tables and then again loaded back to S3
        parameters used
            spark       : Spark Session
            input_data  : location of song_data json files with the songs metadata
            output_data : S3 bucket where output tables in parquet format will be stored
    """
    # get filepath to song data file
    song_data = input_data+'song_data/*/*/*/*.json'
    
    # read song data file
    song_data_schema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("artist_location", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_name", Str()),
        Fld("num_songs",Int()),
        Fld("title", Str()),
        Fld("duration", Dbl()),
        Fld("year", Int())
    ])


    df = spark.read.json(song_data, schema = song_data_schema)

    # extract columns to create songs table
    song_cols = ["title","artist_id","duration","year" ]
    songs_table = df.select(song_cols).dropDuplicates().withColumn("song_id", monotonically_increasing_id())
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year","artist_id").parquet(output_data + "songs/", mode="overwrite")

    # extract columns to create artists table
    artists_cols = ["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]
    artists_table = df.select(artists_cols).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists/", mode="overwrite")

    df.createOrReplaceTempView("song_df_table")



def process_log_data(spark, input_data, output_data):
    """
        Loads log_data from S3 and processes it by extracting the songs and artist tables as well as data for time table
        and then again loaded back to S3.
        parameters used
            spark       : Spark Session
            input_data  : location of log_data(events) json files
            output_data : S3 bucket where output tables in parquet format will be stored       
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file

    log_data_schema = R[(
        Fld("artist", Str()),
        Fld("auth", Str()),
        Fld("firstName", Str()),
        Fld("gender", Str()),
        Fld("itemInSession", Int()),
        Fld("lastName", Str()),
        Fld("length", Dbl()),
        Fld("level", Str()),
        Fld("location", Str()),
        Fld("method", Str()),
        Fld("page", Str()),
        Fld("registration", Dbl()),
        Fld("sessionId", Int()),
        Fld("song", Str()),
        Fld("status",Int()),
        Fld("ts", Dbl()),
        Fld("userAgent", Str()),
        Fld("userId", Str())
    )]

    df = spark.read.json(log_data, schema = log_data_schema)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_cols = ["userdId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"]
    users_table = df.selectExpr(users_cols).dropDuplicates()
    # write users table to parquet files
    users_table.write.parquet(output_data + "users/", mode="overwrite")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    df = df.withColumn("timestamp", get_timestamp("ts"))

    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    df = df.withColumn("start_time", get_datetime("timestamp"))
    
    # extract columns to create time table
    time_table = df.withColumn("hour", hour("start_time")) \
                    .withColumn("day", dayofmonth("start_time")) \
                    .withColumn("week", weekofyear("start_time")) \
                    .withColumn("month", month("start_time")) \
                    .withColumn("year", year("start_time")) \
                    .withColumn("weekday", dayofweek("start_time")) \
                    .select("start_time","hour", "day", "week", "month", "year", "weekday").drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(output_data + "time_table/", mode="overwrite")

    # read in song data to use for songplays table
    song_df = spark.sql("SELECT DISTINCT song_id, artist_id, artist_name FROM song_df_table")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, song_df.artist_name == df.artist, "inner") \
        .distinct() \
        .select(col("start_time"), col("userId").alias("user_id"), col("level"), col("sessionId").alias("session_id"), col("location"), col("userAgent").alias("user_agent"), col("song_id"), col("artist_id")) \
        .withColumn("songplay_id", monotonically_increasing_id())
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(output_data + 'songplays/', mode="overwrite")


def main():
    """
        Extracts songs and events data from S3, transform it into dimensional tables format and load it back to S3 (in parquet format) 
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify-dend/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
