import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics


object Main {

  def castDouble(df: DataFrame, s: String) : DataFrame = {
    df.withColumn(s, df(s).cast("double"))
  }

  def replace(df: DataFrame,column: String, oldString: String, newString: String) : DataFrame = {
    df.withColumn(column, regexp_replace(col(column) , lit(oldString), lit(newString)))
  }

  def newJoin(leftDF: DataFrame, rightDF: DataFrame, colName: String): DataFrame = {
    var ldf = leftDF.withColumnRenamed(colName, "old")
    var df = ldf.join(rightDF, Seq("AppId"), "outer")
      .withColumn(colName, coalesce(col(colName), col("old")))
      .drop("old")
    df.orderBy(asc("AppId"))
  }

  def minMaxScaler(df: DataFrame, colName: String): DataFrame = {
    var colMax = df.agg(max(col(colName))).collect()(0)(0)
    var colMin = df.agg(min(col(colName))).collect()(0)(0)
    var colDiff = lit(colMax) - lit(colMin)
    df.withColumn(colName, (df(colName) - colMin) / colDiff)
  }

  def main(args: Array[String]) {

    // Hide end warnings/infos at the end of build. (En local)
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    //Initialize Session
    val spark = SparkSession.builder().appName("SparkSQL For CSV").master("local[*]").getOrCreate()

    //Import csv file
    var df = spark.read.option("header","true").option("delimiter",",").csv("hdfs:///user/maria_dev/gpsa/googleplaystore.csv")

    println("Data CLeaning...")
    // Remove duplicates, unnecessary rows & cols
    df = df.dropDuplicates(col1 = "App")
    df = df.drop("Genres").drop("Last Updated").drop("Current Ver").drop("Android Ver").drop("Category").drop("App")
    df = df.filter(!(col("Category") === "1.9" || col("Category") === "Face" || col("Category") === " traffic jams" || col("Type") === "NaN"))

    // Add a feature to know if app has been rated
    df = df.withColumn("isRated", when(col("Rating").isNaN, 0).otherwise(1))

    // Transforming NaN values into average
    val avgRating = (df.filter(!isnan(col("Rating"))).select(mean("Rating")).collect()(0)(0).toString.toFloat * 10).round / 10.toDouble
    df = replace(df, "Rating", "NaN", avgRating.toString)

    // Add a unique numerical value to each app
    df = df.withColumn("AppId",monotonically_increasing_id)

    // Rename some cols
    df = df.withColumnRenamed("Size", "Size_In_Octet").withColumnRenamed("Type", "label").withColumnRenamed("Content Rating", "Content_Rating")

    println("Preparing Data for Logistic Regression...")
    // Convert size to numerical
    for (tuple <- Array(("M", "000000"), ("k", "000"))) {
      df = replace(df, "Size_In_Octet", tuple._1, tuple._2)
    }
    df.createOrReplaceTempView("tab")
    var sqlDF = spark.sql("SELECT AppId, Size_In_Octet FROM tab where Size_in_Octet like '%.%'")
    sqlDF = sqlDF.withColumn("Size_In_Octet", expr("substring(Size_In_Octet, 1, length(Size_In_Octet)-1)"))
    sqlDF = sqlDF.withColumn("Size_In_Octet", functions.regexp_replace(sqlDF.col("Size_In_Octet"), "[^A-Z0-9_]", ""))
    df = newJoin(df, sqlDF, "Size_In_Octet")
    val avgSize = (df.filter(!(col("Size_In_Octet") === "Varies with device")).select(mean("Size_In_Octet")).collect()(0)(0).toString.toFloat * 10).round
    df = replace(df, "Size_In_Octet", "Varies with device", avgSize.toString)

    // Convert Installs to numerical
    df = replace(df, "Installs", "[^A-Z0-9_]", "")

    // Convert Price to numerical
    df.createOrReplaceTempView("tab")
    var osqldf = spark.sql("SELECT AppId, Price FROM tab where Price <> '0'")
    osqldf = osqldf.withColumn("Price", expr("substring(Price, 2, length(Price))"))
    df = newJoin(df, osqldf, "Price")

    val arrayContent = Array(("Adults only 18+", "Mature 17"), ("Unrated", "Everyone"), ("Teen", "1"), ("Mature 17", "2"), ("Everyone 10", "3"), ("Everyone", "4"))
    for (tuple <- arrayContent) {
      df = replace(df, "Content_Rating", tuple._1, tuple._2)
    }
    df = df.withColumn("Content_Rating", functions.regexp_replace(df.col("Content_Rating"), "[^A-Z0-9_]", ""))

    for (tuple <- Array(("Free", "1"), ("Paid", "0"))) {
      df = replace(df, "label", tuple._1, tuple._2)
    }

    // Cast the cols to the right type -> double
    for (s <- Array("Rating", "Reviews", "isRated", "label", "Price", "Installs", "Content_Rating", "Size_In_Octet")) {
      df = castDouble(df, s)
    }

    // Scaling some cols
    for (col <- Array("Reviews", "Size_In_Octet", "Installs", "Price")) {
      println("Scaling: "+ col)
      df = minMaxScaler(df, col)
    }

    // Select features for claddification
    val assembler = new VectorAssembler()
      .setInputCols(Array("Rating", "Reviews", "Size_In_Octet", "Installs", "Content_Rating", "isRated"))
      .setOutputCol("features")

    // Split the data  70% training 30% test
    val Array(training,test) = df.randomSplit(Array(0.7,0.3), seed=147)

    // ML Process
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(assembler,lr))
    val model = pipeline.fit(training)
    val results = model.transform(test)

    val predictions = results.select("prediction").rdd.map(_.getDouble(0))
    val labels = results.select("label").rdd.map(_.getDouble(0))
    val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    results.show(10)
    println("RMSE: "+ RMSE)

    df.write.format("csv").option("header", "true").save("hdfs:///user/maria_dev/gpsa/toml.csv")

    spark.close()
  }
}