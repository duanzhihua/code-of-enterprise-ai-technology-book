package com.dt.spark.apps;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MockDatasetSuite {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("MockDatasetSuite")
               // .master("local")
                .getOrCreate();
        final JavaSparkContext jsc = JavaSparkContext.fromSparkContext(spark.sparkContext());
        Random r = new Random();
        List<Tuple2<Double, Double>> x_list = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            if (i >= 100) {
                x_list.add(new Tuple2<>(r.nextDouble() - 0.5, r.nextDouble() - 0.5));
            } else {
                x_list.add(new Tuple2<>(r.nextDouble() + 0.5, r.nextDouble() + 0.5));
            }
        }
        final JavaRDD<Tuple2<Double, Double>> x_dataRDD = jsc.parallelize(x_list);
        x_dataRDD.map(new Function<Tuple2<Double, Double>, String>() {
            @Override
            public String call(Tuple2<Double, Double> v1) {
                return v1._1 + "," + v1._2;
            }
       // }).repartition(1).saveAsTextFile("data/deeplearn_data/x_data");
    }).repartition(1).saveAsTextFile("alluxio://master:19998/data/deeplearn_data/x_data");
        List<Integer> y_list = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            if (i >= 100) {
                y_list.add(1);
            } else {
                y_list.add(0);
            }
        }
        final JavaRDD<Integer> y_RDD = jsc.parallelize(y_list);
      //  y_RDD.repartition(1).saveAsTextFile("data/deeplearn_data/y_data");
        y_RDD.repartition(1).saveAsTextFile("alluxio://master:19998/data/deeplearn_data/y_data");
    }
}
