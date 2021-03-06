/**
  * This object is for locally linear deep factorization machines inference
 */


import org.apache.spark.rdd.RDD

import scala.io._
import java.io._

import DeepFM.DeepFm

import scala.Array._
import scala.collection.mutable.ArrayBuffer

/**
  * Feature format
  * xi: non-zero index
  * xv: corresponding value
  * -----------------------
  * Model format
  * anchor points indicates the number of deep_fm model
  * each deep_fm contains
  * 1. anchor point
  * 2. fm second embedding
  * 3. dnn input and hidden layer weights
  */

object predict {

  // field dict
  var field_info = Array(50, 60, 70, 80, 90, 100, 120)
  val num_feature = 119
  // num of anchor points
  val num_anchor = 100

  // num of nearest neighbours
  val num_nn = 8

  // deep layer
  val input_width = 32

  val hidden_width = 32

  val embedding_size = 4

  var smooth_ratio = 1e-1

  def main(args: Array[String]): Unit = {

    // fm embedding length
    /*
    load fm embedding and nn for each anchor point
    load from table online
    mock by load from file offline
     */

    val table_name = new Array[String](num_anchor)
    for (i <- 0 until num_anchor){
      table_name(i) = "data/deep_fm_" + i
    }

    var LL_Deep_FM = new Array[DeepFm](num_anchor)
    var writer = new PrintWriter(new File("data/ll_deep_fm.res"))

    for (i <- table_name.indices){
      var lines = Source.fromFile(table_name(i)).getLines()
      // anchor point
      var anchor_point = lines.next()
      var tokens = anchor_point.split(" ")
      var ap = new Array[Double](tokens.length)
      for (i <- ap.indices){
        ap(i) = tokens(i).toDouble
      }

      // fm bias
      var fm_bias = lines.next().toDouble

      // fm embedding
      var fm_embedding = new Array[Any](field_info.length)
      for (i <- field_info.indices){
        var rows = 0
        if (i==0){
          rows = field_info(i) + 1
        }
        else{
          rows = field_info(i) - field_info(i-1)
        }

        var value = lines.next().split(" ")
        var field_embedding = ofDim[Double](rows, embedding_size)

        for (i <- value.indices) {
          field_embedding(i / embedding_size)(i % embedding_size) = value(i).toDouble
        }

        fm_embedding(i) = field_embedding

      }

      // input layer weight
      val fm_layer_width = field_info.length*embedding_size
      val input_layer_weight = ofDim[Double](input_width, fm_layer_width)

      var line = lines.next()

      tokens = line.split(" ")

      for(i <- tokens.indices){
        input_layer_weight(i / fm_layer_width)(i % fm_layer_width) = tokens(i).toDouble
      }

      // input layer bias
      var input_layer_bias = new Array[Double](input_width)
      line = lines.next()
      tokens = line.split(" ")

      for(i <- tokens.indices){
        input_layer_bias(i) = tokens(i).toDouble
      }

      // hidden layer weight
      var hidden_layer_weight = ofDim[Double](hidden_width, input_width)
      line = lines.next()
      tokens = line.split(" ")

      for(i <- tokens.indices){
        hidden_layer_weight(i / input_width)(i % input_width) = tokens(i).toDouble
      }

      // hidden layer bias
      var hidden_layer_bias = new Array[Double](hidden_width)
      line = lines.next()
      tokens = line.split(" ")

      for(i <- tokens.indices){
        hidden_layer_bias(i) = tokens(i).toDouble
      }

      val deep_fm = new DeepFm(anchor = ap,
        fm_embedding=fm_embedding,
        fm_bias = fm_bias,
        input_layer_weight = input_layer_weight,
        input_layer_bias = input_layer_bias,
        hidden_layer_weight = hidden_layer_weight,
        hidden_layer_bias = hidden_layer_bias
      )

      LL_Deep_FM(i) = deep_fm

    }

    // make predictions
    for(line <- Source.fromFile("data/0820_116_test.libsvm").getLines()){
      var index = new ArrayBuffer[Int]()
      var value = new ArrayBuffer[Double]()

      val tokens = line.split(" |\\t|\\n")

      for(i <- 1 until tokens.length){
        val pair = tokens(i).split(":")
        if (pair.length == 2){
          if((pair(0) != "")&&(pair(1) != "")){
            if(pair(0).toInt < num_feature){
              index += (pair(0).toInt - 1)
              value += pair(1).toDouble
            }
          }
        }
      }

      val ll_weights = new Array[Double](num_anchor)

      // calculate all the weights and sort
      for(i <- LL_Deep_FM.indices){
        ll_weights(i) = LL_Deep_FM(i).get_weight(index.toArray, value.toArray)
      }

      val (ll_weights_sorted, indices) = ll_weights.zipWithIndex.sorted.unzip

      // calculate the nearest neighbour deep fm result
      val ll_scores = new Array[Double](num_nn)
      for(i <- 0 until num_nn){
        val model_idx = indices(num_anchor - 1 - i)
        ll_scores(i) = LL_Deep_FM(model_idx).predict(index.toArray, value.toArray)
      }

      var sum_weight = 0.0
      var final_score = 0.0

      for(i <- 0 until num_nn){
        final_score = final_score + ll_scores(i) * ll_weights_sorted(num_anchor - 1 - i)
        sum_weight = sum_weight + ll_weights_sorted(num_anchor - 1 - i)
      }

      final_score = final_score/(sum_weight + smooth_ratio)

      // do sigmoid
      final_score = 1 / (1 + math.exp(-final_score))
      writer.write(final_score.toString)
      writer.write("\n")
    }

    writer.close()

  }
}
