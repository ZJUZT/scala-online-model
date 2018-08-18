/**
  * This object is for locally linear deep factorization machines inference
 */

import org.apache.spark.rdd.RDD

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

  def main(args: Array[String]): Unit = {
    // num of anchor points
    var num_anchor = 20

    // num of nearest neighbours
    var num_nn = 3

    // deep layer
    var deep_layer = 2

    // feature number
    var num_feature = 1935

    // field dict
    var field_info = Array(29, 33, 34, 35, 51, 201, 351, 501, 651, 801, 951, 1101, 1251, 1301, 1601, 1636, 1731, 1801, 1950)

    // load fm embedding and nn for each anchor point
    var tmp: Array[Array[Int]] = Array(Array(1,2,3))
    tmp = tmp :+ Array(2,3)

    print(tmp(1)(0))
  }
}
