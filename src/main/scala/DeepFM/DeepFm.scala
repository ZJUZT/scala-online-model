
package DeepFM

import scala.Array._
import scala.collection.mutable.ArrayBuffer
/**
  *
  * init trained deep_fm model
  * @param anchor the anchor point
  * @param fm_embedding fm second embedding
  * @param input_layer_weight deep part input layer weights
  * @param hidden_layer_weight deep part hidden layer weights
  */
class DeepFm (anchor:Array[Double],
              fm_embedding:Array[Any],
              fm_bias: Double,
              input_layer_weight:Array[Array[Double]],
              input_layer_bias:Array[Double],
              hidden_layer_weight:Array[Array[Double]],
              hidden_layer_bias:Array[Double]) {

  // field dict
  val field_info = Array(29, 33, 34, 35, 51, 201, 351, 501, 651, 801, 951, 1101, 1251, 1301, 1601, 1636, 1731, 1801, 1950)
  // feature number
  val num_feature = 1935

  val c = 0.01
  /**
    * convert origin sample into field index representation
    * @param index index in libsvm file
    * @param value corresponding value
    * @return weight and corresponding value
    */
  def predict(index: Array[Int], value: Array[Double]): Array[Double] = {
    var feature_dense = new Array[Double](num_feature)
    for(i <- index.indices){
      feature_dense(index(i)) = value(i)
    }

    // calculate weight
    var square_sum = 0.0
    for(i <- 0 to num_feature){
      square_sum = square_sum + (feature_dense(i) - anchor(i)) * (feature_dense(i) - anchor(i))
    }

    var weight = math.exp(-c * square_sum)

    // construct field input
    var xi = new Array[Array[Int]](field_info.length)
    var xv = new Array[Array[Double]](field_info.length)

    var field_index = 0
    var feature_index = 0

    var field_i = new ArrayBuffer[Int]()
    var field_v = new ArrayBuffer[Double]()

    while (field_index < field_info.length){
      while(feature_index < index.length && index(feature_index) < field_info(field_index)){
        if(field_index == 0){
          field_i += index(feature_index)
        }
        else{
          field_i += (index(feature_index) - field_info(field_index - 1))
        }

        field_v += value(feature_index)
      }

      xi(field_index) = field_i.toArray
      xv(field_index) = field_v.toArray
      field_index = field_index + 1

      field_i = new ArrayBuffer[Int]()
      field_v = new ArrayBuffer[Double]()
    }

    // calculate weight
    var score = predict_(xi, xv)

    var res = Array[Double](score, weight)

    res
  }


  /**
    * predict model
    * @param xi non-zero index for each field
    * @param xv corresponding value
    * @return predict score and corresponding weight
    */

  def predict_(xi:Array[Array[Int]], xv:Array[Array[Double]]): Double = {

    /*
    calculate fm embedding
     */

    val fields_num = 19
    val embedding_size = 4
    var fm_second_order_embedding = ofDim[Double](fields_num, embedding_size)

    for (i <- 0 to fm_embedding.length){
      val xi_field = xi(i)
      val xv_field = xv(i)

      for (j <- xi_field.indices){
        for (k <- 0 until embedding_size){
          var field_embedding = fm_embedding(i).asInstanceOf[Array[Array[Double]]]
          fm_second_order_embedding(i)(j) = fm_second_order_embedding(i)(j) +
            field_embedding(xi_field(j))(k) * xv_field(j)
        }
      }
    }

    /*
    fm part output
    use 2xy = (x+y)^2 - x^2 - y^2 as short path
     */

    var fm_square_sum = 0.0
    var fm_sum_square = 0.0


    for (i <- 0 to embedding_size){
      var tmp = 0.0
      for (j <- 0 to fm_embedding.length){
        fm_square_sum = fm_square_sum + (fm_second_order_embedding(j)(i) * fm_second_order_embedding(j)(i))
        tmp += fm_second_order_embedding(j)(i)
      }
      fm_sum_square = fm_square_sum + tmp * tmp
    }

    var fm_output = (fm_sum_square - fm_square_sum) * 0.5 + fm_bias

    /*
    concat field fm embedding as deep part input
     */
    var deep_input = Array[Double](fields_num * embedding_size)
    for (i <- fm_embedding.indices){
      for (j <-0 until embedding_size){
        deep_input(i * embedding_size + j) = fm_second_order_embedding(i)(j)
      }
    }

    /*
    calculate deep part
     */

    val fm_width = fields_num * embedding_size
    val input_width = input_layer_bias.length
    val hidden_width = hidden_layer_bias.length

    // input to hidden
    var input_layer_output = new Array[Double](input_width)
    for (i <-0 until input_width){
      for (j <- 0 until fm_width){
        input_layer_output(i) = deep_input(j) * input_layer_weight(j)(i)
      }
      input_layer_output(i) += input_layer_bias(i)
      // relu
      if(input_layer_output(i) < 0){
        input_layer_output(i) = 0
      }
    }

    var hidden_layer_output = new Array[Double](hidden_width)

    // hidden to output
    for (i <- 0 until hidden_width){
      for (j <- 0 until input_width){
        hidden_layer_output(i) = input_layer_output(j) * hidden_layer_weight(j)(i)
      }
      hidden_layer_output(i) += hidden_layer_bias(i)

      // relu
      if(hidden_layer_output(i) < 0){
        hidden_layer_output(i) = 0
      }
    }

    // deep output
    var deep_output = 0.0
    for (i <- 0 until hidden_width){
      deep_output = deep_output + hidden_layer_output(i)
    }

    // sum fm and deep output
    // then sigmoid

    val predict_prob = 1 / (1 + math.exp(-(fm_output + deep_output)))

    predict_prob

  }
}
