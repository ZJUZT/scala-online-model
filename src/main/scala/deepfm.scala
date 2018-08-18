import Array._
/**
  *
  * init trained deep_fm model
  * @param anchor the anchor point
  * @param fm_embedding fm second embedding
  * @param input_layer deep part input layer weights
  * @param hidden_layer deep part hidden layer weights
  */
class deepfm (anchor:Array[Double],
              fm_embedding:Array[Array[Array[Double]]],
              input_layer:Array[Array[Double]],
              hidden_layer:Array[Array[Double]]) {
  var this.anchor: Array[Double] = anchor
  var this.fm_embedding:Array[Array[Array[Double]]] = fm_embedding
  var this.input_layer:Array[Array[Double]] = input_layer
  var this.hidden_layer:Array[Array[Double]] = hidden_layer


  /**
    * predict model
    * @param xi non-zero index for each field
    * @param xv corresponding value
    * @return predict score
    */
  def predict(xi:Array[Array[Double]], xv:Array[Array[Double]]): Double = {

    /*
    calculate fm embedding
     */

    val fields_num = 19
    val embedding_size = 4
    var fm_second_order_embedding = ofDim[Double](fields_num, embedding_size)

    for (i <- 0 to this.fm_embedding.length){
      val xi_field = xi(i)
      val xv_field = xv(i)

      for (j <- xi_field.indices){
        for (k <- 0 until embedding_size){
          fm_second_order_embedding(i)(j) += this.fm_embedding(i)(xi_field(j))(k) * xv_field(j)
        }
      }
    }

    /*
    fm part output
    use 2xy = (x+y)^2 - x^2 - y^2 as short path
     */

    var fm_square_sum = 0
    var fm_sum_square = 0


    for (i <- 0 to embedding_size){
      var tmp = 0
      for (j <- 0 to this.fm_embedding.length){
        fm_square_sum += (fm_second_order_embedding(j)(i) ** 2)
        tmp += fm_second_order_embedding(j)(i)
      }
      fm_sum_square += tmp ** 2
    }

    var fm_output = (fm_sum_square - fm_square_sum) * 0.5

    /*
    concat field fm embedding as deep part input
     */
    var deep_input = Array[Double](fields_num * embedding_size)
    for (i <-this.fm_embedding.indices){
      for (j <-0 until embedding_size){
        deep_input(i * embedding_size + j) = fm_second_order_embedding(i)(j)
      }
    }
  }
}
