/**
  *
  * init trained deep_fm model
  * @param anchor the anchor point
  * @param fm_embedding fm second embedding
  * @param input_layer deep part input layer weights
  * @param hidden_layer deep part hidden layer weights
  */
class deepfm (anchor:Array[Double],
              fm_embedding:Array[Array[Double]],
              input_layer:Array[Array[Double]],
              hidden_layer:Array[Array[Double]]) {
  var this.anchor: Array[Double] = anchor
  var this.fm_embedding:Array[Array[Double]] = fm_embedding
  var this.input_layer:Array[Array[Double]] = input_layer
  var this.hidden_layer:Array[Array[Double]] = hidden_layer


  /**
    * predict model
    * @param xi non-zero index for each field
    * @param xv corresponding value
    * @return
    */
  def predicate(xi:Array[Array[Double]], xv:Array[Array[Double]]): Double = {

  }
}
