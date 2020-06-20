from tensorflow.keras.layers import Lambda
import tensorflow as tf


def ctc_loss(y_true, y_pred, input_length, label_length):
    input_length = tf.expand_dims(input_length, axis=-1)
    label_length = tf.expand_dims(label_length, axis=-1)
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return tf.reduce_mean(loss)


def ctc_label_error_rate(y_true, y_pred, input_length, label_length):
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=y_pred, sequence_length=input_length)
    sparse_y = tf.keras.backend.ctc_label_dense_to_sparse(y_true, label_length)
    sparse_y = tf.cast(sparse_y, tf.int64)
    ed_tensor = tf.edit_distance(decoded[0], sparse_y, normalize=True)
    return tf.reduce_mean(ed_tensor)
