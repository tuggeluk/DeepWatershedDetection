import tensorflow as tf

def add_image_summary(image, boxes):
    # add back mean
    # image += cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    channels = tf.unstack(image, axis=-1)
    image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    # dims for normalization
    width = tf.to_float(tf.shape(image)[2])
    height = tf.to_float(tf.shape(image)[1])
    # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
    cols = tf.unstack(boxes, axis=-1)
    boxes = tf.stack([cols[1] / height,
                      cols[0] / width,
                      cols[3] / height,
                      cols[2] / width], axis=1)
    # add batch dimension (assume batch_size==1)
    # assert image.get_shape()[0] == 1
    boxes = tf.expand_dims(boxes, dim=0)
    image = tf.image.draw_bounding_boxes(image, boxes)

    return tf.summary.image('ground_truth', image)


def add_act_summary(tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))


def add_score_summary(key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)


def add_train_summary(var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)