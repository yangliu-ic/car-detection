import tensorflow as tf


def image_augmentation(image, mask):
    """
    Image Augmentation:
    (1) Random flip (left <--> right)
    (2) Random flip (up <--> down)
    (3) Random brightness
    (4) Random hue

    Args:
        image (3-D Tensor): Image tensor of (H, W, C)
        mask (3-D Tensor): Mask image tensor of (H, W, 1)

    Returns:
        image: augmented image (same shape as input `image`)
        mask: augmented mask (same shape as input `mask`)
    """
    concat_image = tf.concat([image, mask], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_image)
    maybe_flipped = tf.image.random_flip_up_down(concat_image)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]

    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.random_hue(image, 0.3)
    return image, mask


def get_image_mask(queue, augmentation=True):
    """
    Returns image and mask

    Input pipeline:
        Queue -> CSV -> FileRead -> Decode JPEG

    (1) Queue contains a CSV filename
    (2) Text Reader opens the CSV
        CSV file contains two columns
        ["path/to/image.jpg", "path/to/mask.jpg"]
    (3) File Reader opens both files
    (4) Decode JPEG to tensors

    Notes:
        height, width = 512, 512

    Returns
        image (3-D Tensor): (512, 512, 3)
        mask (3-D Tensor): (512, 512, 1)
    """
    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(queue)

    image_path, mask_path = tf.decode_csv(csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    mask_file = tf.read_file(mask_path)

    image = tf.image.decode_jpeg(image_file, channels=3)
    image.set_shape([512, 512, 3])
    image = tf.cast(image, tf.float32)

    mask = tf.image.decode_jpeg(mask_file, channels=1)
    mask.set_shape([512, 512, 1])
    mask = tf.cast(mask, tf.float32)
    mask = mask / (tf.reduce_max(mask) + 1e-7)

    if augmentation:
        image, mask = image_augmentation(image, mask)

    return image, mask


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)
