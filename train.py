import time
import os
import logging
import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
from net import make_unet
from util import image_augmentation, get_image_mask, IOU_


logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)

def make_train_op(y_pred, y_true):
    """
    Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
    loss = -IOU_(y_pred, y_true)
    global_step = tf.train.get_or_create_global_step()
    optim = tf.train.AdamOptimizer()
    return optim.minimize(loss, global_step=global_step)


def read_flags():
    """Returns flags"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size",
                        default=4,
                        type=int,
                        help="Batch size (default: 4)")
    parser.add_argument("--logdir",
                        default="logdir",
                        help="Tensorboard log directory (default: logdir)")
    parser.add_argument("--ckdir",
                        default="models",
                        help="Checkpoint directory (default: models)")
    flags = parser.parse_args()
    return flags


def main(flags):
    train = pd.read_csv("./train.csv")
    n_train = train.shape[0]

    valid = pd.read_csv("./valid.csv")
    n_valid = valid.shape[0]

    test = pd.read_csv("./test.csv")
    n_test = test.shape[0]

    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(flags.logdir, "train", current_time)
    test_logdir = os.path.join(flags.logdir, "test", current_time)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")

    pred = make_unet(X, mode)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.histogram("Predicted_Mask", pred)
    tf.summary.image("Predicted_Mask", pred)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y)

    IOU_op = IOU_(pred, y)
    tf.summary.scalar("IOU", IOU_op)

    train_csv = tf.train.string_input_producer(['train.csv'])
    valid_csv = tf.train.string_input_producer(['valid.csv'])
    test_csv = tf.train.string_input_producer(['test.csv'])
    train_image, train_mask = get_image_mask(train_csv)
    valid_image, valid_mask = get_image_mask(valid_csv, augmentation=False)
    test_image, test_mask = get_image_mask(test_csv, augmentation=False)

    X_batch_op, y_batch_op = tf.train.shuffle_batch(
        [train_image, train_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 5,
        min_after_dequeue=flags.batch_size * 2,
        allow_smaller_final_batch=True)
    X_valid_op, y_valid_op = tf.train.batch([valid_image, valid_mask],
                                            batch_size=flags.batch_size,
                                            capacity=flags.batch_size * 2,
                                            allow_smaller_final_batch=True)
    X_test_op, y_test_op = tf.train.batch([test_image, test_mask],
                                          batch_size=flags.batch_size,
                                          capacity=flags.batch_size * 2,
                                          allow_smaller_final_batch=True)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        if os.path.exists(flags.ckdir) and tf.train.checkpoint_exists(flags.ckdir):
            latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
            saver.restore(sess, latest_check_point)

        else:
            os.mkdir(flags.ckdir)
            # try:
            #     os.rmdir(flags.ckdir)
            # except IOError:
            #     pass
            # os.mkdir(flags.ckdir)

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            logging.info("Start optimization")
            logging.info("Num Train: {:}, Num Validation: {:}, Num Test: {:}".format(
                n_train, n_valid, n_test))
            for epoch in range(flags.epochs):

                total_train_iou = 0
                for step in range(0, n_train, flags.batch_size):
                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])
                    _, step_iou, step_summary, global_step_value = sess.run(
                        [train_op, IOU_op, summary_op, global_step],
                        feed_dict={X: X_batch,
                                   y: y_batch,
                                   mode: True})
                    total_train_iou += step_iou * X_batch.shape[0]
                    train_summary_writer.add_summary(step_summary, global_step_value)
                    if step % 5000 == 0:
                        logging.info("Epoch {:}, Iter {:}, IOU = {:.4f}".format(
                            epoch, step, step_iou))

                total_valid_iou = 0
                for step in range(0, n_valid, flags.batch_size):
                    X_valid, y_valid = sess.run([X_valid_op, y_valid_op])
                    step_iou, step_summary = sess.run(
                        [IOU_op, summary_op],
                        feed_dict={X: X_valid,
                                   y: y_valid,
                                   mode: False})
                    total_valid_iou += step_iou * X_valid.shape[0]

                total_test_iou = 0
                for step in range(0, n_test, flags.batch_size):
                    X_test, y_test = sess.run([X_test_op, y_test_op])
                    step_iou, step_summary = sess.run(
                        [IOU_op, summary_op],
                        feed_dict={X: X_test,
                                   y: y_test,
                                   mode: False})
                    total_test_iou += step_iou * X_test.shape[0]
                    test_summary_writer.add_summary(step_summary, (epoch + 1) * (step + 1))

                logging.info("Epoch {:}, Average Train IOU = {:.4f}, Average Validation IOU = {:.4f}, Average Test IOU = {:.4f}".format(
                    epoch, (total_train_iou/n_train), (total_valid_iou/n_valid), (total_test_iou/n_test)))

            saver.save(sess, "{}/model.ckpt".format(flags.ckdir))
            logging.info("Optimization Finished!")

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.ckdir))


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
