import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("version", nargs="+", default="v1", choices=["v1", "v2"])
args = parser.parse_args()


def predict_v1():
    import tensorflow.compat.v1 as tf
    with tf.Session() as sess:
        tf.saved_model.load(
            sess, [tf.saved_model.tag_constants.SERVING], "./savedmodel-v1"
        )
        ress = sess.run(["embedding_lookup/Identity_1:0", "Identity:0"],
                        feed_dict={"Placeholder:0": [10, 20]})
        res = ress[1]
        print(res.shape, res.dtype)


def predict_v2():
    import tensorflow as tf
    model = tf.saved_model.load("./savedmodel-v2")
    print(model([10, 20]).numpy().tolist())


if __name__ == "__main__":
    if "v1" in args.version:
        predict_v1()
    if "v2" in args.version:
        predict_v2()
