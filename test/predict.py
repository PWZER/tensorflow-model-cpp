import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("version", nargs="+", default="v1", choices=["v1", "v2"])
args = parser.parse_args()


def predict_v1():
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.saved_model.load(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING], "./savedmodel-v1"
        )
        res = sess.run("embedding_lookup/Identity:0", feed_dict={"Placeholder:0": [10, 20]})
        print(res.tolist())


def predict_v2():
    model = tf.saved_model.load("./savedmodel-v2")
    print(model([10, 20]).numpy().tolist())


if __name__ == "__main__":
    if "v1" in args.version:
        predict_v1()
    if "v2" in args.version:
        predict_v2()
