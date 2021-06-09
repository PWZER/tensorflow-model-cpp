import os
import shutil
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("version", nargs="+", default="v1", choices=["v1", "v2"])
args = parser.parse_args()


def gen_params(value=0.1):
    params = []
    for i in range(256):
        row = []
        for j in range(10):
            row.append(float(i) + float(j) * value)
        params.append(row)
    return params


def gen_v1_model():
    import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()

    export_dir = "./savedmodel-v1"
    shutil.rmtree(export_dir, ignore_errors=True)
    with tf.Session() as sess:
        ids = tf.placeholder(dtype=tf.int32, shape=[None], name="input")

        # params = tf.constant(gen_params())
        params = tf.Variable(gen_params(), name="embedding_lookup_weight")
        output = tf.nn.embedding_lookup(params, ids, name="embedding_lookup")
        sess.run(tf.global_variables_initializer())
        res = sess.run(output, feed_dict={ids: [10, 20]})
        print(res.tolist())

        # build saved model
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        predict_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"input": tf.saved_model.build_tensor_info(ids)},
            outputs={"output": tf.saved_model.build_tensor_info(output)},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "predict": predict_signature,
            },
            assets_collection=None,
        )
        builder.save(as_text=True)


def gen_v2_model():
    import tensorflow as tf

    export_dir = "./savedmodel-v2"
    shutil.rmtree(export_dir, ignore_errors=True)

    class Model(tf.Module):
        def __init__(self):
            self.params = tf.Variable(gen_params())

        @tf.function(input_signature=[tf.TensorSpec(name="input", shape=[None], dtype=tf.int64)])
        def __call__(self, ids):
            return tf.nn.embedding_lookup(self.params, ids)

    model = Model()
    print(model([10, 20]).numpy().tolist())
    tf.saved_model.save(model, export_dir, signatures={"predict": model.__call__})


if __name__ == "__main__":
    if "v1" in args.version:
        gen_v1_model()
    if "v2" in args.version:
        gen_v2_model()
