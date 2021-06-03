import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("version", nargs="+", default="v1", choices=["v1", "v2"])
args = parser.parse_args()


def gen_params():
    params = []
    for i in range(256):
        row = []
        for j in range(10):
            row.append(float(i) + float(j) * 0.1)
        params.append(row)
    return params


def gen_v1_model():
    with tf.compat.v1.Session() as sess:
        ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])

        # params = tf.constant(gen_params())
        params = tf.Variable(gen_params(), name="embedding_lookup_weight")
        output = tf.nn.embedding_lookup(params, ids, name="embedding_lookup")
        sess.run(tf.compat.v1.initialize_all_variables())
        res = sess.run(output, feed_dict={ids: [10, 20]})
        print(res.tolist())

        # build saved model
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("./savedmodel-v1")
        signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={"input": tf.compat.v1.saved_model.utils.build_tensor_info(ids)},
            outputs={"output": tf.compat.v1.saved_model.utils.build_tensor_info(output)},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={"predict": signature},
            assets_collection=None,
        )
        builder.save(as_text=True)


def gen_v2_model():
    class Model(tf.Module):
        def __init__(self):
            self.params = tf.Variable(gen_params())

        @tf.function(input_signature=[tf.TensorSpec(name="input", shape=[None], dtype=tf.int64)])
        def __call__(self, ids):
            return tf.nn.embedding_lookup(self.params, ids)

    model = Model()
    print(model([10, 20]).numpy().tolist())
    tf.saved_model.save(model, "./savedmodel-v2", signatures={"predict": model.__call__})


if __name__ == "__main__":
    if "v1" in args.version:
        gen_v1_model()
    if "v2" in args.version:
        gen_v2_model()
