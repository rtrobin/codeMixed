import os
import tensorflow as tf

def main():
        
    with tf.Session() as sess:
        # First create a `Saver` object (for saving and rebuilding a
        # model) and import your `MetaGraphDef` protocol buffer into it:

        # saver = tf.train.import_meta_graph('/code/cascade_rcnn_trt/ckpt/model.ckpt-976.meta')
        saver = tf.train.import_meta_graph('./model/voc_120001model.ckpt.meta')

        # Then restore your training data from checkpoint files:
        saver.restore(sess, './model/voc_120001model.ckpt')

        saver = tf.train.export_meta_graph('./model/voc_120001model.ckpt.meta.txt', as_text=True)

        # finally, freeze the graph:
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=['postprocess_fastrcnn_stage3/concat', 'postprocess_fastrcnn_stage3/concat_1'])

        model_dir = './model/'
        file_name_pb = 'eval_model.pb'
        tf.train.write_graph(frozen_graph, model_dir, file_name_pb, as_text=False)

if __name__ == '__main__':
    main()
