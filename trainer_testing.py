import argparse
import importlib
import pickle
import os
import tensorflow as tf
import data_reader
import numpy as np


def run_test_visual(model_arch_module, dataset_dir, model_path):
    import data_visualizer as dv
    import data_config as cfg

    model = model_arch_module.build_model_arch()
    _, _, test_data = data_reader.read_data_set_dir(dataset_dir, cfg.one_hot, 24)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        labels = cfg.one_hot_labels

        for step in range(test_data.batch_count):
            images, one_hot = test_data.next_batch()
            truth_indexes = np.argmax(one_hot, 1)

            pred = model.predict(sess, images)[0]
            pred_indexes = np.argmax(pred, 1)
            pred_labels = [labels[i] for i in pred_indexes]

            dv.show_images_with_truth(images, pred_labels, truth_indexes, pred_indexes)


def run_test(model_arch_module, dataset_dir, model_path, result_dir, top_k=1):
    import data_config as cfg

    os.makedirs(result_dir, exist_ok=True)

    model = model_arch_module.build_model_arch()
    _, _, test_data = data_reader.read_data_set_dir(dataset_dir, cfg.one_hot, 64)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        overall_truth_indices = []
        overall_pred_probs = []
        overall_correct_classification = 0.
        for step in range(test_data.batch_count):
            images, one_hot = test_data.next_batch()
            truth_indexes = np.argmax(one_hot, 1)

            pred = model.predict(sess, images)[0]
            pred_indexes = np.argmax(pred, 1)

            if top_k == 1:
                correct_classification = (truth_indexes == pred_indexes).astype(np.float32)
            else:
                correct_classification = tf.nn.in_top_k(pred, truth_indexes, k=top_k)
                correct_classification = sess.run(correct_classification).astype(np.float32)

            overall_truth_indices.extend(truth_indexes)
            overall_pred_probs.extend(pred)
            overall_correct_classification = overall_correct_classification + correct_classification.sum()
            print("Step %d, accuracy %f" % (step, correct_classification.mean()))

        overall_acc = overall_correct_classification / float(test_data.data_set_size)
        print("Overall accuracy: %f" % overall_acc)

        with open(os.path.join(result_dir, 'truth_indices.pkl'), 'wb') as f:
            pickle.dump(overall_truth_indices, f)
        with open(os.path.join(result_dir, 'pred_probs.pkl'), 'wb') as f:
            pickle.dump(overall_pred_probs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model CNN')
    parser.add_argument('--model-module', type=str, help='Python module string untuk model cnn', required=True)
    parser.add_argument('--dataset-dir', type=str, help='Direktori dataset', required=True)
    parser.add_argument('--model-path', type=str, help='Path model CNN', required=True)
    parser.add_argument('--result-dir', type=str, help='Output direktori', required=False)
    parser.add_argument('--top-k', type=int, help='Akurasi Top-K', default=1, required=False)
    parser.add_argument('--type', type=str, help='Jalankan test visual atau cmd [cmd | vis]', default='cmd', required=False)

    args = parser.parse_args()

    if not args.result_dir:
        args.result_dir = os.path.join('trainer_test_result/', args.model_module)

    print('Run model tester:')
    print('\tModel module name: %s' % args.model_module)
    print('\tDataset dir: %s' % args.dataset_dir)
    print('\tModel path: %s' % args.model_path)
    print('\tTop-K: %d' % args.top_k)
    if args.type == 'vis':
        print('================= Visualize =================')
        run_test_visual(importlib.import_module(args.model_module), args.dataset_dir, args.model_path)
    elif args.type == 'cmd':
        print('Run test')
        run_test(importlib.import_module(args.model_module), args.dataset_dir, args.model_path, args.result_dir, top_k=args.top_k)
