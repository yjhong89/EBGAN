from model import *
import tensorflow as tf
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='../CelebA')
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--pt_weight', type=float, default=0.1)
    parser.add_argument('--partition_index', type=str, default=10000)
    parser.add_argument('--is_training', type=str2bool, default=1)
    parser.add_argument('--pt', type=str2bool, default=True)
    parser.add_argument('--input_size', type=int, default=108)
    parser.add_argument('--margin', type=int, default=20, help='20 for celeba data')
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--num_channel', type=int, default=3)
    parser.add_argument('--final_dim', type=int, default=128)
    parser.add_argument('--disc_channel', type=int, default=64)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--showing_height', type=int, default=8)
    parser.add_argument('--showing_width', type=int, default=8)
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        ebgan = EBGAN(args, sess)
        if args.is_training:
            print('Training starts')
            ebgan.train()
        else:
            print('Test')
            ebgan.generator_test()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n' '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')

if __name__ == "__main__":
    main()
