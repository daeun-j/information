from __future__ import print_function
import argparse


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=6, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2022, metavar='S',
                    help='random seed (default: 2022)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_features', type=int, default=8, metavar='N',
                    help='how many features are used for training status')
parser.add_argument('--embedding_dims', type=int, default=4, metavar='N',
                    help='how many dims considered to be middle layer')
parser.add_argument('--lr', type=float, default=0.0005, metavar='N',
                    help='learning rate')
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                    help='# of workers in device')
parser.add_argument('--test_size', type=float, default=0.999, metavar='N',
                    help='test_size')
parser.add_argument('--result', type=str, default="result_0607/", metavar='N',
                    help='# of workers in device')
parser.add_argument('--clf', type=str, default="rf", metavar='N',
                    help='# of workers in device')