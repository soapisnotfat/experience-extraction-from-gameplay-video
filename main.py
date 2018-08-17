from __future__ import print_function
import argparse
from solver import *

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')


args = parser.parse_args()


def main():
    domain_adaption_solver = TeacherStudent(args)
    domain_adaption_solver.run()


if __name__ == '__main__':
    main()
