import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", default="./data", type=str, help="Data root path.")
parser.add_argument("--dataset", default="IEMOCAP", type=str, help="Name of used dataset.")
parser.add_argument("--cuda", default=True, type=bool, help="Use gpu to train.")
parser.add_argument('--batch-size', type=int, default=30, help='batch size')
parser.add_argument('--n_classes', type=int, default=6, help='Number of emotion classes.')
parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
parser.add_argument('--attention', default='general', help='Attention type')
parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')

args = parser.parse_args()
