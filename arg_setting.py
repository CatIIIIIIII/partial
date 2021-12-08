import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

# ----- experiment ----- #
parser.add_argument("--data-root", default="./data", type=str, help="Data root path.")
parser.add_argument("--data-name", default="IEMOCAP", type=str, help="Name of used dataset.")
parser.add_argument("--data-path", default="", type=str, help="Path to data file.")
parser.add_argument("--utterance-path", default="", type=str, help="Path to utterance data file.")
parser.add_argument('--missing-rate', type=float, default=0.4, help='View missing rate [default: 0].')
parser.add_argument('--device', type=str, default="cuda:0", help='Train and test device.')

# ----- ep algorithm ----- #
parser.add_argument("--epochs-ep", default=1, type=int, help="Number of ep algorithm epochs.")
parser.add_argument("--num-views", default=0, type=int, help="Number of data views.")
parser.add_argument("--dim-features", default=[], type=list, help="Dimension of multi-view feature.")
parser.add_argument('--n-classes', type=int, default=0, help='Number of emotion classes.')

# ----- emotion algorithm ----- #
parser.add_argument('--e-batch-size', type=int, default=32, help='emotion batch size')

# ----- partial multi-view algorithm ----- #
parser.add_argument("--lr-p", default=[0.01, 0.01], type=list, help="Learning rate of partial multi-view algorithm.")
parser.add_argument('--lambda-p', type=float, default=10, help='trade off parameter [default: 1]')
parser.add_argument("--epochs-p", default=[60, 30], type=int, help="Number of partial algorithm epochs, [train, test]")
parser.add_argument('--dim-h', type=int, default=64, help='Dimension of representation h.')
parser.add_argument('--steps-p', type=list, default=[5, 5], help='Steps for inner optimize, [p(x|h), p(y|h)]')


parser.add_argument("--cuda", default=True, type=bool, help="Use gpu to train.")
parser.add_argument('--batch-size', type=int, default=30, help='batch size')
parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
parser.add_argument('--attention', default='general', help='Attention type')
parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
parser.add_argument('--tensorboard', action='store_true', default=True, help='Enables tensorboard log')
parser.add_argument('--log-dir', default="./runs/", help='Tensorboard log path.')

args = parser.parse_args()

args.num_views = 3 if args.data_name == "IEMOCAP" else 0
args.data_path = str(Path(args.data_root) / (args.data_name + "_features_raw.pkl"))
args.utterance_path = str(Path(args.data_root) / (args.data_name + "_utterance_raw.pkl"))
args.dim_features = [100, 512, 100] if args.data_name == "IEMOCAP" else []
args.n_classes = 6 if args.data_name == "IEMOCAP" else 0
