import argparse
import math
from pathlib import Path
import torch
import numpy as np

parser = argparse.ArgumentParser()

# ----- experiment ----- #
parser.add_argument("--data-root", default="./data", type=str, help="Data root path.")
parser.add_argument("--data-name", default="IEMOCAP", type=str, help="Name of used dataset.")
parser.add_argument("--data-path", default="", type=str, help="Path to data file.")
parser.add_argument("--utterance-path", default="", type=str, help="Path to utterance data file.")
parser.add_argument('--missing-rate', type=float, default=0, help='View missing rate [default: 0].')
parser.add_argument('--device', type=str, default="cuda", help='Train and test device.')
parser.add_argument('--use-p', type=bool, default=True, help='Use partial multi-view algorithm.')
parser.add_argument('--party', type=int, default=0, help='Dialogue party number.')

# ----- ep algorithm ----- #
parser.add_argument("--epochs-init", default=30, type=int, help="Number of context init epochs.")
parser.add_argument("--epochs-ep", default=5, type=int, help="Number of ep algorithm epochs.")
parser.add_argument("--num-views", default=0, type=int, help="Number of data views.")
parser.add_argument("--dim-features", default=[], type=list, help="Dimension of multi-view feature. [text, visual, "
                                                                  "audio]")
parser.add_argument('--n-classes', type=int, default=0, help='Number of emotion classes.')
parser.add_argument('--loss-weights', default=None, help='Use weight to balanced classes.')

# ----- emotion algorithm ----- #
parser.add_argument('--e-batch-size', type=int, default=30, help='emotion batch size')
parser.add_argument('--dim-g', type=int, default=512, help='Dimension state of global state.')
parser.add_argument('--dim-p', type=int, default=512, help='Dimension state of personal state.')
parser.add_argument('--dim-e', type=int, default=256, help='Dimension state of emotional state.')
parser.add_argument('--dim-y', type=int, default=128, help='Dimension state of classifier state.')
parser.add_argument('--dim-a', type=int, default=128, help='Dimension state of attention state.')
parser.add_argument('--context-attention', default='general', help='Global state attention type.')
parser.add_argument('--party-attention', default='general', help='Party state attention type')

parser.add_argument('--lr-e', type=list, default=[1e-4, 1e-5], help='learning rate, [lr, L2 regularization]')
parser.add_argument('--model-type', type=str, default="base", help='Model used to classify emotion.')
parser.add_argument("--epochs-e", default=10, type=int, help="Number of emotional algorithm epochs.")
parser.add_argument("--steps-e", default=[1, 1], type=int, help="Steps of emotional algorithm train and test")

# ----- partial multi-view algorithm ----- #
parser.add_argument("--lr-p", default=[1e-3, 1e-3], type=list, help="Learning rate of partial multi-view algorithm.")
parser.add_argument('--lambda-p', type=float, default=0.1, help='trade off parameter [default: 1]')
parser.add_argument("--epochs-p", default=[400, 400], type=list,
                    help="Number of partial algorithm epochs, [train, test]")  # [60, 30]
parser.add_argument('--dim-h', type=int, default=128, help='Dimension of representation h.')
parser.add_argument('--steps-p', type=list, default=[5, 5], help='Steps for inner optimize, [p(x|h), p(y|h)]')
parser.add_argument('--p-batch-size', type=int, default=128, help='Batch size for partial algorithm')

parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
parser.add_argument('--tensorboard', action='store_true', default=True, help='Enables tensorboard log')
parser.add_argument('--log-dir', default="./runs/", help='Tensorboard log path.')

args = parser.parse_args()

args.data_path = str(Path(args.data_root) / (args.data_name + "_features_raw.pkl"))
args.utterance_path = str(Path(args.data_root) / (args.data_name + "_utterance_raw.pkl"))

if args.data_name == "IEMOCAP":
    args.num_views = 3
    args.dim_features = [100, 512, 100]
    args.n_classes = 6
    args.loss_weights = torch.tensor([
        1 / 0.086747,
        1 / 0.144406,
        1 / 0.227883,
        1 / 0.160585,
        1 / 0.127711,
        1 / 0.252668,
    ], dtype=torch.float)
    args.party = 2

if args.data_name == "MELD":
    args.num_views = 2
    args.dim_features = [600, 300]
    args.n_classes = 7

    # unique = [0, 1, 2, 3, 4, 5, 6]
    # labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}
    # total = np.sum(list(labels_dict.values()))
    # weights = []
    # for key in unique:
    #     score = math.log(total / labels_dict[key])
    #     weights.append(score)
    weights = [1, 1, 1, 1, 1, 1, 1]
    args.loss_weights = torch.tensor(weights, dtype=torch.float)
    args.party = 9

if args.data_name == "EMORY":
    args.num_views = 4
    args.dim_features = [1024, 1024, 1024, 1024]
    args.n_classes = 7

    # unique = [0, 1, 2, 3, 4, 5, 6]
    # labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}
    # total = np.sum(list(labels_dict.values()))
    # weights = []
    # for key in unique:
    #     score = math.log(total / labels_dict[key])
    #     weights.append(score)
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 2099, 1: 968, 2: 831, 3: 3095, 4: 595, 5: 717, 6: 1184}
    # 0 happy, 1 neutral, 2 anger, 3 sad, 4 fear, 5 surprise, 6 disgust
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(total / labels_dict[key])
        weights.append(score)
    # weights = [1, 1, 1, 1, 1, 1, 1]

    args.loss_weights = torch.tensor(weights, dtype=torch.float)
    args.party = 2
    args.e_batch_size = 128

