import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(prog="THDNE")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_const', const='train', dest='mode', help="Run in training mode")
    mode_group.add_argument('--sampling', action='store_const', const='sampling', dest='mode', help="Run in sampling mode")

    parser.add_argument('--load-dir', metavar="DIR", type=Path,
                        help="Directory where trained models are stored. If not provided, new untrained models are created")
    parser.add_argument('--load-label', metavar="LABEL",
                        help="Optional label to select correct pair from load directory")

    training_group = parser.add_argument_group(title="Training", description="Settings affecting networks training process")
    training_group.add_argument('--iter', type=int, default=1000, help="Number of training iterations")
    training_group.add_argument('--batch', type=int, default=64, help="Batch size")
    training_group.add_argument('-k', '--disc-steps', metavar="STEPS", type=int, default=5,
                                help="Number of discriminator training steps per one generator training step")
    training_group.add_argument('-v', '--verbose', action='store_true', help="Display progress bar while training")
    training_group.add_argument('--sample-freq', metavar="N", type=int, help="Sample images every N iterations")
    training_group.add_argument('--histogram-freq', metavar="N", type=int,
                                help="Plot discriminator outputs on real and fake batches every N iterations")
    training_group.add_argument('--histogram-dir', metavar="DIR", type=Path,
                                help="Directory where plots should be saved")

    training_group.add_argument('--save-dir', metavar="DIR", type=Path,
                                help="Directory where trained models will be saved")
    training_group.add_argument('--save-label', metavar="LABEL", help="Optional label to mark models")

    sampling_group = parser.add_argument_group(title="Sampling", description="Settings affecting new image sampling process")
    sampling_group.add_argument('--sample-dir', metavar="DIR", type=Path,
                                help="Directory where generated images should be saved")
    sampling_group.add_argument('--images', metavar="N", type=int, default=1, help="Generate exactly N images at once")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    print(config)

