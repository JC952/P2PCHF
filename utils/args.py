from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')

    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')

    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')

    parser.add_argument('--csv_log', type=int, default=1,
                        help='Enable csv logging')

