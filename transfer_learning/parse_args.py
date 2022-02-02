from argparse import ArgumentParser

def parse_args(args):
    ap = ArgumentParser()
    ap.add_argument(
        "-i", "--images", required=True, help="Path to images"
    )
    ap.add_argument(
        "-l", "--labels", required=True, help="Path to labels"
    )
    ap.add_argument(
        "-m", "--model", required=False, help="Path to model"
    )
    ap.add_argument(
        "-w", "--weights", required=True, help="Path to weights of pretrained model"
    )
    ap.add_argument(
        "-f", "--filter", required=False, help="string to specify which files to use: eg. '('ids_' in file or 'passports_' in file)' only ids and passports are used for training" 
    )
    ap.add_argument(
        "--flow_from_dir", action='store_true'
    )
    args = vars(ap.parse_args(args))
    return args