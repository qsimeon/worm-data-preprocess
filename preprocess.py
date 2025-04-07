from preprocess._main import process_data
from preprocess._pkg import PREPROCESS_CONFIG, json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess worm data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed configuration')
    parser.add_argument(
        "-n", "--neural", action="store_true", help="Process only neural data"
    )
    parser.add_argument(
        "-c", "--connectome", action="store_true", help="Process only connectome data"
    )

    args = parser.parse_args()

    if args.verbose:
        print("Configuration:", json.dumps(PREPROCESS_CONFIG, indent=2), end="\n\n")
    else:
        print("Using configuration from preprocess/config.py\n")
    
    if args.neural: # python preprocess.py -n -c just defaults to neural
        process_data(PREPROCESS_CONFIG, data_type="neural")
    elif args.connectome:
        process_data(PREPROCESS_CONFIG, data_type="connectome")
    else:
        process_data(PREPROCESS_CONFIG, data_type="all")
