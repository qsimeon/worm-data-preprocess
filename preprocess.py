from preprocess._main import process_data, PREPROCESS_CONFIG, json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess worm data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed configuration')
    args = parser.parse_args()

    print("Beginning preprocessing...\n")
    if args.verbose:
        print("Configuration:", json.dumps(PREPROCESS_CONFIG, indent=2), end="\n\n")
    else:
        print("Using configuration from preprocess/config.py\n")
    
    process_data(PREPROCESS_CONFIG)