from preprocess._main import process_data, PREPROCESS_CONFIG, json

if __name__ == "__main__":
    print("Beginning preprocessing...\n")
    print("Configuration:", json.dumps(PREPROCESS_CONFIG, indent=2), end="\n\n")
    process_data(PREPROCESS_CONFIG)