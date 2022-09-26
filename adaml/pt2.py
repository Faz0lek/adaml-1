import helper
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True, help="Path to a file containing data.")
    parser.add_argument("--compression", default="infer", help="Compression format.")

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_arguments()

    df = helper.load_data(args.data_path, args.compression)
    df = helper.preprocess_data(df)
    
    #df = (df - df.mean()) / df.std()
    
    #df.to_csv("output.txt")
    
    print(df.isna().sum().to_string())
