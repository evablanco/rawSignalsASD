import argparse
import train
import resample_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="results", help="Output directory (default: ./results/)")
    parser.add_argument("-om", "--outputdirmodels", type=str, default="models", help="Output directory (default: ./models/)")
    parser.add_argument("-d", "--datadir", type=str, default="dataset",
                        help="Data directory (default: ./dataset/)")
    parser.add_argument("-rs", type=bool, default=False, help="Resample signals.")
    parser.add_argument("-a", "--modelversion", type=int, default=2, help="Model version: 0, 1, 2 or 3")
    parser.add_argument("-featsonly", type=bool, default=False, help="Train models with raw features only")
    parser.add_argument("-tp", type=int, default=15, help="Number of seconds in the past (default is 60 = 1 min)")
    parser.add_argument("-tf", type=int, default=15, help="Number of seconds in the future (default is 180 = 3 min)")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: population model. Use 1 for individual models)")
    parser.add_argument("-bs", "--bin_size", type=int, default='15', help="Bin duration in seconds (default=15)")
    args = parser.parse_args()
    print(args)


    tp, tf = args.tp, args.tf
    #path_results = args.outputdir
    #path_models = args.outputdirmodels
    #path_data = args.datadir
    data_path = './dataset/'
    data_path_resampled = './dataset_resampled/'
    results_path = './results/'
    models_path = './models/'
    freq = 32

    #########
    model_version = args.modelversion
    # Version 0: generate samples with the entire past complete window, extract features and concatenate with
    # variable indicating if there was an aggressive behavior in the previous window.
    # Version 1: Generate samples in 15 second bins, group into N sequences based on tp, extract features per bin and
    # concatenate with t-1 label.
    # Version 2: Generate samples in 15 second bins, group into N sequences based on tp, extract features per bin and
    # concatenate with variable indicating if there was an aggressive behavior in the bin.
    # Version 3: generate samples with the entire past complete window, extract features and concatenate with
    # variable indicating if there was an aggressive behavior in the window. (wrong label)
    ##########

    #########
    only_features = args.featsonly # Test the model using only the raw signals
    # (without observed aggressive episode variable or previous labels).
    #########

    if args.rs:
        resample_dataset.resample_dataset(data_path,data_path_resampled)

    if args.model == 0:
        print('Exps PM:')
        train.start_exps_PM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, only_features)
    if args.model == 1:
        print('Exps DPM:')
        train.start_exps_PDM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, only_features)