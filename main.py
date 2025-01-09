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
    parser.add_argument("-a", "--modelversion", type=int, default=1, help="Model version: 1")
    parser.add_argument("-tp", type=int, default=60, help="Number of seconds in the past (default is 60 = 1 min)")
    parser.add_argument("-tf", type=int, default=180, help="Number of seconds in the future (default is 180 = 3 min)")
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
    # Version 0: SRC Base (depreciated), Generate samples in 15 second bins, extract features per bin, labels
    #   indicate if there was an aggresive episode within the bin.
    # Version 1: Generate samples in 15-second bins, get observation windows composed of bins in the interval (t-tp,t),
    #   extract features per bin, labels indicate whether there was an aggressive episode within the prediction window,
    #   i.e., bins in the interval (t, t+tf).
    # Version 2: Generate samples in 15-second bins, get observation windows composed of bins in the interval (t-tp,t),
    #   extract features per bin, concatenate features with aggObs,  which indicates if there was an aggressive episode
    #   within the bin, labels indicate whether there was an aggressive episode within the prediction window,
    #   i.e., bins in the interval (t, t+tf).
    ##########


    if args.rs:
        resample_dataset.resample_dataset(data_path,data_path_resampled)

    if args.model == 0:
        print('Exps PM:')
        train.start_exps_PM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version)
    if args.model == 1:
        print('Exps PDM:')
        train.start_exps_PDM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version)