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
    parser.add_argument("-a", "--algorithm", type=str, default="LR", help="classifier name default (SVM). Use 'LR' for"
                                                                          " Logistic Regression")
    parser.add_argument("-onset", type=int, default=1, help="onset")
    parser.add_argument("-tp", type=int, default=60, help="Number of seconds in the past (default is 60 = 1 min)")
    parser.add_argument("-tf", type=int, default=180, help="Number of seconds in the future (default is 180 = 3 min)")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: population model. Use 1 for individual models)")
    parser.add_argument("-s", "--min_sessions", type=int, default=0,
                        help="minimum number of sessions per subject (default is 2, only needed with -m 3)")
    parser.add_argument("-cv_folds", type=int, default=1, help="Number of cv folds (default is 10)")
    # parser.add_argument("-cv_reps", type=int, default=1, help="Number of cv repetitions (default is 5)")
    parser.add_argument("-fc", "--feature_code", type=int, default=7, help="Feature Code: 6 (default) for all signals ")
    parser.add_argument("-lr", "--lr", type=float, default=0.001, help="learning rate (Only required for NNs)")
    parser.add_argument("-ex", "--extraction", type=float, default=0, help="feature extraction type")
    parser.add_argument("-bs", "--bin_size", type=int, default='15', help="Bin duration in seconds (default=15)")
    args = parser.parse_args()
    print(args)

    feat_code = args.feature_code
    tp, tf = args.tp, args.tf
    #path_results = args.outputdir
    #path_models = args.outputdirmodels
    #path_data = args.datadir
    data_path = './dataset/'
    data_path_resampled = './dataset_resampled/'
    results_path = './results/'
    models_path = './models/'
    freq = 32

    model_version = 1
    #########
    # Model 0: generate samples with the entire past complete window, extract features and concatenate with
    # variable indicating if there was an aggressive behavior in the window itself.
    # Model 1: Generate samples in 15 second bins, group into N sequences based on tp, extract features and
    # concatenate with t-1 label.
    # Model 2: Generate samples in 15 second bins, group into N sequences based on tp, extract features and
    # concatenate with variable indicating if there was an aggressive behavior in the window itself.

    if args.rs:
        resample_dataset.resample_dataset(data_path,data_path_resampled)

    if args.model == 0:
        print('Exps PM:')
        train.start_exps_PM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version)
    if args.model == 1:
        print('Exps DPM:')
        train.start_exps_PDM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version)