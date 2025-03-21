import argparse
import train
import resample_dataset
import models_analysis
import baseline_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="results", help="Output directory (default: ./results/)")
    parser.add_argument("-om", "--outputdirmodels", type=str, default="models", help="Output directory (default: ./models/)")
    parser.add_argument("-d", "--datadir", type=str, default="dataset",
                        help="Data directory (default: ./dataset/)")
    parser.add_argument("-rs", type=bool, default=False, help="Resample signals.")
    parser.add_argument("-test", type=bool, default=False, help="Analyse model results.")
    parser.add_argument("-dummy", type=bool, default=False, help="Run experiments for baseline models.")
    parser.add_argument("-a", "--modelversion", type=int, default=2, help="Model version: 1: FV, 2: AFV")
    parser.add_argument("-f", "--featscode", type=int, default=0, help="Features used: 0 all, 1 all (ACC Norm.), 2 ...")
    parser.add_argument("-tp", type=int, default=180, help="Number of seconds in the past (default is 180 = 3 min)")
    parser.add_argument("-tf", type=int, default=180, help="Number of seconds in the future (default is 180 = 3 min)")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: population model. Use 1 for person-dependent models)")
    parser.add_argument("-sp", "--splitcode", type=int, default=0,
                       help="Split type (default is 0: session split first 80 train/ final 20 test for test users. Use 1 for full session splits 80/20 for test users)")
    parser.add_argument("-bs", "--bin_size", type=int, default='15', help="Bin duration in seconds (default=15). Optimized only for PM so far...")
    parser.add_argument("-ct", "--classifiertype", type=int, default='1', help="1) Predict aggresive episode combined (AGG|SIB|ED) or 2) different aggresive behaviors with independent models (default=1)")
    args = parser.parse_args()
    print(args)


    tp, tf = args.tp, args.tf
    bin_size = args.bin_size
    #path_results = args.outputdir
    #path_models = args.outputdirmodels
    #path_data = args.datadir
    data_path = './dataset/'
    data_path_resampled = './dataset_resampled/'
    results_path = './results/'
    models_path = './models/'
    freq = 32

    #########
    class_type = args.classifiertype
    # 1: Predict whether there is aggressive behaviour in general (1 model, binary task).
    # 2: Predict whether there is aggressive behaviour taking into account the 3 types (3 independent models, binary tasks).
    # 3: Predict whether there is aggressive behaviour taking into account the 3 types (1 model, 3 classifiers).
    ##########

    #########
    model_version = args.modelversion
    # Version 0: SRC Base (depreciated), Generate samples in 15 second bins, extract features per bin, labels
    #   indicate if there was an aggresive episode within the bin.
    # Version 1: Generate samples in 15-second bins, get observation windows composed of bins in the interval (t-tp,t),
    #   extract features per bin, labels indicate whether there was an aggressive episode within the prediction window,
    #   i.e., bins in the interval (t, t+tf).
    # Version 2: Generate samples in 15-second bins, get observation windows composed of bins in the interval (t-tp,t),
    #   extract features per bin, concatenate features with aggObs, which indicates if there was an aggressive episode
    #   within the bin, labels indicate whether there was an aggressive episode within the prediction window,
    #   i.e., bins in the interval (t, t+tf).
    ##########

    #########
    feats_code = args.featscode
    if feats_code == 2:
        model_version = 2 # para asociarlo con las funciones de creación de DS que tienen en cuenta la var AGGObsr. Modificar en el futuro
    # 0: All features, ACC in 3 axes (BVP, EDA, ACCx, ACCy, ACCz, AGGObs)
    # 1: All features, ACC Norm
    # 2: Only AGGObs
    # 3: All but ACC
    # 4: All but BVP
    # 5: All but EDA
    # 6: Only BVP
    # 7: Only ACC x,y,z
    # 8: Only EDA
    ##########

    #########
    split_code = args.splitcode
    # 0: Session split 80/20 for test users
    # 1: Session split, full sessions for test users
    ##########


    if args.dummy:
        print('Exps. baseline...')
        if class_type != 2:
            print('class_type: Not implemented yet... default 1.')
            class_type = 1
        if args.model == 0:
            baseline_models.start_exps_PM_dummy(tp, tf, freq, data_path_resampled, results_path, model_version, feats_code, split_code, bin_size, seed=1)

        elif args.model == 1:
            baseline_models.start_exps_PDM_dummy(tp, tf, freq, data_path_resampled, results_path, model_version, feats_code, split_code,
                             bin_size, seed=1)
        else:
            print('model_type: Error. default 1.')
    else:
        if args.test:
            print('Start testing')
            if args.model == 0:
                if class_type == 1:
                    models_analysis.test_model(models_path, model_version, feats_code, tf, tp, bin_size, split_code)
                elif class_type == 2:
                    models_analysis.test_model_multi(models_path, model_version, feats_code, tf, tp, bin_size, split_code)
            elif args.model == 1:
                models_analysis.analyse_PDM_results(results_path, model_version, feats_code, tf, tp, bin_size, split_code)

        else:
            if args.rs:
                resample_dataset.resample_dataset(data_path,data_path_resampled)
            if class_type == 1:
                if args.model == 0:
                    print('Exps PM:')
                    train.start_exps_PM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size)
                if args.model == 1:
                    print('Exps PDM:')
                    train.start_exps_PDM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size)
            elif class_type == 2:
                print('Exps. 3 independent models for each behavior (SIB, AGG, ED):')
                train.start_exps_PM_multi(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size)
                print('For ensemble results (soft voting), run -test = True.')
            else:
                print('Not supported yet...')