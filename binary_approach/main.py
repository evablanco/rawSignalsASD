import argparse
import train
import resample_dataset
import models_analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="results", help="Output directory (default: ./results/)")
    parser.add_argument("-om", "--outputdirmodels", type=str, default="models", help="Output directory (default: ./models/)")
    parser.add_argument("-d", "--datadir", type=str, default="dataset",
                        help="Data directory (default: ./dataset/)")
    parser.add_argument("-rs", type=bool, default=False, help="Resample signals.")
    parser.add_argument("-test", type=bool, default=False, help="Analyse model results.")
    parser.add_argument("-a", "--modelversion", type=int, default=2, help="Model version: 1 Fv, 2 AFV")
    parser.add_argument("-f", "--featscode", type=int, default=0, help="Features used: 0 all, 1 all (ACC Norm.), 2 ...")
    parser.add_argument("-tp", type=int, default=180, help="Number of seconds in the past (default is 60 = 1 min)")
    parser.add_argument("-tf", type=int, default=180, help="Number of seconds in the future (default is 180 = 3 min)")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: PM-SS. Use 1 for PDM, 2 for HM, and 3 for PM-LOO)")
    parser.add_argument("-sp", "--splitcode", type=int, default=0,
                       help="Split type (default is 0: session split 80/20. Use 1 for full session splits)")
    parser.add_argument("-bs", "--bin_size", type=int, default='1', help="Bin duration in seconds (default=15)")
    parser.add_argument("-ct", "--classifiertype", type=int, default='1', help="Predict aggresive episode or different aggresive episodes (default=1)")
    parser.add_argument("-cw", "--clasweightstype", type=int, default='0',
                        help="1 per label sample, 2 per prev-onset sample")
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

    ### quick testing arch. complexity: versions 3 and 4 (1-layer and 2-layes LSTMS)
    ### quick testing onset labels: model v1 (FV) -> v6 y model v2 (AFV) -> v5
    ##########

    #########
    cw_type = args.clasweightstype
    if cw_type == 2 and model_version < 2:
        print('2: Default model for this type. ')
        model_version = model_version + 1
    # 0: Balanced pos/neg samples, pero antes de implementar los diferentes exps...
    # 1: Balanced pos/neg samples.
    # 2: Onset samples
    ##########

    #########
    feats_code = args.featscode
    if feats_code == 2 and model_version < 2:  ### OJO CON MODEL VERSION 4... FV con onset labels...
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


    if args.test:
        print('Start testing')
        if args.model == 0:
            if class_type == 1:
                models_analysis.test_model(models_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, onset_only=False)
            elif class_type == 2:
                models_analysis.test_model_multi(models_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type)
        elif args.model == 1:
            models_analysis.analyse_PDM_results(results_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type, onset_only=False)

    else:
        if args.rs:
            resample_dataset.resample_dataset(data_path,data_path_resampled)
        if class_type == 1:
            if args.model == 0:
                print('Exps PM-SS:')
                train.start_exps_PM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
            if args.model == 1:
                print('Exps PDM:')
                train.start_exps_PDM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
            if args.model == 2:
                print('Exps HM:')
                train.start_exps_HM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
            if args.model == 3:
                print('Exps PM-LOO:')
                train.start_exps_PM_v2(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
        elif class_type == 2:
            print('Exps Multi-label (SIB, AGG, ED):')
            train.start_exps_PM_multi(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)

        else:
            print('Not supported yet...')