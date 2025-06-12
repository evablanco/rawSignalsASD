import argparse
import resample_dataset
import train
import test
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="results", help="Output directory (default: ./results/)")
    parser.add_argument("-om", "--outputdirmodels", type=str, default="models", help="Output directory (default: ./models/)")
    parser.add_argument("-d", "--datadir", type=str, default="dataset",
                        help="Data directory (default: ./dataset/)")
    parser.add_argument("-rs", type=bool, default=False, help="Resample signals.")
    parser.add_argument("-test", type=bool, default=False, help="Analyse model results.")
    parser.add_argument("-v", "--model_version", type=int, default=1, help="Model version: 1")
    parser.add_argument("-f", "--feats_code", type=int, default=0, help="Features used: 0 all, 1-3 leave one out, 4-6 only one source.")
    parser.add_argument("-tp", type=int, default=300, help="Number of seconds in the past (default is 60 = 1 min)")
    parser.add_argument("-tf", type=int, default=300, help="Number of seconds in the future (default is 180 = 3 min)")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: PM. Use 1 for intra TL in PM, and 3 for PDM.")
    parser.add_argument("-sp", "--split_code", type=int, default=1,
                       help="Split type (default is 0: PM-Leave Subjects Out (PMs). Use 1 for PM-Session Splits 80/20) (default in PDMs)")
    parser.add_argument("-bs", "--bin_size", type=int, default='15', help="Bin duration in seconds (default=15)")
    parser.add_argument("-sb", "--specific_behavior", type=int, default='0', help="Predict aggresive episode combined (0) or different aggresive episodes (1) (default=0)")
    parser.add_argument("-cw", "--class_weights_type", type=int, default='1',
                        help="0 no class weights, 1 balanced, 2 custom")
    args = parser.parse_args()
    print(args)


    tp, tf = args.tp, args.tf
    bin_size = args.bin_size
    #path_results = args.outputdir
    #path_models = args.outputdirmodels
    #path_data = args.datadir
    data_path = './dataset/'
    root_exps = './normalized_sigs_smooth/rolling_avg/'  # './normalized_sigs/
    data_path_resampled = './dataset_resampled/'
    results_path = root_exps + 'results/'
    results_analysis_path = root_exps + 'results_analysis/'
    models_path = root_exps + 'models/'
    freq = 32

    ###os.makedirs(results_path, exist_ok=True)

    #########
    class_type = args.specific_behavior
    # 1: Predict whether there is aggressive behaviour in general (1 model, combined labels).
    # 2: Predict whether there is aggressive behaviour taking into account the 3 types (3 independent models, independent labels). TO-DO!!!
    ##########


    #########
    model_version = args.model_version
    # Version 1: Generate samples in N-second bins, get observation windows composed of bins in the interval (t-tp,t),
    #   extract features per bin, labels indicate the following depending on if there was an aggressive episode within
    #   the prediction window, i.e., bins in the interval (t, t+tf), or in the observation window, i.e., bins in
    #   the interval (t-tf, t).
    # - Attack: si hay ataque activo en t
    # - Pre-attack: si no hay ataque en t y hay uno en [t, t+tf]
    # - Post-attack: si no hay ataque en t ni futuro, pero uno terminó antes de t
    # - Calm: si no hay ataque en [t−tp, t+tf]
    ##########

    #########
    cw_type = args.class_weights_type
    # 0: No cw
    # 1: Balanced cw
    # 2: Custom cw, pre-attack x2 default
    ##########

    #########
    feats_code = args.feats_code
    # Features used: 0 all, 1-3 leave one out, 4-6 only one source.
    # 0: All features, ACC in 3 axes (BVP, EDA, ACCx, ACCy, ACCz)
    # 1: All but BVP
    # 2: All but EDA
    # 3: All but ACC
    # 4: Only BVP
    # 5: Only EDA
    # 6: Only ACC
    ##########

    #########
    split_code = args.split_code
    # 0: PM-LSS
    # 1: PM-SS
    ##########


    if args.test:
        print('Start testing')
        if args.model == 0: ## PM-SS
            if class_type == 0: # Combined labels
                test.run_full_evaluation_analysis(models_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type)
                #test.get_best_performing_models(results_folder="results", f1_threshold_csv="./results_analysis/f1_best_summary.csv",
                #    output_csv="summary_best_models.csv")
                test.run_binary_calm_vs_aggressive_analysis(models_path, model_version, feats_code, tf, tp, bin_size, split_code, cw_type)
            elif class_type == 1:  # Independent models
                pass
        elif args.model == 1: # Intra-domain TL
            #test.run_full_evaluation_analysis(models_path, model_version, feats_code, tf, tp, bin_size, split_code,
            #                                  cw_type)  ## TO-DO: Pasar model name (PM/PMTL)
            #test.run_binary_calm_vs_aggressive_analysis(models_path, model_version, feats_code, tf, tp, bin_size,
            #                                            split_code, cw_type)
            test.run_full_evaluation_analysis(models_path, model_version, feats_code, tf, tp, bin_size, split_code,
                                              cw_type)
            test.evaluate_and_plot_confusion_matrices(
                models_path, model_version, feats_code, tf, tp, bin_size,
                split_code, cw_type, output_dir=results_analysis_path, threshold_active=0.5, seed=1
            )

    else:
        if args.rs:
            resample_dataset.resample_dataset(data_path,data_path_resampled)
        if class_type == 0:
            if args.model == 0:
                print('Exps PM:')
                train.start_exps_PM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
            if args.model == 1:
                print('Exps intra_TL:')
                train.start_exps_PM_intraTL(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
            if args.model == 3:
                print('Exps PDM:')
                train.start_exps_PDM(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
            if args.model == 4:
                print('Exps PM-LOO: #TO-DO!!! :)')
                #train.start_exps_PM_v2(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)
        elif class_type == 1:
            print('Exps Multi-label (SIB, AGG, ED): #TO-DO!!! :)') # Independent models
            #train.start_exps_PM_multi(tp, tf, freq, data_path_resampled, results_path, models_path, model_version, feats_code, split_code, bin_size, cw_type)

        else:
            print('Not supported yet...')