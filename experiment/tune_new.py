from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
import pandas as pd
from sklearn.model_selection import ParameterGrid
from utils.progress import WorkSplitter
import os
from types import SimpleNamespace
from models.dvae import dvae, predict
from evaluation.metrics import evaluate
from copy import deepcopy

def hyper_parameter_tuning(data, train_R, valid_R, params, defaul_params,save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    param_list = list(params.keys())


    try:
        df = load_dataframe_csv(table_path, save_path)
    except:

        df = pd.DataFrame(columns=param_list)

    param_grid_list = list(ParameterGrid(params))
    print(len(param_grid_list))

    for param_set in param_grid_list:
        # check parameter set that already existed in the table and concatenate progress format string
        test_duplicate = None
        progress_format = ''
        for i in range(len(param_list)):
            if test_duplicate is None:
                test_duplicate = df[param_list[i]] == param_set[param_list[i]]
            else:
                test_duplicate = test_duplicate & (df[param_list[i]] == param_set[param_list[i]])
            progress_format = progress_format + param_list[i] + ': ' + str(param_set[param_list[i]]) + ', '

        if test_duplicate.any():
            print('skip ' + progress_format)
            continue
        progress.section(progress_format)
        # ##########train
        defaul_params.update(param_set)
        final_params = SimpleNamespace(**defaul_params)
        log_dir = '%s-%dT-%dB-%glr-%grg-%gkp-%gb-%gt-%gs-%dk-%dd-%d' % (
            data, final_params.epoch, final_params.batch_size, final_params.lr, final_params.lam, final_params.keep,
            final_params.beta, final_params.tau, final_params.std, final_params.kfac, final_params.dfac, final_params.seed)
        if final_params.nogb:
            log_dir += '-nogb'
        log_dir = os.path.join(final_params.logdir, log_dir)
        print(log_dir)
        prediction = dvae(log_dir, train_R, train_R.shape[1], final_params.kfac, final_params.dfac, final_params.lam, final_params.lr, final_params.seed, final_params.tau, final_params.std,
             final_params.nogb, final_params.beta,
             final_params.keep, final_params.topK[-1], final_params.epoch, final_params.batch_size)

        progress.section("Create Metrics")
        result = evaluate(prediction, valid_R, final_params.metric, final_params.topK)
        result_dict = deepcopy(param_set)
        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)
        save_dataframe_csv(df, table_path, save_path)

