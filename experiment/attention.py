import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.progress import WorkSplitter
from utils.optimizers import Optimizer
from utils.modelnames import models
from utils.functions import get_attention_example_items, write_latex, read_template
from plots.rec_plots import multi_modes_histogram,multi_modes_count
import glob


def attention(Rtrain, Rvalid, Rtest, index_map, item_names, latex_path, fig_path, settings_df, template_path,
              case_study=False,
              gpu_on=True):
    progress = WorkSplitter()
    m, n = Rtrain.shape

    Rtest = Rvalid + Rtest


    for idx, row in settings_df.iterrows():
        row = row.to_dict()

        progress.section(json.dumps(row))

        if 'optimizer' not in row.keys():
            row['optimizer'] = 'Adam'

        # row['epoch'] = 1

        mmup_model = models[row['model']](Rtrain,
                                          embedded_matrix=np.empty((0)),
                                          mode_dim=row['mode_dimension'],
                                          key_dim=row['key_dimension'],
                                          batch_size=row['batch_size'],
                                          optimizer=row['optimizer'],
                                          learning_rate=row['learning_rate'],
                                          normalize=False,
                                          iteration=row['iteration'],
                                          epoch=row['epoch'],
                                          rank=row['rank'],
                                          corruption=row['corruption'],
                                          gpu_on=gpu_on,
                                          lamb=row['lambda'],
                                          alpha=row['alpha'],
                                          seed=1,
                                          root=row['root'],
                                          return_model=True)

        train_batches = mmup_model.get_batches(Rtrain, 100)
        test_batches = mmup_model.get_batches(Rtest, 100)

        cmd = "rm {0}/*.pdf".format(fig_path)
        os.system(cmd)

        interaction_modes_counts = []

        train_batches = train_batches

        for i in range(len(train_batches)):

            attentions, kernels, predictions = mmup_model.interprate(train_batches[i])

            visualization_samples = get_attention_example_items(train_batches[i], predictions, test_batches[i], 9)

            interaction_counts = np.sum(train_batches[i], axis=1).tolist()
            for j in range(kernels.shape[0]):
                if interaction_counts[j][0] > 200:
                    continue
                interaction_modes_counts.append([interaction_counts[j][0],
                                                 len(np.unique(kernels[j][visualization_samples[j][1]]))])


            if case_study:
                items = []

                for i in index_map:
                    try:
                        name = item_names[item_names['ItemID'] == i]['Name'].values[0]
                    except:
                        name = "Unknown"

                    items.append(name)

                items = np.array(items)

                latex_template = read_template(template_path)

                write_latex(visualization_samples, attentions, kernels, items, latex_template, latex_path)

                tex_files = glob.glob(latex_path + "/*.tex")

                for tex in tex_files:
                    cmd = "pdflatex -halt-on-error -output-directory {0} {1}".format(fig_path, tex)
                    os.system(cmd)

                cmd = "rm {0}/*.log".format(fig_path)
                os.system(cmd)
                cmd = "rm {0}/*.aux".format(fig_path)
                os.system(cmd)
                cmd = "rm {0}/*.tex".format(latex_path)
                os.system(cmd)

        interaction_modes_counts = pd.DataFrame(np.array(interaction_modes_counts), columns=['x', 'y'])

        multi_modes_histogram(interaction_modes_counts)
        multi_modes_count(interaction_modes_counts)

        interaction_modes_counts.to_csv(template_path+"/modes_count.csv")

        mmup_model.sess.close()
        tf.reset_default_graph()

    return
