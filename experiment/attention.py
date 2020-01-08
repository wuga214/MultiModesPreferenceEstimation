import json
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.progress import WorkSplitter
from utils.optimizers import Optimizer
from utils.modelnames import models
from utils.functions import get_attention_example_items, write_latex,read_template


def attention(Rtrain, Rtest, index_map, item_names, latex_path, fig_path, settings_df, template_path, gpu_on=True):
    progress = WorkSplitter()
    m, n = Rtrain.shape


    for idx, row in settings_df.iterrows():
        row = row.to_dict()

        progress.section(json.dumps(row))

        if 'optimizer' not in row.keys():
            row['optimizer'] = 'Adam'

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

        attentions, kernels, predictions = mmup_model.interprate(Rtest[:100])

        attentions = attentions*1000

        visualization_samples = get_attention_example_items(Rtest[:100], predictions, 9)

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

        mmup_model.sess.close()
        tf.reset_default_graph()

    return
