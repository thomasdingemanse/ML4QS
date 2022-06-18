##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

import sys
import re
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

def main():

    # Set up file names and locations.
    DATA_PATH = Path('./intermediate_datafiles/own_data')
    DATASET_FNAME = 'chapter2_result.csv'
    RESULT_FNAME =  'chapter3_result_outliers.csv'

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Of course we repeat some stuff from Chapter 3, namely to load the dataset

    # Read the result from the previous chapter, and make sure the index is of the type datetime.
    DATA_PATH = Path('./intermediate_datafiles/own_data')
    DATASET_FNAME = 'chapter5_result.csv'

    DataViz = VisualizeDataset(f'{__file__}_{FLAGS.features}')

    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    dataset.index = pd.to_datetime(dataset.index)

    def simplifyColumnName(name):
        name = re.sub(r'_X \(.+?\)', '_x', name)
        name = re.sub(r'_Y \(.+?\)', '_y', name)
        name = re.sub(r'_Z \(.+?\)', '_z', name)
        return name

    dataset = dataset.rename(columns=simplifyColumnName)

    # Let us consider our second task, namely the prediction of the accelerometer X-coordinate. We consider this as a temporal task.

    prepare = PrepareDatasetForLearning()

    train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression_by_time(dataset, 'acc_phone_x',
                                                                                        '2022-06-07 16:14:47',
                                                                                        '2022-06-07 16:32:00',
                                                                                        '2022-06-07 16:40:16')

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features that we will consider:

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))

    # Select subsets of the features that we will consider:

    basic_features = ['acc_phone_y','acc_phone_z', # acc_phone_x is excluded because it is the learning target
                    'gyr_phone_x','gyr_phone_y','gyr_phone_z',
                    'mag_phone_x','mag_phone_y','mag_phone_z',
                    'labelOptafel','labelLopen','labelRennen','labelFietsen','labelSpringen']
    pca_features = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','pca_7']
    time_features = [name for name in dataset.columns if ('temp_' in name and not 'acc_x' in name)]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    print('#basic features: ', len(basic_features))
    print('#PCA features: ', len(pca_features))
    print('#time features: ', len(time_features))
    print('#frequency features: ', len(freq_features))
    cluster_features = ['cluster']
    print('#cluster features: ', len(cluster_features))
    features_after_chapter_3 = list(set().union(basic_features, pca_features))
    features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))

    selected_features = ['gyr_phone_x']
    possible_feature_sets = [basic_features, features_after_chapter_3, features_after_chapter_4, features_after_chapter_5, selected_features]
    feature_names = ['initial set', 'Chapter 3', 'Chapter 4', 'Chapter 5', 'Selected features']

    # Let us first study whether the time series is stationary and what the autocorrelations are.

    dftest = adfuller(dataset['acc_phone_x'], autolag='AIC')

    plt.Figure(); autocorrelation_plot(dataset['acc_phone_x'])
    DataViz.save(plt)
    plt.show()

    # Now let us focus on the learning part.

    learner = TemporalRegressionAlgorithms()
    eval = RegressionEvaluation()

    # We repeat the experiment a number of times to get a bit more robust data as the initialization of e.g. the NN is random.

    repeats = 10

    # we set a washout time to give the NN's the time to stabilize. We do not compute the error during the washout time.

    washout_time = 10

    scores_over_all_algs = []

    # for i in range(0, len(possible_feature_sets)):

    if FLAGS.features == 'initial':
        i = 0
    elif FLAGS.features == 'ch3':
        i = 1
    elif FLAGS.features == 'ch4':
        i = 2
    elif FLAGS.features == 'ch5':
        i = 3
    elif FLAGS.features == 'selected':
        i = 4
    
    print(f'Evaluating for features {possible_feature_sets[i]}')
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]
    
    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_res = 0
    performance_tr_res_std = 0
    performance_te_res = 0
    performance_te_res_std = 0
    performance_tr_rnn = 0
    performance_tr_rnn_std = 0
    performance_te_rnn = 0
    performance_te_rnn_std = 0

    for repeat in range(0, repeats):
        print(f'---- run {repeat} ---')
        regr_train_y, regr_test_y = learner.reservoir_computing(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True, per_time_step=False)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.iloc[washout_time:,], regr_train_y.iloc[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.iloc[washout_time:,], regr_test_y.iloc[washout_time:,])

        performance_tr_res += mean_tr
        performance_tr_res_std += std_tr
        performance_te_res += mean_te
        performance_te_res_std += std_te

        regr_train_y, regr_test_y = learner.recurrent_neural_network(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.iloc[washout_time:,], regr_train_y.iloc[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.iloc[washout_time:,], regr_test_y.iloc[washout_time:,])

        performance_tr_rnn += mean_tr
        performance_tr_rnn_std += std_tr
        performance_te_rnn += mean_te
        performance_te_rnn_std += std_te


    # We only apply the time series in case of the basis features.
    if (feature_names[i] == 'initial set'):
        regr_train_y, regr_test_y = learner.time_series(selected_train_X, train_y, selected_test_X, test_y, gridsearch=True)

        mean_tr, std_tr = eval.mean_squared_error_with_std(train_y.iloc[washout_time:,], regr_train_y.iloc[washout_time:,])
        mean_te, std_te = eval.mean_squared_error_with_std(test_y.iloc[washout_time:,], regr_test_y.iloc[washout_time:,])

        overall_performance_tr_ts = mean_tr
        overall_performance_tr_ts_std = std_tr
        overall_performance_te_ts = mean_te
        overall_performance_te_ts_std = std_te
    else:
        overall_performance_tr_ts = 0
        overall_performance_tr_ts_std = 0
        overall_performance_te_ts = 0
        overall_performance_te_ts_std = 0

    overall_performance_tr_res = performance_tr_res/repeats
    overall_performance_tr_res_std = performance_tr_res_std/repeats
    overall_performance_te_res = performance_te_res/repeats
    overall_performance_te_res_std = performance_te_res_std/repeats
    overall_performance_tr_rnn = performance_tr_rnn/repeats
    overall_performance_tr_rnn_std = performance_tr_rnn_std/repeats
    overall_performance_te_rnn = performance_te_rnn/repeats
    overall_performance_te_rnn_std = performance_te_rnn_std/repeats

    scores_with_sd = [(overall_performance_tr_res, overall_performance_tr_res_std, overall_performance_te_res, overall_performance_te_res_std),
                    (overall_performance_tr_rnn, overall_performance_tr_rnn_std, overall_performance_te_rnn, overall_performance_te_rnn_std),
                    (overall_performance_tr_ts, overall_performance_tr_ts_std, overall_performance_te_ts, overall_performance_te_ts_std)]
    util.print_table_row_performances_regression(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), scores_with_sd)
    scores_over_all_algs.append(scores_with_sd)

    DataViz.plot_performances_regression(['Reservoir', 'RNN', 'Time series'], feature_names, scores_over_all_algs)

    regr_train_y, regr_test_y = learner.reservoir_computing(train_X[features_after_chapter_5], train_y, test_X[features_after_chapter_5], test_y, gridsearch=False)
    DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['acc_phone_x'], test_X.index, test_y, regr_test_y['acc_phone_x'], 'accelerometer X (m/s^2)')
    regr_train_y, regr_test_y = learner.recurrent_neural_network(train_X[basic_features], train_y, test_X[basic_features], test_y, gridsearch=True)
    DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['acc_phone_x'], test_X.index, test_y, regr_test_y['acc_phone_x'], 'accelerometer X (m/s^2)')
    regr_train_y, regr_test_y = learner.time_series(train_X[basic_features], train_y, test_X[basic_features], test_y, gridsearch=True)
    DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y, regr_train_y['acc_phone_x'], test_X.index, test_y, regr_test_y['acc_phone_x'], 'accelerometer X (m/s^2)')

    # And now some example code for using the dynamical systems model with parameter tuning (note: focus on predicting accelerometer data):

    train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression(copy.deepcopy(dataset), ['acc_phone_x', 'acc_phone_y'], 0.9, filter=False, temporal=True)

    output_sets = learner.dynamical_systems_model_nsga_2(train_X, train_y, test_X, test_y, ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
                                                        ['self.a * self.acc_phone_x + self.b * self.acc_phone_y', 'self.c * self.acc_phone_y + self.d * self.acc_phone_z', 'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
                                                        ['self.acc_phone_x', 'self.acc_phone_y'],
                                                        ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                        pop_size=10, max_generations=10, per_time_step=True)
    DataViz.plot_pareto_front(output_sets)

    DataViz.plot_numerical_prediction_versus_real_dynsys_mo(train_X.index, train_y, test_X.index, test_y, output_sets, 0, 'acc_phone_x')

    regr_train_y, regr_test_y = learner.dynamical_systems_model_ga(train_X, train_y, test_X, test_y, ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
                                                        ['self.a * self.acc_phone_x + self.b * self.acc_phone_y', 'self.c * self.acc_phone_y + self.d * self.acc_phone_z', 'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
                                                        ['self.acc_phone_x', 'self.acc_phone_y'],
                                                        ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                        pop_size=5, max_generations=10, per_time_step=True)

    DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['acc_phone_x'], regr_train_y['acc_phone_x'], test_X.index, test_y['acc_phone_x'], regr_test_y['acc_phone_x'], 'acc_phone_x')

    regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(train_X, train_y, test_X, test_y, ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
                                                        ['self.a * self.acc_phone_x + self.b * self.acc_phone_y', 'self.c * self.acc_phone_y + self.d * self.acc_phone_z', 'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
                                                        ['self.acc_phone_x', 'self.acc_phone_y'],
                                                        ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                        max_generations=10, per_time_step=True)

    DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['acc_phone_x'], regr_train_y['acc_phone_x'], test_X.index, test_y['acc_phone_x'], regr_test_y['acc_phone_x'], 'acc_phone_x')

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='selected',
                        help="Select what feature set to run: initial, ch3, ch4, ch5, or selected_features. \
                        'initial' to use only the initial feature set without any preprocessing \
                        'ch3' to use the features generated up to chapter 3 \
                        'ch4' to use the features generated up to chapter 4 \
                        'ch3' to use the features generated up to chapter 5 \
                        'selected' to use the final selected feature set",
                        choices=['selected', 'initial', 'ch3', 'ch4', 'ch5'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
