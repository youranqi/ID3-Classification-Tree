from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import sys
from scipy.io import arff
from collections import Counter


# The following codes implement an ID3 tree with the assumptions:
# (i) the class attribute is binary
# (ii) the class attribute is named 'class'
# (iii) the class attribute is the last attribute listed in the header section


def get_data_meta(arff_name, save_file=False):
    # meta = {'age': ('numeric', ('<=', '>')), 'sex': ('nominal', ('female', 'male')), ...}
    data0, meta0 = arff.loadarff(arff_name)
    meta = {}
    for i in range(len(meta0.names())):
        if meta0.types()[i] != 'numeric':
            meta[meta0.names()[i]] = meta0.__getitem__(meta0.names()[i])
        else:
            meta[meta0.names()[i]] = meta0.__getitem__(meta0.names()[i])[:1] + (('<=', '>'), )
    data = pd.DataFrame(data0)
    for i in range(data.shape[1]):
        if meta[data.ix[:, i].name][0] != 'numeric':
            for j in range(data.shape[0]):
                data.ix[j, i] = data.ix[j, i].decode('utf-8')
    if save_file:
        # data.to_pickle('data_' + arff_name[:-5] + '.pkl')
        data.to_csv('data_' + arff_name[:-5] + '.csv')
        np.save('meta_' + arff_name[:-5], meta, fix_imports=True)
    return data, meta


def load_data_meta(arff_name):
    # data = pd.read_pickle('data_' + arff_name[:-5] + '.pkl')
    data = pd.read_csv('data_' + arff_name[:-5] + '.csv', index_col=0)
    meta = np.load('meta_' + arff_name[:-5] + '.npy').item()
    return data, meta


def get_split_index(feature, response, meta):
    # split_index for 'age' = {50: {'<=': [INDEX], '>': [INDEX]}, 60: {'<=': [INDEX], '>': [INDEX]}, ...}
    # split_index for 'sex' = {0: {'female': [INDEX], 'male': [INDEX]}}
    # For non-numeric features, some [INDEX]'s could be an empty list but split_index always has the above structure.
    # For numeric features, if all its values are equal or the responses are the same, split_index could be {}.
    # In ID3, there is only one way to split the nominal features so there is only one key - 0 in split_index.
    # INDEX is always the index in the original entire data set, because of the setting of pandas.
    split_index = {}
    if meta[feature.name][0] != 'numeric':
        level = meta[feature.name][1]
        sub_split_index = {}
        for i in range(len(level)):
            sub_split_index[level[i]] = feature.index[feature == level[i]].tolist()
        split_index[0] = sub_split_index
    else:
        # DetermineCandidateNumericSplits
        value = np.sort(feature.unique())
        value_index = []
        for i in range(value.shape[0]):
            value_index.append(feature.index[feature == value[i]].tolist())
        for i in range(len(value_index)):
            if (i > 0) and (pd.concat([response[value_index[i]], response[value_index[i - 1]]]).unique().shape[0] > 1):
                split_point = (value[i] + value[i - 1]) / 2.0
                split_index[split_point] = {'<=': feature.index[feature <= split_point].tolist(),
                                            '>': feature.index[feature > split_point].tolist()}
    return split_index


def get_all_split_index(subset, meta):
    # DetermineCandidateSplits
    all_split_index = {}
    for i in range(subset.shape[1] - 1):
        all_split_index[subset.ix[:, i].name] = get_split_index(subset.ix[:, i], subset.ix[:, -1], meta)
    return all_split_index


def get_infogain(split, response):
    label = response.unique()
    px = np.zeros((len(split), ))
    hyx = np.zeros((len(split), ))
    for i in range(len(split)):
        current_index = split[list(split.keys())[i]]
        # If current_index = [], the following block will not run and we go to the next i.
        if len(current_index) != 0:
            px[i] = len(current_index) / response.shape[0]
            conditional = response[current_index]
            for j in range(label.shape[0]):
                pyx = sum(conditional == label[j]) / conditional.shape[0]
                # Don't forget to consider the case when pyx is 0.
                if pyx != 0:
                    hyx[i] -= pyx * np.log2(pyx)
    hy = 0
    for j in range(label.shape[0]):
        py = sum(response == label[j]) / response.shape[0]
        # py can never be zero since label = response.unique().
        hy -= py * np.log2(py)
    return hy - np.sum(px * hyx)


def get_best_split(subset, all_split_index, eps=1e-5):
    # FindBestSplit
    # In order to handle the ties, we find the best split in a given order.
    # best_split = [infogain, ('age', 50)] or [infogain, ('sex', 0)] etc.
    best_split = [-np.inf, None]
    sorted_features = subset.columns.values[:(subset.shape[1] - 1)]
    for i in range(sorted_features.shape[0]):
        the_key = [k for k in all_split_index.keys() if k == sorted_features[i]][0]
        # If all_split_index[current_key] = {}, the following block will not run and we go to the next i.
        # If for every current_key, all_split_index[current_key] = {}, it returns best_split = [-np.inf, None],
        # which means there is no candidate splits (this can happen only when all features are numeric because
        # non-numeric features always have candidate splits).
        for j in sorted(list(all_split_index[the_key].keys())):
            infogain = get_infogain(all_split_index[the_key][j], subset.ix[:, -1])
            if infogain > best_split[0] + eps:
                best_split[0] = infogain
                best_split[1] = (the_key, j)
    return best_split


def get_splited_subset(subset, all_split_index, best_split):
    # splited_subset = {'<=': DATASET1, '>': DATASET2} or {'female': DATASET1, 'male': DATASET2} etc.
    current_split_index = all_split_index[best_split[1][0]][best_split[1][1]]
    splited_subset = {}
    for i in current_split_index.keys():
        splited_subset[i] = subset.ix[current_split_index[i], :]
    return splited_subset


def get_subtree(subset, meta, tree, m, last_predicted_label):
    # MakeSubtree
    if subset.shape[0] == 0:
        predicted_label = last_predicted_label
    elif subset.ix[:, -1].unique().shape[0] > 1:
        count_label = Counter(subset.ix[:, -1]).most_common()
        if count_label[0][1] == count_label[1][1]:
            predicted_label = last_predicted_label
        else:
            predicted_label = Counter(subset.ix[:, -1]).most_common(1)[0][0]
    else:
        predicted_label = subset.ix[:, -1].unique()[0]

    all_split_index = get_all_split_index(subset, meta)
    best_split = get_best_split(subset, all_split_index)
    stop = ((subset.shape[0] < m) or
            (subset.ix[:, -1].unique().shape[0] == 1) or
            (best_split[0] <= 0) or
            (best_split[1] is None))
    if stop:
        tree['class'] = predicted_label
    else:
        sorted_level = meta[best_split[1][0]][1]
        for i in range(len(sorted_level)):
            tree[best_split[1] + (sorted_level[i], )] = {}
        splited_subset = get_splited_subset(subset, all_split_index, best_split)
        assert len(splited_subset) == len(sorted_level)
        for i in range(len(sorted_level)):
            get_subtree(splited_subset[sorted_level[i]], meta, tree[best_split[1] + (sorted_level[i], )], m, predicted_label)


def plot_subtree(subtree, meta, depth=0, print_space='|       '):
    current_key = list(subtree.keys())
    assert np.unique(np.array([k[0] for k in current_key])).shape[0] == 1
    if len(current_key) == 1:
        assert current_key[0] == 'class'
        print(': ' + subtree[current_key[0]])
    else:
        sorted_level = meta[current_key[0][0]][1]
        # This guarantees we plot the branches according to the order of the feature values listed in the ARFF file.
        sorted_current_key = [current_key[j] for i in range(len(sorted_level)) for j in range(len(current_key))
                                             if current_key[j][2] == sorted_level[i]]
        for i in range(len(sorted_current_key)):
            if meta[sorted_current_key[i][0]][0] != 'numeric':
                if len(subtree[sorted_current_key[i]]) == 1:
                    print(print_space * depth + sorted_current_key[i][0] + ' = ' + sorted_current_key[i][2], end='')
                else:
                    print(print_space * depth + sorted_current_key[i][0] + ' = ' + sorted_current_key[i][2])
            else:
                if len(subtree[sorted_current_key[i]]) == 1:
                    print(print_space * depth + sorted_current_key[i][0] + ' ' + sorted_current_key[i][2] + ' ' + '{:.6f}'.format(sorted_current_key[i][1]), end='')
                else:
                    print(print_space * depth + sorted_current_key[i][0] + ' ' + sorted_current_key[i][2] + ' ' + '{:.6f}'.format(sorted_current_key[i][1]))
            plot_subtree(subtree[sorted_current_key[i]], meta, depth + 1)


def predict_subtree_single(testset_single, subtree, meta):
    while True:
        if len(subtree) == 1:
            assert list(subtree.keys())[0] == 'class'
            return subtree['class']
        current_key = list(subtree.keys())
        feature_name = current_key[0][0]
        if meta[feature_name][0] != 'numeric':
            branch = [k for k in current_key if k[2] == testset_single.ix[feature_name]][0]
        else:
            if current_key[0][1] >= testset_single.ix[feature_name]:
                branch = [k for k in current_key if k[2] == '<='][0]
            else:
                branch = [k for k in current_key if k[2] == '>'][0]
        subtree = subtree[branch]


def predict_subtree(testset, subtree, meta, return_correct=False):
    print('<Predictions for the Test Set Instances>')
    correct = 0
    for i in range(testset.shape[0]):
        predicted_label = predict_subtree_single(testset.ix[i, :], subtree, meta)
        if testset.ix[i, -1] == predicted_label:
            correct += 1
        print(str(i + 1) + ': Actual: ' + testset.ix[i, -1] + ' Predicted: ' + predicted_label)
    print('Number of correctly classified: ' + str(correct) + ' Total number of test instances: ' + str(testset.shape[0]))
    if return_correct:
        return correct


def get_accuracy(name_train, name_test, m=4, p=(0.05, 0.1, 0.2, 0.5, 1), rep=10, seed=0):
    np.random.seed(seed)
    data_train, meta_train = get_data_meta(name_train)
    data_test, meta_test = get_data_meta(name_test)
    accuracy = np.zeros((len(p), rep))
    for i in range(len(p)):
        for j in range(rep):
            tree = {}
            random_select = np.random.choice(data_train.shape[0], int(p[i] * data_train.shape[0]), replace=False)
            get_subtree(data_train.ix[random_select, :], meta_train, tree, m, Counter(data_train.ix[random_select, -1]).most_common(1)[0][0])
            accuracy[i, j] = predict_subtree(data_test, tree, meta_test, True) / data_test.shape[0]
    return accuracy
