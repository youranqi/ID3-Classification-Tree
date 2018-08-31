from functions import *

for arff_name in ('heart', 'diabetes'):
    for m in (2, 4, 10, 20):
        print('======================== ' + arff_name + ' m = ' + str(m) + ' ========================')
        data_train, meta_train = get_data_meta(arff_name + '_train.arff')
        tree = {}
        get_subtree(data_train, meta_train, tree, m, Counter(data_train.ix[:, -1]).most_common(1)[0][0])
        plot_subtree(tree, meta_train)

        data_test, meta_test = get_data_meta(arff_name + '_test.arff')
        predict_subtree(data_test, tree, meta_test)
