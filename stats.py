import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm

sns.set_theme()



def f1_score_for_dist(df_in, distance):
    df_look = df_in[(df_in.dist <= distance) | (df_in.y_true == -1)]
    return skm.f1_score(df_look.y_true.values, df_look.y_pred.values, average='micro')

def plot_dist_dep_bars():
    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Open data from pickle file
    collector_X, collector_y, collector_z, clf, X_train, X_test, y_train, y_test, o, z_train, z_test = pickle.load(open('all_0.pickle', 'rb'))

    # Create Dataframe for distnace depending stats
    steering_handle_overhang = 19
    dist_dep_metric = pd.DataFrame({'y_true': y_test, 'y_pred': o, 'dist': z_test - steering_handle_overhang})

    # Create dists TP, FP and FN for Classes 0 and 1
    TP0 = dist_dep_metric[(dist_dep_metric.y_true==0) & (dist_dep_metric.y_pred==0)].dist
    FP0 = dist_dep_metric[((dist_dep_metric.y_true==-1) | (dist_dep_metric.y_true==1)) & (dist_dep_metric.y_pred==0)].dist
    FN0 = dist_dep_metric[((dist_dep_metric.y_pred==-1) | (dist_dep_metric.y_pred==1)) & (dist_dep_metric.y_true==0)].dist

    TP1 = dist_dep_metric[(dist_dep_metric.y_true==1) & (dist_dep_metric.y_pred==1)].dist
    FP1 = dist_dep_metric[((dist_dep_metric.y_true==-1) | (dist_dep_metric.y_true==0)) & (dist_dep_metric.y_pred==1)].dist
    FN1 = dist_dep_metric[((dist_dep_metric.y_pred==-1) | (dist_dep_metric.y_pred==0)) & (dist_dep_metric.y_true==1)].dist

    hist_data = [[TP0, FP0, FN0], [TP1, FP1, FN1]]

    # Create figure
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 5))

    # Create both plots
    for ax, data, invert, text in zip(axes.ravel(), hist_data, [False, True], ['für Entgegenkommen', 'für Überholen']):
        # Plot histogram
        ax.hist(data, bins=24, range=(0,300), stacked=True)
        ax.set_ylim(bottom=0, top=105)

        # Add y label
        ax.set_ylabel(f"Anzahl Samples\n{text}", fontproperties=font_property)

        if invert:
            # Invert second subplot
            ax.invert_yaxis()

            # Remove second zero tick
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)

            # Plot vertical line at 1.5m
            ax.axvline(x=150, c=sns.color_palette()[3])

        else:
            ax.legend(['True Positive', 'False Positive', 'False Negative'], prop=font_property)

        # Set Computer Moedern as tick font
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_property)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_property)

    # Remove space
    plt.subplots_adjust(hspace=0)

    # Add text
    plt.xlabel('Abstand [cm]', fontproperties=font_property)

    # Export as pdf
    fig.savefig(os.path.join('plot_export', 'dist_dep_bar.eps'), bbox_inches='tight')

    plt.show()

def plot_ratios_classes():
    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Open data from pickle file
    collector_X, collector_y, collector_z, clf, X_train, X_test, y_train, y_test, o, z_train, z_test = pickle.load(open('all_0.pickle', 'rb'))

    # Values for raw duration
    class_duration_s = np.array([13096.54 - (189.28 + 1178.98), 189.28, 1178.98])
    class_duration_s = class_duration_s / np.sum(class_duration_s) * 100

    # Number of windows after positive subsampling
    classes, classes_count_positive_subsampling = np.unique(collector_y, return_counts=True)
    classes_count = np.copy(classes_count_positive_subsampling)
    classes_count_positive_subsampling = classes_count_positive_subsampling / np.sum(classes_count_positive_subsampling) * 100

    # Number of windows before positive subsampling
    classes_count[1:] = classes_count[1:] / 2
    classes_count = classes_count / np.sum(classes_count) * 100

    # Assemble list of data
    labels = ['a', 'b', 'c', 'd']
    data = np.array([class_duration_s, classes_count, classes_count_positive_subsampling, [1/3*100,1/3*100,1/3*100]])
    data_df = pd.DataFrame(data, columns = ['-1', '0', '1'])
    data_df['label'] = labels
    data_df = data_df.set_index('label')
    print(1)

    data_df.plot.bar(stacked=True)
    plt.show()

#plot_dist_dep_bars()
plot_ratios_classes()