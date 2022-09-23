import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox
import seaborn as sns
import sklearn.metrics as skm
import itertools
from sklearn.metrics import *
from palettable.cartocolors.sequential import agGrnYl_7_r
from palettable.cartocolors.diverging import Temps_7_r
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_theme()



def f1_score_for_dist(df_in, distance):
    df_look = df_in[(df_in.dist <= distance) | (df_in.y_true == -1)]
    return skm.f1_score(df_look.y_true.values, df_look.y_pred.values, average='micro')


def plot_dist_dep_plots_detection():
    # Set working directory
    working_dir = 'H:/bagfiles_unpack/'

    # Get detected encounters
    encounter_db_manual = pd.read_feather(os.path.join(working_dir, 'encounter_db_v2_backup_after_manual.feather'))
    encounter_db_manual = encounter_db_manual.sort_values("begin")
    encounter_db_manual = encounter_db_manual.drop_duplicates(subset=["begin", "end"])

    # Get detected encounters
    encounter_db_auto = pd.read_feather(os.path.join(working_dir, 'encounter_db_v2_backup_pre_manual.feather'))
    encounter_db_auto = encounter_db_auto.sort_values("begin")
    encounter_db_auto = encounter_db_auto.drop_duplicates(subset=["begin", "end"])

    # Create Dataframe for distnace depending stats
    steering_handle_overhang = 19

    dist_dep_metric = pd.DataFrame({'y_true': encounter_db_manual.direction.values, 'y_pred': encounter_db_auto.direction.values,
                  'dist': encounter_db_manual.distance.values - steering_handle_overhang})


    # Create dists TP, FP and FN for Classes 0 and 1
    TP0 = dist_dep_metric[(dist_dep_metric.y_true==0) & (dist_dep_metric.y_pred==0)].dist
    FP0 = dist_dep_metric[((dist_dep_metric.y_true==-1) | (dist_dep_metric.y_true==1)) & (dist_dep_metric.y_pred==0)].dist
    FN0 = dist_dep_metric[((dist_dep_metric.y_pred==-1) | (dist_dep_metric.y_pred==1)) & (dist_dep_metric.y_true==0)].dist

    TP1 = dist_dep_metric[(dist_dep_metric.y_true==1) & (dist_dep_metric.y_pred==1)].dist
    FP1 = dist_dep_metric[((dist_dep_metric.y_true==-1) | (dist_dep_metric.y_true==0)) & (dist_dep_metric.y_pred==1)].dist
    FN1 = dist_dep_metric[((dist_dep_metric.y_pred==-1) | (dist_dep_metric.y_pred==0)) & (dist_dep_metric.y_true==1)].dist

    hist_data = [[TP0, FP0, FN0], [TP1, FP1, FN1]]

    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Create figure
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 5))

    # Create both plots
    for ax, data, invert, text in zip(axes.ravel(), hist_data, [False, True], ['für Entgegenkommen', 'für Überholen']):
        # Plot histogram
        ax.hist(data, bins=24, range=(0,300), stacked=True)
        ax.set_ylim(bottom=0, top=35)

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
    fig.savefig(os.path.join('plot_export', 'dist_dep_bar_res1.eps'), bbox_inches='tight')

    plt.show()



def plot_dist_dep_bars():
    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Open data from pickle file
    collector_X, collector_y, collector_z, clf, X_train, X_test, y_train, y_test, o, z_train, z_test, _, _= pickle.load(open('all_5_0.5_diff_small_no_shuffle.pickle', 'rb'))

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

    # Number of symples in y_train and y_test
    test_size = 0.2
    y_train_dist = np.array([1/3,1/3,1/3]) * (1 - test_size) * 100
    y_test_dist = np.unique(y_test, return_counts=True)[1] / np.sum(np.unique(y_test, return_counts=True)[1]) * test_size * 100

    # Assemble list of data
    labels = ['a)', 'b)', 'c)', 'd)']
    data = np.array([class_duration_s, classes_count, classes_count_positive_subsampling, y_train_dist])
    data_df = pd.DataFrame(data, columns = ['-1', '0', '1'])
    data_df['label'] = labels
    data_df = data_df.set_index('label')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot first chunk
    data_df.plot.bar(stacked=True, ax=ax)

    # Add y_test to col 4 (i know, that this is ugly)
    ax.bar(3, y_test_dist[0], 0.5, bottom=(1 - test_size) * 100)
    ax.bar(3, y_test_dist[1], 0.5, bottom=(1 - test_size) * 100 + y_test_dist[0])
    ax.bar(3, y_test_dist[2], 0.5, bottom=(1 - test_size) * 100 + y_test_dist[0] + y_test_dist[1])

    # Add horizontal line to seperate test and train
    ax.hlines((1 - test_size) * 100, 2.6, 3.4, colors=sns.color_palette()[3])
    plt.text(2.65, (1 - test_size) * 100 / 2, 'Train', rotation='vertical', fontproperties=font_property, verticalalignment='center')
    plt.text(2.65, (1 - test_size) * 100 + test_size * 100 / 2, 'Test', rotation='vertical', fontproperties=font_property, verticalalignment='center')

    # Add class legend
    plt.legend(['Split: Train/Test', 'kein Fahrzeug', 'entgegenkommendes Fahrzeug', 'überholendes Fahrzeug'], prop=font_property)

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)

    # Rotate x ticks back to normal
    plt.xticks(rotation=0)

    # Add labels to axes
    plt.xlabel('Verarbeitungsschritt', fontproperties=font_property)
    plt.ylabel('Klassenanteil [%]', fontproperties=font_property)

    # Export as pdf
    fig.savefig(os.path.join('plot_export', 'class_imbalance_evolution.eps'), bbox_inches='tight')

    plt.show()

def plot_hyper_para_grid():
    # Value lists
    stride_list = [1, 0.75, 0.5, 0.25]
    width_list = [5, 7.5, 10, 12.5, 15]

    matrix = np.empty((4,5))

    for (stride, width), (x, y) in zip(list(itertools.product(stride_list, width_list)) , list(itertools.product(range(4), range(5)))):
        file_name = f"all_{width}_{stride}.pickle"

        _, _, _, clf, _, _, _, y_test, o, _, _, sc, _ = pickle.load(open(file_name, 'rb'))

        #print(x,y)
        #print(accuracy_score(y_test, o))
        #confus = confusion_matrix(y_test, o)
        #matrix[x,y] = np.mean((confus.diagonal() / confus.sum(axis=1))[1:])


        matrix[x,y] = accuracy_score(y_test, o)

    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(matrix*100, cmap=agGrnYl_7_r.mpl_colormap,  annot=True, fmt='.1f', ax=ax, yticklabels=[float(val) for val in stride_list], xticklabels=[float(val) for val in width_list], vmin=90, vmax=100, annot_kws={'fontproperties': font_property})

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)
    for label in ax.figure.axes[-1].get_yticklabels():
        label.set_fontproperties(font_property)

    #ax.set_xticks(np.arange(5), labels=[5, 7.5, 10, 12.5, 15])
    #ax.set_yticks(np.arange(4), labels=[1, 0.75, 0.5, 0.25])

    # Add labels to axes
    plt.xlabel('Fenster-Breite [s]', fontproperties=font_property)
    plt.ylabel('Stride-Distanz [s]', fontproperties=font_property)
    ax.figure.axes[-1].set_ylabel('Overall Accuracy [%]', fontproperties=font_property)

    # Export as pdf
    fig.savefig('hyperparameter_matrix.eps', bbox_inches='tight')

    plt.show()


def plot_feature_quali():
    # Value lists
    working_dir = 'feature_set'
    file_list = os.listdir(working_dir)

    results_list = []
    name_list = ['Referenz', 'nur Barometerdaten', 'nur Magnetometerdaten', 'mit Diff-Operator', 'mit Diff-Operator (reduzierte Features)', 'Normierung Magnetometer', 'nur x- und y-Achse Magnetometer']
    name_list = list(np.arange(1,8))
    #name_list = ['Referenz', 'nur Barometerdaten','nur Barometerdaten', 'nur Barometerdaten', 'nur Magnetometerdaten',  'nur Magnetometerdaten', 'nur Magnetometerdaten', 'mit Diff-Operator', 'mit Diff-Operator (reduzierte Features)', 'Normierung Magnetometer', 'nur x- und y-Achse Magnetometer']

    #feature_list = ['OA', '-1_pr', '-1_re', '-1_f1', '0_pr', '0_re', '0_f1', '1_pr', '1_re', '1_f1']
    feature_list = ['OA', '-1_f1', '0_f1', '1_f1']
    feature_list = ['Klasse: Kein.', 'Klasse: Entg.', 'Klasse: Über.', 'Gesamt']

    importance_list = []

    for file_name in file_list:
        file_path = os.path.join(working_dir, file_name)

        _, _, _, clf, _, _, _, y_test, o, _, _, sc, _ = pickle.load(open(file_path, 'rb'))

        #name_list.append([file_name])
        results_list.append(list(f1_score(y_test, o, average=None)) + [accuracy_score(y_test, o)])

        #results_list.append([accuracy_score(y_test, o)] + list(np.array(precision_recall_fscore_support(y_test, o))[:-1,:].ravel(order='F')))

        print(file_name, min(clf.feature_importances_), max(clf.feature_importances_), np.std(clf.feature_importances_))

        importance_list.append(clf.feature_importances_)

    print(1)

    data_array = np.array(results_list)


    # Set up font
    font_property = fm.FontProperties(fname='cmunrm2.ttf')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    #fig.subplots_adjust(left=0.336, right=0.98)

    #sns.heatmap(matrix*100, cmap=agGrnYl_7_r.mpl_colormap,  annot=True, fmt='.1f', ax=ax, yticklabels=[float(val) for val in stride_list], xticklabels=[float(val) for val in width_list], vmin=90, vmax=100, annot_kws={'fontproperties': font_property})
    #sns.heatmap((data_array - data_array[0,:])*100, cmap=Temps_7_r.mpl_colormap,  annot=True, fmt='.1f', ax=ax, yticklabels=name_list, xticklabels=feature_list, vmin=-10, vmax=10, annot_kws={'fontproperties': font_property})
    sns.heatmap((data_array) * 100, cmap=agGrnYl_7_r.mpl_colormap, annot=True, fmt='.1f', ax=ax,
                yticklabels=name_list, xticklabels=feature_list, vmin=75, vmax=100,
                annot_kws={'fontproperties': font_property})

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)
    for label in ax.figure.axes[-1].get_yticklabels():
        label.set_fontproperties(font_property)

    #ax.set_xticks(np.arange(5), labels=[5, 7.5, 10, 12.5, 15])
    #ax.set_yticks(np.arange(4), labels=[1, 0.75, 0.5, 0.25])

    # Add labels to axes
    plt.xlabel('Genauigkeitsmetrik', fontproperties=font_property)
    plt.ylabel('Feature-Auswahl', fontproperties=font_property)
    ax.figure.axes[-1].set_ylabel('Unterschied zur Referenz [%P]', fontproperties=font_property)

    # Export as pdf
    fig.savefig('feature_matrix.eps', bbox_inches='tight')

    plt.show()


def plot_wifi_timing():
    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Load Dataset IMU
    imu_data = pd.read_feather('inertial_measurement_unit_0.feather')

    time_a = "12:03:00"
    time_b = "12:03:01"

    imu_sens_time = pd.to_datetime(imu_data.timestamp_sensor.values, unit='s')
    imu_sens_time_idx = imu_sens_time.indexer_between_time(time_a, time_b)
    imu_bag_time = pd.to_datetime(imu_data.timestamp_bagfile.values + max(imu_data.timestamp_sensor.values - imu_data.timestamp_bagfile.values), unit='s')
    imu_bag_time_idx = imu_bag_time.indexer_between_time(time_a, time_b)

    common_idx = list(set(imu_bag_time_idx).intersection(imu_sens_time_idx))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    fig.subplots_adjust(left=0.22)

    ax.eventplot(imu_sens_time[imu_sens_time_idx], linelengths=0.8, linewidths=1.5, lineoffsets=0, colors=sns.color_palette()[1])
    ax.eventplot(imu_bag_time[imu_bag_time_idx], linelengths=0.8, linewidths=1.5, lineoffsets=1)

    for index in common_idx:
        plt.plot((imu_sens_time[index], imu_bag_time[index]), (0.4, 0.6), 'k--', lw=0.7)

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)

    ax.set_yticks([0,1])
    ax.set_yticklabels(['Zeitstempel Messung', 'Zeitstempel Aufzeichnung'])
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)

    plt.xlabel('Zeit [m:s:ms]', fontproperties=font_property)

    plt.show()

    # Export as pdf
    fig.savefig('timing_wifi.eps', bbox_inches='tight')


def plot_car_mag():
    # Set up font
    font_property = fm.FontProperties(fname='cmunrm2.ttf')

    mag=True

    car_t1 = pd.to_datetime(1659938730.142288628, unit='s')
    car_t2 = pd.to_datetime(1659938730.670628146, unit='s')

    # Load Dataset IMU
    if mag:
        mag_data = pd.read_feather('magnetic_field_sensor_0.feather')
        mag_data['timestamp_bagfile'] = pd.to_datetime(mag_data.timestamp_sensor.values - max(mag_data.timestamp_sensor.values - mag_data.timestamp_bagfile.values), unit='s')
        mag_data.drop('timestamp_sensor', inplace=True, axis=1)
        mag_data.set_index('timestamp_bagfile', inplace=True)

    else:
        pre_data = pd.read_feather('pressure_sensor_0.feather')
        pre_data['timestamp_bagfile'] = pd.to_datetime(pre_data.timestamp_sensor.values - max(pre_data.timestamp_sensor.values - pre_data.timestamp_bagfile.values), unit='s')
        pre_data.drop('timestamp_sensor', inplace=True, axis=1)
        pre_data.set_index('timestamp_bagfile', inplace=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(bottom=0.15)

    plt.xlabel('Zeit [h:m:s]', fontproperties=font_property)

    if mag:
        ax.plot(mag_data)
        ax.legend(['x','y','z'], prop=font_property)
        plt.ylabel('Magnetischer Fluss [T]', fontproperties=font_property)
        ax.yaxis.offsetText.set_fontproperties(font_property)

    else:
        ax.plot(pre_data/100)
        plt.ylabel('Luftdruck [hPa]', fontproperties=font_property)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)

    plt.axvspan(car_t1, car_t2, color=sns.color_palette()[3], alpha=0.3)

    plt.show()

    if mag:
        exp_str = 'mag'
    else:
        exp_str = 'pre'

    # Export as pdf
    fig.savefig(f'example_{exp_str}.svg', bbox_inches='tight')


def list_balancing():
    # Value lists
    working_dir = 'balnce'
    stride_list = [1, 0.75, 0.5, 0.25]
    width_list = [5, 7.5, 10, 12.5, 15]

    matrix_bal = np.empty((4,5))
    matrix_samp = np.empty((4, 5))

    for (stride, width), (x, y) in zip(list(itertools.product(stride_list, width_list)) , list(itertools.product(range(4), range(5)))):
        file_name = f"all_{width}_{stride}.pickle"

        _, _, _, clf, X_train, X_test, y_train, y_test, o, z_train, z_test, sc, _ = pickle.load(open(os.path.join(working_dir, file_name), 'rb'))

        counts = np.unique(y_train, return_counts=True)[1]

        print(stride, width, counts)

        print(sum(counts[1:]) / sum(counts))

        matrix_bal[x, y] = sum(counts[1:]) / sum(counts)
        matrix_samp[x, y] = len(y_train)

    # Build mean along axis 1
    samp = np.average(matrix_samp, axis=1)

    # Build mean along axis 0
    bal = np.average(matrix_bal, axis=0)

    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(stride_list, samp, '+-')

    # Set Computer Moedern as tick font
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_property)

    #ax.set_xticks(np.arange(5), labels=[5, 7.5, 10, 12.5, 15])
    #ax.set_yticks(np.arange(4), labels=[1, 0.75, 0.5, 0.25])

    # Add labels to axes
    ax1.set_xlabel('Stride-Distanz [s]', fontproperties=font_property)
    ax1.set_ylabel('Gesamtanzahl Samples', fontproperties=font_property)

    ax2.plot(width_list, bal, '+-')

    # Set Computer Moedern as tick font
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_property)

    #ax.set_xticks(np.arange(5), labels=[5, 7.5, 10, 12.5, 15])
    #ax.set_yticks(np.arange(4), labels=[1, 0.75, 0.5, 0.25])

    # Add labels to axes
    ax2.set_xlabel('Fenster-Breite [s]', fontproperties=font_property)
    ax2.set_ylabel('Anteil der Klassen "Entg." und "Über." an allen Samples [%]', fontproperties=font_property)

    fig.tight_layout(pad=1.0)

    # Export as pdf
    fig.savefig('stride_n_width.eps', bbox_inches='tight')

    plt.show()


plot_dist_dep_bars()
#plot_ratios_classes()
#plot_hyper_para_grid()
#plot_wifi_timing()
#plot_car_mag()
#plot_feature_quali()
#list_balancing()
#plot_dist_dep_plots_detection()