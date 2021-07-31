import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
RESULT_DIR = "./results/"
RESULT_FIGURE_DIR = "./results/figure/"
COLUMNS = dict.fromkeys(['data_set', 'iterations', 'window_size', 'disorder', 'name', 'variant', 'total_ns',
                         'evict_ns', 'insert_ns', 'query_ns', 'model_size', 'build_time', 'op_func'])


def init():
    create_dir(RESULT_FIGURE_DIR)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_all_results():
    df = pd.DataFrame(columns=COLUMNS)
    for file in os.listdir(RESULT_DIR):
        if not file.endswith(".txt"):
            continue

        full_path = os.path.join(RESULT_DIR, file)
        file_name = file.split('.')[0]
        data_set_name = file_name.split('-')[0]
        (num_iterations, window_size, disorder, op_func) = file_name.split('_')[-4:]
        print("Processing:", file_name)

        with open(full_path, 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                if not line.startswith('RESULT'):
                    continue
                result_str = line.split(':')[1].strip()
                results = result_str.split(',')

                new_row = COLUMNS
                new_row['data_set'] = data_set_name
                new_row['iterations'] = int(num_iterations)
                new_row['window_size'] = int(window_size)
                new_row['disorder'] = int(disorder)
                new_row['name'] = results[0]
                new_row['variant'] = int(results[1])
                new_row['total_ns'] = float(results[2])
                new_row['evict_ns'] = float(results[3])
                new_row['insert_ns'] = float(results[4])
                new_row['query_ns'] = float(results[5])
                new_row['model_size'] = int(results[6])
                new_row['build_time'] = float(results[7])
                new_row['op_func'] = results[8]

                df = df.append(new_row, ignore_index=True)

    return df


# draw a plot.
# X - op_func
# Y - query time
def print_result_for_different_aggregation_functions(dataframe, data_sets, iterations,
                                                     window_size, disorder, show_result=False):
    # print(dataframe)
    df = dataframe[dataframe['data_set'].isin(data_sets) & (dataframe['iterations'] == iterations) &
                   (dataframe['window_size'] == window_size) & (dataframe['disorder'] == disorder)]

    op_funcs = df.groupby(['op_func']).groups.keys()
    fig, axs = plt.subplots(1, len(op_funcs), figsize=(12, 4))
    for i, op in enumerate(op_funcs):
        grouped_result = df.groupby(['op_func']).get_group(op)
        name = grouped_result['name']
        index = np.arange(len(name))
        total_ns = grouped_result['total_ns']
        evict_ns = grouped_result['evict_ns']
        insert_ns = grouped_result['insert_ns']
        query_ns = grouped_result['query_ns']
        # other_ns = total_ns - evict_ns - insert_ns - query_ns
        # axs[i].bar(index, other_ns, width=0.4, label='other')
        axs[i].bar(index, evict_ns, width=0.4, label='evict')
        axs[i].bar(index, insert_ns, width=0.4, bottom=evict_ns, label='insert')
        axs[i].bar(index, query_ns, width=0.4, bottom=evict_ns+insert_ns, label='query')
        axs[i].set_xticks(index)
        axs[i].set_xticklabels(name)
        axs[i].set_xlabel(op)
        axs[i].set_ylabel("time(ns)")
        axs[i].legend(loc='upper right', shadow=True)
        # axs[i].set_title('Performance for different aggregation functions')

    plt.savefig(RESULT_FIGURE_DIR+'aggregation.png', dpi=1000)
    if show_result:
        plt.show()


# draw a plot.
# X - window size
# Y - operation time
def print_result_for_different_window_size(dataframe, data_sets, iterations, disorder, op_func, show_result=False):
    # print(dataframe)
    df = dataframe[dataframe['data_set'].isin(data_sets) & (dataframe['iterations'] == iterations) &
                   (dataframe['disorder'] == disorder) & (dataframe['op_func'] == op_func)]
    df = df.sort_values(by=['window_size'])

    names = df.groupby(['name']).groups.keys()
    fig, axs = plt.subplots(1, len(names), figsize=(12, 4))
    for i, name in enumerate(names):
        grouped_result = df.groupby(['name']).get_group(name)
        window_size = grouped_result['window_size']
        total_ns = grouped_result['total_ns']
        evict_ns = grouped_result['evict_ns']
        insert_ns = grouped_result['insert_ns']
        query_ns = grouped_result['query_ns']
        model_size = grouped_result['model_size']
        axs[i].plot(window_size, evict_ns, label='evict')
        axs[i].plot(window_size, insert_ns, label='insert')
        axs[i].plot(window_size, query_ns, label='query')
        axs[i].set_xlabel(name)
        axs[i].set_ylabel("time(ns)")
        axs[i].legend(loc='upper right', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR+'window_size(performance).png', dpi=500)
    if show_result:
        plt.show()


# draw a plot.
# X - window size
# Y - model size
def print_model_size_for_different_window_size(dataframe, data_sets, iterations, disorder, op_func, show_result=False):
    # print(dataframe)
    df = dataframe[dataframe['data_set'].isin(data_sets) & (dataframe['iterations'] == iterations) &
                   (dataframe['disorder'] == disorder) & (dataframe['op_func'] == op_func)]
    df = df.sort_values(by=['window_size'])

    names = df.groupby(['name']).groups.keys()
    plt.figure(figsize=(12, 4))
    plt.title("change of model size over window size")

    for name in names:
        grouped_result = df.groupby(['name']).get_group(name)
        window_size = grouped_result['window_size']
        model_size = grouped_result['model_size']
        plt.plot(window_size, model_size, label=name)
        plt.xlabel("window size")
        plt.ylabel("model size(bytes)")
        plt.legend(loc='upper right', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR+'window_size(model_size).png', dpi=1000)
    if show_result:
        plt.show()


# draw a plot.
# X - disorder
# Y - operation time
def print_result_for_different_disorder(dataframe, data_sets, iterations, window_size, op_func, show_result=False):
    # print(dataframe)
    df = dataframe[dataframe['data_set'].isin(data_sets) & (dataframe['iterations'] == iterations) &
                   (dataframe['window_size'] == window_size) & (dataframe['op_func'] == op_func)]
    df = df.sort_values(by=['disorder'])

    names = df.groupby(['name']).groups.keys()
    fig, axs = plt.subplots(1, len(names), figsize=(12, 4))
    for i, name in enumerate(names):
        grouped_result = df.groupby(['name']).get_group(name)
        disorder = grouped_result['disorder']
        total_ns = grouped_result['total_ns']
        evict_ns = grouped_result['evict_ns']
        insert_ns = grouped_result['insert_ns']
        query_ns = grouped_result['query_ns']
        model_size = grouped_result['model_size']
        axs[i].plot(disorder, evict_ns, label='evict')
        axs[i].plot(disorder, insert_ns, label='insert')
        axs[i].plot(disorder, query_ns, label='query')
        axs[i].set_xlabel(name)
        axs[i].set_ylabel("time(ns)")
        axs[i].legend(loc='upper right', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR+'disorder.png', dpi=1000)
    if show_result:
        plt.show()


# draw a plot.
# X - op_func
# Y - query time
def print_result_for_different_datasets(dataframe, iterations, window_size, disorder, op_func, show_result=False):
    # print(dataframe)
    df = dataframe[(dataframe['iterations'] == iterations) & (dataframe['window_size'] == window_size) &
                   (dataframe['disorder'] == disorder) & (dataframe['op_func'] == op_func)]

    data_sets = df.groupby(['data_set']).groups.keys()
    COLUMN_PER_ROW = 3
    ROW_NUM = len(data_sets) // 3 + 1
    VOID_PLOT_NUM = ROW_NUM*COLUMN_PER_ROW - len(data_sets)
    fig, axs = plt.subplots(ROW_NUM, COLUMN_PER_ROW, figsize=(14, 4))
    for i, data_set in enumerate(data_sets):
        grouped_result = df.groupby(['data_set']).get_group(data_set)
        name = grouped_result['name']
        index = np.arange(len(name))
        total_ns = grouped_result['total_ns']
        evict_ns = grouped_result['evict_ns']
        insert_ns = grouped_result['insert_ns']
        query_ns = grouped_result['query_ns']
        # other_ns = total_ns - evict_ns - insert_ns - query_ns
        # axs[i].bar(index, other_ns, width=0.4, label='other')
        axs[i].bar(index, evict_ns, width=0.4, label='evict')
        axs[i].bar(index, insert_ns, width=0.4, bottom=evict_ns, label='insert')
        axs[i].bar(index, query_ns, width=0.4, bottom=evict_ns+insert_ns, label='query')
        axs[i].set_xticks(index)
        axs[i].set_xticklabels(name)
        axs[i].set_xlabel(data_set)
        axs[i].set_ylabel("time(ns)")
        axs[i].legend(loc='upper right', shadow=True)
        # axs[i].set_title('Performance for different aggregation functions')

    for i in range(VOID_PLOT_NUM):
        plt.delaxes()
    plt.savefig(RESULT_FIGURE_DIR+'dataset.png', dpi=1000)
    if show_result:
        plt.show()


init()
dataframe = read_all_results()
print_result_for_different_aggregation_functions(dataframe, ["synthetic"], 100000, 1000, 50)
print_result_for_different_window_size(dataframe, ["synthetic"], 100000, 50, "sum")
print_model_size_for_different_window_size(dataframe, ["synthetic"], 100000, 50, "sum")
print_result_for_different_disorder(dataframe, ["synthetic"], 100000, 1000, "sum")
print_result_for_different_datasets(dataframe, 100000, 1000, 50, "sum")
print("Results are saved to:", RESULT_FIGURE_DIR)

