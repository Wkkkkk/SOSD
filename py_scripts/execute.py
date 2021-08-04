import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess

DATASET_FILE = "./py_scripts/datasets_under_test.txt"
RESULT_DIR = "./results/"
RESULT_FIGURE_DIR = "./results/figure/"
COLUMNS = dict.fromkeys(['data_set', 'iterations', 'window_size', 'disorder', 'name', 'variant', 'total_ns',
                         'evict_ns', 'insert_ns', 'query_ns', 'model_size', 'build_time', 'op_func'])
NUM_REPEATS = 1
NUM_ITERATIONS = 100000
ALL_WINDOW_SIZES = [100, 200, 500, 1000, 2000, 5000]
ALL_DISORDER = [0, 10, 20, 50, 100, 200, 500]
ALL_AGGREGATION_FUNCTION = ["sum", "max", "geometric-mean", "sample-std-dev", "bloom-filter", "min-count"]
ALL_DATASETS = []


def init():
    print("initialize the environment")
    # load datasets
    with open(DATASET_FILE) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    global ALL_DATASETS
    ALL_DATASETS = [x.strip() for x in content if len(x) != 0]

    create_dir(RESULT_FIGURE_DIR)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_all_results():
    df = pd.DataFrame(columns=COLUMNS)
    for file in os.listdir(RESULT_DIR):
        if not file.endswith(".txt"):
            continue
        if file.endswith("perf.txt"):
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
    df.to_csv(RESULT_DIR + "summary.csv", index=False)
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
    COLUMN_PER_ROW = 4
    ROW_NUM = len(op_funcs) // COLUMN_PER_ROW + 1
    VOID_PLOT_NUM = ROW_NUM * COLUMN_PER_ROW - len(op_funcs)
    fig, axs = plt.subplots(ROW_NUM, COLUMN_PER_ROW, figsize=(15, 8))
    for i, op in enumerate(op_funcs):
        cur_row = i // COLUMN_PER_ROW
        cur_column = i % COLUMN_PER_ROW
        grouped_result = df.groupby(['op_func']).get_group(op)
        name = grouped_result['name']
        index = np.arange(len(name))
        total_ns = grouped_result['total_ns']
        evict_ns = grouped_result['evict_ns']
        insert_ns = grouped_result['insert_ns']
        query_ns = grouped_result['query_ns']
        # other_ns = total_ns - evict_ns - insert_ns - query_ns
        # axs[cur_row][cur_column].bar(index, other_ns, width=0.4, label='other')
        axs[cur_row][cur_column].bar(index, evict_ns, width=0.4, label='evict')
        axs[cur_row][cur_column].bar(index, insert_ns, width=0.4, bottom=evict_ns, label='insert')
        axs[cur_row][cur_column].bar(index, query_ns, width=0.4, bottom=evict_ns + insert_ns, label='query')
        axs[cur_row][cur_column].set_xticks(index)
        axs[cur_row][cur_column].set_xticklabels(name)
        axs[cur_row][cur_column].set_xlabel(op)
        axs[cur_row][cur_column].set_ylabel("time(ns)")
        axs[cur_row][cur_column].legend(loc='upper right', shadow=True)
        # axs[cur_row][cur_column].set_title('Performance for different aggregation functions')

    for i in range(VOID_PLOT_NUM):
        plt.delaxes()
    plt.savefig(RESULT_FIGURE_DIR + 'aggregation.png', dpi=1000)
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
        axs[i].legend(loc='best', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR + 'window_size(performance).png', dpi=500)
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
        plt.legend(loc='best', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR + 'window_size(model_size).png', dpi=1000)
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
        axs[i].legend(loc='best', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR + 'disorder.png', dpi=1000)
    if show_result:
        plt.show()


# draw a plot.
# X - op_func
# Y - operation time
def print_detailed_result_for_uint64_datasets(dataframe, iterations, window_size, disorder, op_func, show_result=False):
    df = dataframe[(dataframe['iterations'] == iterations) & (dataframe['window_size'] == window_size) &
                   (dataframe['disorder'] == disorder) & (dataframe['op_func'] == op_func)]
    all_data_set_names = df.groupby(['data_set']).groups.keys()
    uint64_data_set_names = [name for name in all_data_set_names if str(name).endswith("200M_uint64")]
    df = df[df['data_set'].isin(uint64_data_set_names)]

    data_sets = df.groupby(['data_set']).groups.keys()
    COLUMN_PER_ROW = 4
    ROW_NUM = len(data_sets) // COLUMN_PER_ROW + 1
    VOID_PLOT_NUM = ROW_NUM * COLUMN_PER_ROW - len(data_sets)
    fig, axs = plt.subplots(ROW_NUM, COLUMN_PER_ROW, figsize=(15, 8))
    for i, data_set in enumerate(data_sets):
        cur_row = i // COLUMN_PER_ROW
        cur_column = i % COLUMN_PER_ROW
        grouped_result = df.groupby(['data_set']).get_group(data_set)
        name = grouped_result['name']
        index = np.arange(len(name))
        total_ns = grouped_result['total_ns']
        evict_ns = grouped_result['evict_ns']
        insert_ns = grouped_result['insert_ns']
        query_ns = grouped_result['query_ns']
        # other_ns = total_ns - evict_ns - insert_ns - query_ns
        # axs[cur_row][cur_column].bar(index, other_ns, width=0.4, label='other')
        axs[cur_row][cur_column].bar(index, evict_ns, width=0.4, label='evict')
        axs[cur_row][cur_column].bar(index, insert_ns, width=0.4, bottom=evict_ns, label='insert')
        axs[cur_row][cur_column].bar(index, query_ns, width=0.4, bottom=evict_ns + insert_ns, label='query')
        axs[cur_row][cur_column].set_xticks(index)
        axs[cur_row][cur_column].set_xticklabels(name)
        axs[cur_row][cur_column].set_xlabel(data_set)
        axs[cur_row][cur_column].set_ylabel("time(ns)")
        axs[cur_row][cur_column].legend(loc='upper right', shadow=True)
        # axs[cur_row][cur_column].set_title('Performance for different aggregation functions')

    for i in range(VOID_PLOT_NUM):
        plt.delaxes()
    plt.savefig(RESULT_FIGURE_DIR + 'dataset.png', dpi=1000)
    if show_result:
        plt.show()


# draw a plot.
# X - op_func
# Y - operation time
def print_summary_for_different_datasets(dataframe, iterations, window_size, disorder, op_func, show_result=False):
    df = dataframe[(dataframe['iterations'] == iterations) & (dataframe['window_size'] == window_size) &
                   (dataframe['disorder'] == disorder) & (dataframe['op_func'] == op_func)]
    median = df.groupby('name').median()
    median['name'] = median.index
    print("median performance for each structure: \n", median)
    median.to_csv(RESULT_DIR + "median.csv", index=False)

    name = median['name']
    index = list(name)
    total_ns = median['total_ns']
    evict_ns = median['evict_ns']
    insert_ns = median['insert_ns']
    query_ns = median['query_ns']

    plt.figure()
    plt.title("Overall Performance")
    plt.bar(index, evict_ns, width=0.4, label='evict')
    plt.bar(index, insert_ns, width=0.4, bottom=evict_ns, label='insert')
    plt.bar(index, query_ns, width=0.4, bottom=evict_ns + insert_ns, label='query')
    plt.xticks(index)
    plt.xlabel("structures")
    plt.ylabel("time(ns)")
    plt.legend(loc='upper right', shadow=True)

    plt.savefig(RESULT_FIGURE_DIR + 'overall_performance.png', dpi=1000)
    if show_result:
        plt.show()


# run the program
def run_benchmark(params):
    BENCHMARK = "./build/benchmark"
    if not os.path.isfile(BENCHMARK):
        print("Executable not found")
        exit(1)

    if len(params) < 5:
        print("Not enough parameters for program")
        exit(2)

    num_repeats = str(params[0])
    dataset = params[1]
    num_iterations = str(params[2])
    window_size = str(params[3])
    disorder = str(params[4])
    op_func = params[5]
    data_test = params[6]

    param_str = " -r {0} ./data/{1} ./data/{1}_equality_lookups_1M --query true " \
                "--it {2} --ws {3} --di {4} --af {5} " \
                "--record true".format(num_repeats, dataset, num_iterations, window_size, disorder, op_func)

    if not data_test:
        dataset = "synthetic"
    # save result to
    output_file_name = RESULT_DIR + dataset + "-results_" + \
                       "_".join([num_iterations, window_size, disorder, op_func]) + ".txt"
    if os.path.isfile(output_file_name):
        print("Already have results for ", output_file_name)
        return

    if data_test:
        param_str += " --dtest true"
    print("Executing workload ", BENCHMARK, param_str)
    subprocess.run([BENCHMARK, param_str])


def run_benchmark_with_different_aggregation_function():
    """
    Here we run benchmark with different aggregation function on synthetic data.
    Window size and disorder is both decided automatically (by getting the median)
    """
    print("\nrun_benchmark_with_different_aggregation_function")
    for fn in ALL_AGGREGATION_FUNCTION:
        run_benchmark([NUM_REPEATS, "wiki_ts_200M_uint64",  # <-this parameter doesn't matter
                       NUM_ITERATIONS, ALL_WINDOW_SIZES[len(ALL_WINDOW_SIZES) // 2],  # <-choose a median value
                       ALL_DISORDER[len(ALL_DISORDER) // 2], fn, False])  # <- we use synthetic data to do test


def run_benchmark_with_different_window_size():
    """
    Here we run benchmark with different window size on synthetic data.
    Disorder is decided automatically (by getting the median) and we use sum op.
    """
    print("\nrun_benchmark_with_different_window_size")
    for window_size in ALL_WINDOW_SIZES:
        run_benchmark([NUM_REPEATS, "wiki_ts_200M_uint64",  # <-this parameter doesn't matter
                       NUM_ITERATIONS, window_size,  ALL_DISORDER[len(ALL_DISORDER) // 2],
                       "sum", False])  # <- we use synthetic data to do test


def run_benchmark_with_different_disorder():
    """
    Here we run benchmark with different disorder on synthetic data.
    Window size is decided automatically (by getting the median) and we use sum op.
    """
    print("\nrun_benchmark_with_different_disorder")
    for disorder in ALL_DISORDER:
        run_benchmark([NUM_REPEATS, "wiki_ts_200M_uint64",  # <-this parameter doesn't matter
                       NUM_ITERATIONS, ALL_WINDOW_SIZES[len(ALL_WINDOW_SIZES) // 2],  # <-choose a median value
                       disorder, "sum", False])  # <- we use synthetic data to do test


def run_benchmark_on_real_world_data():
    """
    Here we run benchmark on real-world data.
    Window size and disorder is both decided automatically (by getting the median)
    """
    print("\nrun_benchmark_on_real_world_data")
    for data_set in ALL_DATASETS:
        run_benchmark([NUM_REPEATS, data_set,
                       NUM_ITERATIONS, ALL_WINDOW_SIZES[len(ALL_WINDOW_SIZES) // 2],  # <-choose a median value
                       ALL_DISORDER[len(ALL_DISORDER) // 2], "sum", True])


def main():
    print("Hello World!")
    init()
    # run_benchmark_with_different_aggregation_function()
    # run_benchmark_with_different_window_size()
    # run_benchmark_with_different_disorder()
    # run_benchmark_on_real_world_data()

    print("We have all results, let's draw them into plots")
    show_result = False
    dataframe = read_all_results()
    print_result_for_different_aggregation_functions(dataframe, ["synthetic"], 100000, 1000, 50, show_result)
    # print_result_for_different_window_size(dataframe, ["synthetic"], 100000, 50, "sum", show_result)
    # print_model_size_for_different_window_size(dataframe, ["synthetic"], 100000, 50, "sum", show_result)
    # print_result_for_different_disorder(dataframe, ["synthetic"], 100000, 1000, "sum", show_result)
    # print_detailed_result_for_uint64_datasets(dataframe, 100000, 1000, 50, "sum", show_result)
    # print_summary_for_different_datasets(dataframe, 100000, 1000, 50, "sum", show_result)
    print("Results are saved")


if __name__ == "__main__":
    main()
