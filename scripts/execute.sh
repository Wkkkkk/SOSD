#! /usr/bin/env bash

echo "Executing benchmark and saving results..."
num_repeats=1;

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_data_benchmark() {
    num_iterations=$3
    window_size=$4
    disorder=$5
    fn=$6

    RESULTS=./results/$1-results_$3_$4_$5_$6.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK -r $2 ./data/$1 ./data/$1_equality_lookups_1M --query true --it $num_iterations --ws $window_size --di $disorder --af $fn --dtest true --record true | tee $RESULTS
    fi
}

function do_synthetic_benchmark() {
    num_iterations=$3
    window_size=$4
    disorder=$5
    fn=$6

    RESULTS=./results/synthetic-results_$3_$4_$5_$6.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for synthetic data"
    else
        echo "Executing workload $1"
        $BENCHMARK -r $2 ./data/$1 ./data/$1_equality_lookups_1M --query true --it $num_iterations --ws $window_size --di $disorder --af $fn --record true | tee $RESULTS
    fi
}

function do_synthetic_benchmark_perf() {
    num_iterations=$3
    window_size=$4
    disorder=$5
    fn=$6

    RESULTS=./results/synthetic-results_$3_$4_$5_$6_perf.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for synthetic data"
    else
        echo "Executing workload $1"
        $BENCHMARK -r $2 ./data/$1 ./data/$1_equality_lookups_1M --query true --it $num_iterations --ws $window_size --di $disorder --af $fn --record true --perf | tee $RESULTS
    fi
}

mkdir -p ./results

synthetic_data=wiki_ts_200M_uint64;
# perf
do_synthetic_benchmark_perf $synthetic_data $num_repeats 100000 1000 50 sum

# aggregation functions
for fn in sum max # mean
do
  do_synthetic_benchmark $synthetic_data $num_repeats 100000 1000 50 $fn
done

# window size
for window_size in 100 200 500 1000 2000 5000
do
  do_synthetic_benchmark $synthetic_data $num_repeats 100000 $window_size 50 sum
done

# disorder
for disorder in 0 10 20 50 100 200 500
do
  do_synthetic_benchmark $synthetic_data $num_repeats 100000 1000 $disorder sum
done

do_synthetic_benchmark $synthetic_data $num_repeats 100000 1000 50 sum true

# real-world data
for dataset in $(cat scripts/datasets_under_test.txt);
do
  do_data_benchmark $dataset $num_repeats 100000 1000 50 sum
done
