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

    RESULTS=./results/$1-results_$3_$4.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK -r $2 ./data/$1 ./data/$1_equality_lookups_1M --query true --it $num_iterations --ws $window_size --di 100 --dtest true --record true | tee $RESULTS
    fi
}

function do_synthetic_benchmark() {
    num_iterations=$3
    window_size=$4

    RESULTS=./results/synthetic-results_$3_$4.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for synthetic data"
    else
        echo "Executing workload $1"
        $BENCHMARK -r $2 ./data/$1 ./data/$1_equality_lookups_1M --query true $num_iterations --ws $window_size --di 100 --record true | tee $RESULTS
    fi
}

mkdir -p ./results

for dataset in $(cat scripts/datasets_under_test.txt);
do
  for window_size in 100 200 500 1000 2000 5000
  do
    do_synthetic_benchmark $dataset $num_repeats 100000 $window_size
    do_data_benchmark $dataset $num_repeats 100000 $window_size
  done
done
