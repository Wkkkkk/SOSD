#! /usr/bin/env bash

echo "Executing benchmark and saving results..."
num_iterations=3;
while getopts n:c arg; do
    case $arg in
        c) do_csv=true;;
        n) num_iterations=${OPTARG};;
    esac
done

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results/synthetic_results.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for synthetic data"
    else
        echo "Executing workload"
        $BENCHMARK -r $2 ./data/$1 ./data/$1_equality_lookups_1M --query true --it 100000 --ws 1000 --di 100 | tee $RESULTS
    fi
}

mkdir -p ./results

do_benchmark books_200M_uint64 $num_iterations