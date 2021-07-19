#pragma once
#include "benchmark.h"

template <template <typename> typename Searcher>
void benchmark_32_alex(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                       bool pareto);

template <template <typename> typename Searcher>
void benchmark_64_alex(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                       bool pareto);

template <template <typename> typename Searcher>
void benchmark_32_alex_insert(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                              sosd::Experiment exp);

template <template <typename> typename Searcher>
void benchmark_64_alex_insert(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                       sosd::Experiment exp);