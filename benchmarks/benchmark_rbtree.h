#pragma once
#include "benchmark.h"

template <template <typename> typename Searcher>
void benchmark_32_rbtree_aggregate(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                                  sosd::Experiment exp);

template <template <typename> typename Searcher>
void benchmark_64_rbtree_aggregate(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                                  sosd::Experiment exp);