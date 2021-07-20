//
// Created by kunwu on 7/20/21.
//

#include "benchmark_fiba.h"

#include "common.h"
#include "competitors/FiBA.hpp"
#include "competitors/AggregationFunctions.hpp"

template <template <typename> typename Searcher>
void benchmark_32_fiba_query(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template QueryTest( btree::Aggregate<uint32_t, 2, btree::Kind::finger, typeof(func)>(func), exp);
}

template <template <typename> typename Searcher>
void benchmark_64_fiba_query(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template QueryTest( btree::Aggregate<uint64_t, 2, btree::Kind::finger, typeof(func)>(func), exp);
}

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_fiba_query, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_fiba_query, uint64_t);