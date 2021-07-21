//
// Created by kunwu on 7/20/21.
//

#include "benchmark_fiba.h"

#include "common.h"
#include "competitors/FiBA.hpp"

template <template <typename> typename Searcher>
void benchmark_32_fiba_aggregate(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  visit([&](auto&& f) {
        benchmark.template QueryTest(btree::Aggregate<uint32_t, 2, btree::Kind::finger, typeof(f)>(f), exp);
    }, exp.func);
}

template <template <typename> typename Searcher>
void benchmark_64_fiba_aggregate(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
    visit([&](auto&& f) {
        benchmark.template QueryTest(btree::Aggregate<uint64_t, 2, btree::Kind::finger, typeof(f)>(f), exp);
    }, exp.func);
}

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_fiba_aggregate, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_fiba_aggregate, uint64_t);