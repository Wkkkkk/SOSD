#include "benchmarks/benchmark_rbtree.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/stdmap.h"

template <template <typename> typename Searcher>
void benchmark_32_rbtree_aggregate(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                                  sosd::Experiment exp) {
  visit([&](auto&& f) {
    benchmark.template QueryTest(RBTree<uint32_t, typeof(f)>(f), exp);
  }, exp.func);
}

template <template <typename> typename Searcher>
void benchmark_64_rbtree_aggregate(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                                  sosd::Experiment exp) {
  visit([&](auto&& f) {
    benchmark.template QueryTest(RBTree<uint64_t, typeof(f)>(f), exp);
  }, exp.func);
}

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_rbtree_aggregate, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_rbtree_aggregate, uint64_t);