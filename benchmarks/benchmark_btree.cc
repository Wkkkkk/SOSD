#include "benchmarks/benchmark_btree.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/stx_btree.h"
#include "competitors/AggregationFunctions.hpp"

template <template <typename> typename Searcher>
void benchmark_32_btree(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                        bool pareto) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template Run(STXBTree<uint32_t, 32, typeof(func)>(func));
  if (pareto) {
    benchmark.template Run(STXBTree<uint32_t, 1, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint32_t, 4, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint32_t, 16, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint32_t, 64, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint32_t, 128, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint32_t, 512, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint32_t, 1024, typeof(func)>(func));
    if (benchmark.uses_binary_search()) {
      benchmark.template Run(STXBTree<uint32_t, 4096, typeof(func)>(func));
      benchmark.template Run(STXBTree<uint32_t, 16384, typeof(func)>(func));
      benchmark.template Run(STXBTree<uint32_t, 65536, typeof(func)>(func));
      benchmark.template Run(STXBTree<uint32_t, 262144, typeof(func)>(func));
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_btree(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                        bool pareto) {
  // tuned for Pareto efficiency
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template Run(STXBTree<uint64_t, 32, typeof(func)>(func));
  if (pareto) {
    benchmark.template Run(STXBTree<uint64_t, 1, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint64_t, 4, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint64_t, 16, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint64_t, 64, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint64_t, 128, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint64_t, 512, typeof(func)>(func));
    benchmark.template Run(STXBTree<uint64_t, 1024, typeof(func)>(func));
    if (benchmark.uses_binary_search()) {
      benchmark.template Run(STXBTree<uint64_t, 4096, typeof(func)>(func));
      benchmark.template Run(STXBTree<uint64_t, 16384, typeof(func)>(func));
      benchmark.template Run(STXBTree<uint64_t, 65536, typeof(func)>(func));
      benchmark.template Run(STXBTree<uint64_t, 262144, typeof(func)>(func));
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_32_btree_insert(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                               sosd::Experiment exp) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template UpdatingTest(STXBTree<uint32_t, 32, typeof(func)>(func), exp);
}

template <template <typename> typename Searcher>
void benchmark_64_btree_insert(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                               sosd::Experiment exp) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template UpdatingTest(STXBTree<uint64_t, 32, typeof(func)>(func), exp);
}

INSTANTIATE_TEMPLATES(benchmark_32_btree, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_btree, uint64_t);

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_btree_insert, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_btree_insert, uint64_t);