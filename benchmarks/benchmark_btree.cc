#include "benchmarks/benchmark_btree.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/stx_btree.h"

template <template <typename> typename Searcher>
void benchmark_32_btree(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                        bool pareto) {
  benchmark.template Run<STXBTree<uint32_t, 32>>();
  if (pareto) {
    benchmark.template Run<STXBTree<uint32_t, 1>>();
    benchmark.template Run<STXBTree<uint32_t, 4>>();
    benchmark.template Run<STXBTree<uint32_t, 16>>();
    benchmark.template Run<STXBTree<uint32_t, 64>>();
    benchmark.template Run<STXBTree<uint32_t, 128>>();
    benchmark.template Run<STXBTree<uint32_t, 512>>();
    benchmark.template Run<STXBTree<uint32_t, 1024>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<STXBTree<uint32_t, 4096>>();
      benchmark.template Run<STXBTree<uint32_t, 16384>>();
      benchmark.template Run<STXBTree<uint32_t, 65536>>();
      benchmark.template Run<STXBTree<uint32_t, 262144>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_btree(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                        bool pareto) {
  // tuned for Pareto efficiency
  benchmark.template Run<STXBTree<uint64_t, 32>>();
  if (pareto) {
    benchmark.template Run<STXBTree<uint64_t, 1>>();
    benchmark.template Run<STXBTree<uint64_t, 4>>();
    benchmark.template Run<STXBTree<uint64_t, 16>>();
    benchmark.template Run<STXBTree<uint64_t, 64>>();
    benchmark.template Run<STXBTree<uint64_t, 128>>();
    benchmark.template Run<STXBTree<uint64_t, 512>>();
    benchmark.template Run<STXBTree<uint64_t, 1024>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<STXBTree<uint64_t, 4096>>();
      benchmark.template Run<STXBTree<uint64_t, 16384>>();
      benchmark.template Run<STXBTree<uint64_t, 65536>>();
      benchmark.template Run<STXBTree<uint64_t, 262144>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_32_btree_insert(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                               sosd::Experiment exp) {
  benchmark.template UpdatingTest<STXBTree<uint32_t, 32>>(exp);
}

template <template <typename> typename Searcher>
void benchmark_64_btree_insert(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                               sosd::Experiment exp) {
  benchmark.template UpdatingTest<STXBTree<uint64_t, 32>>(exp);
}

INSTANTIATE_TEMPLATES(benchmark_32_btree, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_btree, uint64_t);

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_btree_insert, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_btree_insert, uint64_t);