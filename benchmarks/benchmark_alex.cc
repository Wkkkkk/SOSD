#include "benchmarks/benchmark_alex.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/alex.h"
#include "competitors/AggregationFunctions.hpp"

template <template <typename> typename Searcher>
void benchmark_32_alex(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                       bool pareto) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template Run(Alex<uint32_t, 1, typeof(func)>(func));
  if (pareto) {
    benchmark.template Run(Alex<uint32_t, 2, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 4, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 8, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 16, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 32, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 64, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 128, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 256, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 512, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 1024, typeof(func)>(func));
    benchmark.template Run(Alex<uint32_t, 2048, typeof(func)>(func));
    if (benchmark.uses_binary_search()) {
      benchmark.template Run(Alex<uint32_t, 4096, typeof(func)>(func));
      benchmark.template Run(Alex<uint32_t, 8192, typeof(func)>(func));
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_alex(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                       bool pareto) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template Run(Alex<uint64_t, 1, typeof(func)>(func));
  if (pareto) {
    benchmark.template Run(Alex<uint64_t, 2, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 4, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 8, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 16, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 32, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 64, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 128, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 256, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 512, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 1024, typeof(func)>(func));
    benchmark.template Run(Alex<uint64_t, 2048, typeof(func)>(func));
    if (benchmark.uses_binary_search()) {
      benchmark.template Run(Alex<uint64_t, 4096, typeof(func)>(func));
      benchmark.template Run(Alex<uint64_t, 8192, typeof(func)>(func));
    }
  }
}
template <template <typename> typename Searcher>
void benchmark_32_alex_insert(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template UpdatingTest(Alex<uint32_t, 1, typeof(func)>(func), exp);
}

template <template <typename> typename Searcher>
void benchmark_64_alex_insert(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  Sum<uint64_t, uint64_t, uint64_t> func;
  benchmark.template UpdatingTest(Alex<uint64_t, 1, typeof(func)>(func), exp);
}

INSTANTIATE_TEMPLATES(benchmark_32_alex, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_alex, uint64_t);

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_alex_insert, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_alex_insert, uint64_t);