#include "benchmarks/benchmark_alex.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/alex.h"

template <template <typename> typename Searcher>
void benchmark_32_alex(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                       bool pareto) {
  Sum<uint64_t> func;
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
  Sum<uint64_t> func;
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
void benchmark_32_alex_aggregate(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  visit([&](auto&& f) {
    benchmark.template QueryTest(Alex<uint32_t, 1, typeof(f)>(f), exp);
  }, exp.func);
}

template <template <typename> typename Searcher>
void benchmark_64_alex_aggregate(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                              sosd::Experiment exp) {
  visit([&](auto&& f) {
    benchmark.template QueryTest(Alex<uint64_t, 1, typeof(f)>(f), exp);
  }, exp.func);
}

INSTANTIATE_TEMPLATES(benchmark_32_alex, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_alex, uint64_t);

INSTANTIATE_TEMPLATES_INSERT(benchmark_32_alex_aggregate, uint32_t);
INSTANTIATE_TEMPLATES_INSERT(benchmark_64_alex_aggregate, uint64_t);