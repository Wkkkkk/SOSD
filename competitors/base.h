#pragma once

#include "../util.h"

class Competitor {
 public:
  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  int variant() const { return 0; }

  std::string op_func() const { return ""; }
};

class UpdatableCompetitor : public Competitor {
 public:
  bool updatable() {
    return true;
  }
};