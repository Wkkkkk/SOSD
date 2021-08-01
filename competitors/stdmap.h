#pragma once

#include <map>

#include "base.h"

template <class KeyType, typename BinOpFunc>
class RBTree : public UpdatableCompetitor {
 public:
  typedef typename BinOpFunc::In inT;
  typedef typename BinOpFunc::Partial aggT;
  typedef typename BinOpFunc::Out outT;

  RBTree(BinOpFunc func)
      : func_(func) {}

  void insert(const KeyType& key, const uint64_t& value) {
    btree_.insert(std::pair(key, value));
    data_size_ += 1;
  }

  void evict() {
    btree_.erase(btree_.begin());
    if (data_size_ > 0) { data_size_ -= 1; }
  }

  outT query() {
    aggT result = BinOpFunc::identity;
    for (auto const& [key, val] : btree_)
    {
      result = func_.combine(result, val);
    }
    return func_.lower(result);
  }

  KeyType oldest() const {
    if(btree_.empty()) return KeyType();

    return btree_.begin()->first;
  }

  KeyType youngest() const {
    if(btree_.empty()) return KeyType();

    return btree_.end()->first;
  }

  std::string name() const { return "RBTree"; }

  std::size_t size() const {
    return btree_.size();
  }

  std::size_t data_size() const { return data_size_; }

  int variant() const { return 1; }

  std::string op_func() const { return func_.name(); }

 private:
  uint64_t data_size_ = 0;
  std::map<KeyType, uint64_t> btree_;
  BinOpFunc func_;
};
