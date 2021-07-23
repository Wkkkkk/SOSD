#! /usr/bin/env bash

FOLDER=results

function CreateTable() {
  ParseFile books_200M_uint64_results_perf.txt 1
  ParseFile fb_200M_uint64_results_perf.txt
  ParseFile osm_cellids_200M_uint64_results_perf.txt
  ParseFile wiki_ts_200M_uint64_results_perf.txt
}

function ParseFile() {
  file_name=$1
  is_first_file=$2

  cat $FOLDER/$file_name | grep -v Repeating | grep -v read | grep -v "data contains duplicates" | \
  awk -v file_name=$file_name -v is_first_file=$is_first_file -F'[, ]' '

  BEGIN {
    column_names[0] = "index"
    column_names[1] = "variant"
    column_names[2] = "time(s)"
    column_names[3] = "cycles"
    column_names[4] = "instructions"
    column_names[5] = "L1-misses"
    column_names[6] = "LLC-misses"
    column_names[7] = "branch-misses"
    column_names[8] = "task-clock"
    column_names[9] = "scale"
    column_names[10] = "IPC"
    column_names[11] = "CPUs"
    column_names[12] = "GHz"

    idx_names[0] = "BTree"
    idx_names[1] = "FiBA"
    idx_names[2] = "ALEX"

    if(is_first_file) {
      for(i=0; i<length(column_names); i++) {
        printf("%15s |", column_names[i])
      }
      print ""
      print " --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:| --------------:|"
    }

    # Translate file name into nice date set name to be displayed in the table
    data_set = "unknown"
    if(file_name == "books_200M_uint64_results_perf.txt") data_set = "amzn64"
    if(file_name == "fb_200M_uint64_results_perf.txt") data_set = "face64"
    if(file_name == "osm_cellids_200M_uint64_results_perf.txt") data_set = "osmc64"
    if(file_name == "wiki_ts_200M_uint64_results_perf.txt") data_set = "wiki64"

    if(data_set == "unknown") {
      print "Unknown data set, please extend the script."
      exit
    }
  }

  /RESULT/ {
    # Gather all measurements
    for(i=0; i<10; i++) {
      j = i + 2
      printf("%15s |", $j)
    }
  }

  END {
    printf("| %-13s |", data_set)
#    for(i=0; i<length(idx_names); i++) {
#      res = idx_result[idx_names[i]]
#      printf("%10s |", length(res) == 0 ? "n/a" : res)
#    }
    print ""
  }
  '
}

CreateTable
