#/bin/bash
sort -R data/cut.csv > data/sort.csv
# 512 1024 2048 3072 4096 5120 6144 7168 7478
head -n 8812 data/sort.csv > data/train.csv
tail -n 1024 data/sort.csv > data/tmp.csv
head -n 512 data/tmp.csv > data/dev.csv
tail -n 512 data/tmp.csv > data/test.csv
rm data/tmp.csv
rm data/sort.csv
