#!/usr/bin/env bash

if [ ! -f my_model.h5 ]; then
    curl ftp://140.112.107.150/r06725028/my_model.h5 -o my_model.h5
fi
echo "my_model.h5 ok!!!!"

python3.6 hw3_test.py $1 $2 $3 './my_model.h5'