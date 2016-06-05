#!/bin/bash

repeat=10

prog='
test_dTNN_chayoneko_magma
test_dTNN_cublas
'
./test_dTNN_chayoneko_magma141 3072 5120 64 4 0

for p in $prog; do 
for (( n = 1024; n <= 10240; n += 512 )); do
for (( r = 0; r < repeat; ++r )); do
	./$p $n 5120 64 4 0
done
done
done

prog='
test_sTNN_chayoneko_magma
test_sTNN_cublas
'

for p in $prog; do 
for (( n = 1536; n <= 12288; n += 768 )); do
for (( r = 0; r < repeat; ++r )); do
	./$p $n 7680 64 4 0
done
done
done
