.PHONY: all test clean

CC = gcc
CFLAGS = -lm -Wall -Wextra -std=c11 -g

tests/test_bitset: tests/test_bitset.c symgen/bitset.h
	$(CC) tests/test_bitset.c -Isymgen/ $(CFLAGS) -o tests/test_bitset

test: tests/test_bitset
	./tests/test_bitset

