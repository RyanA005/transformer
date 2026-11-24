CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm

SRC = src/main.c src/helpers.c src/model.c src/normalization.c src/train.c
BIN = transformer

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC) include/helpers.h include/model.h include/train.h include/tokenizer.h include/hashmap.h include/normalization.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
