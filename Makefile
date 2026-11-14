CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm

SRC = src/main.c src/helpers.c
BIN = transformer

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC) include/helpers.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
