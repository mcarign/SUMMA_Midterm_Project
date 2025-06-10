CC = mpicc
CFLAGS = -Wall -O3
INCLUDES = -I./include
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = .

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = main

all: directories $(TARGET)

directories:
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

run:
	mpirun -np 16 ./main -m 32768 -n 32768 -k 128 -s b -v -p

.PHONY: all clean run directories