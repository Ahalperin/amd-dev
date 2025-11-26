# Makefile for Vector Addition on AMD GPU using HIP

# Compiler
HIPCC = hipcc

# Compiler flags
CXXFLAGS = -O2 -Wall -std=c++11 -pthread

# Target executables
TARGET = vector_add_test

# Source files
SOURCES = vector_add.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target - build both programs
all: $(TARGET)

# Build 2 GPU version
$(TARGET): $(OBJECTS)
	$(HIPCC) $(CXXFLAGS) -o $@ $^

# Compile source files
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Run
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)

# Print GPU information
info:
	rocminfo

.PHONY: all run run-2gpu run-all clean info help
