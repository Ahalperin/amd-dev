# Makefile for Vector Addition on AMD GPU using HIP

# Compiler
HIPCC = hipcc

# Compiler flags
CXXFLAGS = -O2 -Wall

# Target executable
TARGET = vector_add_test

# Source files
SOURCES = vector_add.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(HIPCC) $(CXXFLAGS) -o $@ $^

# Compile source files
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)

# Print GPU information
info:
	rocminfo

.PHONY: all run clean info

