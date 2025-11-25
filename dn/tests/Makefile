# Makefile for Vector Addition on AMD GPU using HIP

# Compiler
HIPCC = hipcc

# Compiler flags
CXXFLAGS = -O2 -Wall -std=c++11 -pthread

# Target executables
TARGET_1GPU = vector_add_test
TARGET_2GPU = vector_add_2gpu_test

# Source files
SOURCES_1GPU = vector_add.cpp
SOURCES_2GPU = vector_add_2gpu.cpp

# Object files
OBJECTS_1GPU = $(SOURCES_1GPU:.cpp=.o)
OBJECTS_2GPU = $(SOURCES_2GPU:.cpp=.o)

# Default target - build both programs
all: $(TARGET_1GPU) $(TARGET_2GPU)

# Build single GPU version
$(TARGET_1GPU): $(OBJECTS_1GPU)
	$(HIPCC) $(CXXFLAGS) -o $@ $^

# Build 2 GPU version
$(TARGET_2GPU): $(OBJECTS_2GPU)
	$(HIPCC) $(CXXFLAGS) -o $@ $^

# Compile source files
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Run single GPU version
run: $(TARGET_1GPU)
	./$(TARGET_1GPU)

# Run 2 GPU version
run-2gpu: $(TARGET_2GPU)
	./$(TARGET_2GPU)

# Run both programs
run-all: $(TARGET_1GPU) $(TARGET_2GPU)
	@echo "========================================="
	@echo "Running Single GPU Version:"
	@echo "========================================="
	./$(TARGET_1GPU)
	@echo ""
	@echo "========================================="
	@echo "Running 2 GPU Version:"
	@echo "========================================="
	./$(TARGET_2GPU)

# Clean build artifacts
clean:
	rm -f $(TARGET_1GPU) $(TARGET_2GPU) $(OBJECTS_1GPU) $(OBJECTS_2GPU)

# Print GPU information
info:
	rocminfo

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build both programs (default)"
	@echo "  vector_add - Build single GPU version only"
	@echo "  vector_add_2gpu - Build 2 GPU version only"
	@echo "  run        - Run single GPU version"
	@echo "  run-2gpu   - Run 2 GPU version"
	@echo "  run-all    - Run both programs sequentially"
	@echo "  clean      - Remove all build artifacts"
	@echo "  info       - Display GPU information"
	@echo "  help       - Show this help message"

.PHONY: all run run-2gpu run-all clean info help
