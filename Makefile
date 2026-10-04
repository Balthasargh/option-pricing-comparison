# Makefile pour option_pricing (C++)
CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
LDFLAGS  = -lm

# Optionnel : parallélisation Monte Carlo
# CXXFLAGS += -fopenmp

TARGET = option_pricing
SRC    = option_pricing.cpp

.PHONY: all clean run run-btc run-soja run-all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

run-btc: $(TARGET)
	./$(TARGET) --btc

run-soja: $(TARGET)
	./$(TARGET) --soja

run-all: $(TARGET)
	./$(TARGET) --all

clean:
	rm -f $(TARGET)
