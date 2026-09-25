# ==============================================================================
# Makefile for 3024-Dimensional Korean Utterance Embedding Engine
# ==============================================================================

.PHONY: all build test fmt vet clean run-embed run-sim run-cluster help

BIN_DIR := bin
EMBEDDER_BIN := $(BIN_DIR)/embedder

all: fmt vet build test

build: $(EMBEDDER_BIN)

GO_SRCS := $(shell find cmd pkg -type f -name '*.go' 2>/dev/null)

$(EMBEDDER_BIN): $(GO_SRCS)
	@mkdir -p $(BIN_DIR)
	@echo "==> Building Korean Utterance Embedding Engine CLI..."
	go build -v -o $(EMBEDDER_BIN) ./cmd/embedder

test:
	@echo "==> Running all Go tests..."
	go test -v ./pkg/...

bench:
	@echo "==> Running benchmarks..."
	go test -bench=. -benchmem ./pkg/...

fmt:
	@echo "==> Formatting Go code with gofmt..."
	gofmt -s -w .

vet:
	@echo "==> Running go vet..."
	go vet ./...

clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf $(BIN_DIR)

run-embed: build
	@echo "==> Running test embedding..."
	./$(EMBEDDER_BIN) embed --text="오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다." --dim=3024

run-sim: build
	@echo "==> Running test similarity comparison..."
	./$(EMBEDDER_BIN) similarity \
		--text1="오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다." \
		--text2="금일 안건 회의는 15시 회의실 진행 예정입니다."

run-cluster: build
	@echo "==> Running test clustering..."
	./$(EMBEDDER_BIN) cluster --k=2

run-rag: build
	@echo "==> Running Korean RAG demonstration..."
	./$(EMBEDDER_BIN) rag demo

help:
	@echo "Available commands:"
	@echo "  make build       - Build CLI binary ($(EMBEDDER_BIN))"
	@echo "  make test        - Run unit tests"
	@echo "  make bench       - Run benchmarks"
	@echo "  make fmt         - Run gofmt on all Go source files"
	@echo "  make vet         - Run go vet"
	@echo "  make clean       - Remove compiled binaries"
	@echo "  make run-embed   - Test single utterance embedding"
	@echo "  make run-sim     - Test semantic similarity breakdown"
	@echo "  make run-cluster - Test utterance clustering"
	@echo "  make run-rag     - Run Korean RAG pipeline demo"

