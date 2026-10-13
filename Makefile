export PATH := $(HOME)/.local/go/bin:$(HOME)/go/bin:$(PATH)

# pkg/tensorgen is its own Go module (see pkg/tensorgen/go.mod): every check runs in both.
TENSORGEN := pkg/tensorgen

# Printed by rts --version. The release workflow sets it to the release tag.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
GOLANGCI_LINT ?= golangci-lint

.PHONY: build test race lint golangci proto

build:
	CGO_ENABLED=0 go build -ldflags "-X main.buildVersion=$(VERSION)" -o bin/rts ./cmd/rts

test:
	go test ./...
	cd $(TENSORGEN) && go test ./...

# The load path under the race detector.
race:
	go test -race -count=1 ./internal/... ./cmd/...
	cd $(TENSORGEN) && go test -race -count=1 ./...

lint:
	@test -z "$$(gofmt -l -s cmd pkg internal)" || (gofmt -d -s cmd pkg internal; exit 1)
	go vet ./...
	cd $(TENSORGEN) && go vet ./...
	@# The core must stay free of generated protobuf code (it cannot be linked twice into one binary).
	@cd $(TENSORGEN) && ! go list -deps ./... | grep -q 'tritonpb\|grpc' || (echo "pkg/tensorgen must not depend on generated protobuf or grpc"; exit 1)

# Needs golangci-lint v2 on PATH (the version CI uses is pinned in .github/workflows/ci.yml).
golangci:
	$(GOLANGCI_LINT) run ./...
	cd $(TENSORGEN) && $(GOLANGCI_LINT) run ./...

proto:
	bash scripts/gen-proto.sh
