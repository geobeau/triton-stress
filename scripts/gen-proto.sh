#!/usr/bin/env bash
# Regenerates pkg/tritonpb from proto/ (the proto files stay untouched: the Go package is set
# with M flags). Needs protoc, protoc-gen-go and protoc-gen-go-grpc on PATH:
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PKG=github.com/criteo/triton-stress/pkg/tritonpb
rm -f pkg/tritonpb/*.pb.go
protoc -I proto \
  --go_out=. --go_opt=module=github.com/criteo/triton-stress \
  --go_opt=Mgrpc_service.proto=$PKG --go_opt=Mmodel_config.proto=$PKG \
  --go-grpc_out=. --go-grpc_opt=module=github.com/criteo/triton-stress \
  --go-grpc_opt=Mgrpc_service.proto=$PKG --go-grpc_opt=Mmodel_config.proto=$PKG \
  proto/grpc_service.proto proto/model_config.proto
