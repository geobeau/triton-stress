FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY pkg/tensorgen/go.mod pkg/tensorgen/go.sum pkg/tensorgen/
RUN go mod download
COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -ldflags "-X main.buildVersion=${VERSION}" -o /out/triton-stress ./cmd/triton-stress

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/triton-stress /usr/local/bin/triton-stress
# triton-stress writes runs/<UTC start> under the working directory by default; / is not writable as nonroot.
WORKDIR /home/nonroot
ENTRYPOINT ["/usr/local/bin/triton-stress"]
