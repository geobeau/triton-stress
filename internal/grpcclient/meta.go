package grpcclient

import (
	"context"
	"fmt"

	"github.com/criteo/triton-stress/pkg/tensorgen"
	pb "github.com/criteo/triton-stress/pkg/tritonpb"
	"github.com/criteo/triton-stress/pkg/tritonpbadapter"
)

// FetchModel reads ModelMetadata and ModelConfig for model (version empty is
// Triton's latest) over the regular proto codec and returns the model description.
func (c *Conn) FetchModel(ctx context.Context, model, version string) (*tensorgen.Model, error) {
	cl := pb.NewGRPCInferenceServiceClient(c.cc)
	meta, err := cl.ModelMetadata(ctx, &pb.ModelMetadataRequest{Name: model, Version: version})
	if err != nil {
		return nil, fmt.Errorf("fetching model metadata: %w", err)
	}
	cfg, err := cl.ModelConfig(ctx, &pb.ModelConfigRequest{Name: model, Version: version})
	if err != nil {
		return nil, fmt.Errorf("fetching model config: %w", err)
	}
	if cfg.GetConfig() == nil {
		return nil, fmt.Errorf("ModelConfig response has no config")
	}
	return tritonpbadapter.NewModel(meta, cfg.GetConfig(), model, version)
}
