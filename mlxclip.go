//go:build mlxclip

package mlxclip

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"

	wof_embeddings "github.com/whosonfirst/go-dedupe/embeddings"
)

type MLXClipEmbedder struct {
	wof_embeddings.Embedder
	embeddings_py string
}

func init() {
	ctx := context.Background()
	err := wof_embeddings.RegisterEmbedder(ctx, "mlxclip", NewMLXClipEmbedder)
	if err != nil {
		panic(err)
	}
}

func NewMLXClipEmbedder(ctx context.Context, uri string) (wof_embeddings.Embedder, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, fmt.Errorf("Failed to parse URI, %w", err)
	}

	embeddings_py := u.Path

	_, err = os.Stat(embeddings_py)

	if err != nil {
		return nil, err
	}

	e := &MLXClipEmbedder{
		embeddings_py: embeddings_py,
	}

	return e, nil
}

func (e *MLXClipEmbedder) Embeddings(ctx context.Context, content string) ([]float64, error) {

	e32, err := e.Embeddings32(ctx, content)

	if err != nil {
		return nil, err
	}

	return e.asFloat64(e32), nil
}

func (e *MLXClipEmbedder) Embeddings32(ctx context.Context, content string) ([]float32, error) {

	return e.generate_embeddings(ctx, "text", content)
}

func (e *MLXClipEmbedder) ImageEmbeddings(ctx context.Context, data []byte) ([]float64, error) {

	e32, err := e.ImageEmbeddings32(ctx, data)

	if err != nil {
		return nil, err
	}

	return e.asFloat64(e32), nil
}

func (e *MLXClipEmbedder) ImageEmbeddings32(ctx context.Context, data []byte) ([]float32, error) {

	tmp, err := os.CreateTemp("", "mlxclip.*.img")

	if err != nil {
		return nil, fmt.Errorf("Failed to create tmp file, %w", err)
	}

	defer os.Remove(tmp.Name()) // clean up

	_, err = tmp.Write(data)

	if err != nil {
		return nil, err
	}

	err = tmp.Close()

	if err != nil {
		return nil, err
	}

	return e.generate_embeddings(ctx, "image", tmp.Name())
}

func (e *MLXClipEmbedder) generate_embeddings(ctx context.Context, target string, input string) ([]float32, error) {

	tmp, err := os.CreateTemp("", "mlxclip.*.json")

	if err != nil {
		return nil, fmt.Errorf("Failed to create tmp file, %w", err)
	}

	defer os.Remove(tmp.Name())

	err = tmp.Close()

	if err != nil {
		return nil, err
	}

	cmd := exec.CommandContext(ctx, "python3", e.embeddings_py, target, input, tmp.Name())
	err = cmd.Run()

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

	r, err := os.Open(tmp.Name())

	if err != nil {
		return nil, err
	}

	defer r.Close()

	var emb []float32

	dec := json.NewDecoder(r)
	err = dec.Decode(&emb)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w (%s)", err, tmp.Name())
	}

	return emb, nil
}

func (e *MLXClipEmbedder) asFloat64(e32 []float32) []float64 {

	e64 := make([]float64, len(e32))

	for idx, v := range e32 {
		e64[idx] = float64(v)
	}

	return e64
}
