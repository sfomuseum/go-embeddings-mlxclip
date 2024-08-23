# go-embeddings-mlxclip

Go package to implement the `whosonfirst/go-dedupe/embeddings.Embedder` interface using the [MLX_CLIP](https://github.com/harperreed/mlx_clip) Python and Apple's `MLX` libraries.

## Usage

_Error handling removed for the sake of brevity._

```
import (
	"context"

	_ "github.com/sfomuseum/go-embeddings-mlxclip"
	"github.com/whosonfirst/go-dedupe/embeddings"
)	
	
func main() {

	ctx := context.Background()

	// Set setup notes below for details (they are important)
	emb_uri := "mlxclip:///path/to/your/embeddings.py"
	
	emb, _ := embeddings.NewEmbedder(ctx, emb_uri)

	embeddings, _ := emb.Embeddings(ctx, "Hello world")
	// Do something with embeddings here...
}
```

## Setup (this part is important)

This package assumes that you have already installed and configured the [mlx_clip](https://github.com/harperreed/mlx_clip) Python library and all its dependencies (including the need for the code to be run on Apple Silicon hardware).

It is still the case that "installing [insert machine-learning thing here] and all its dependencies" can be a challenge so there is no attempt to automate it here. If you can run the `embeddings.py` script detailed below from the command-line then the rest of this package should work as documented.

### embeddings.py

What follows is the "simplest and dumbest" `embeddings.py` script possible. You can write your own version, and call it whatever you want. The only requirements are that the script accept (3) ordered input parameters. They are:

1. The "target" for the embedding types. Valid options are: image, text.
2. The "input" data to process. If `target` is "text" then this value is a string. If `target` is "image" then this value is the path to an image on the local disk.
3. The "output" file where JSON-encoded embeddings should be written to the local disk.

For example:

```
$> python3 embeddings text "hello world" /tmp/mlx-tmp-1234.json
```

For example:

```
from mlx_clip import mlx_clip

import sys
import json

if __name__ == "__main__":

    model_dir = "/usr/local/src/mlx-examples/clip/mlx_model"
    clip = mlx_clip(model_dir)

    target = sys.argv[1]
    input = sys.argv[2]
    output = sys.argv[3]

    with open(output, "w") as wr :

        if target == "image":
            image_embedding = clip.image_encoder(input)
            json.dump(image_embedding, wr)
        else :
            text_embedding = clip.text_encoder(text)
            json.dump(text_embedding, wr)
```

## See also

* https://github.com/whosonfirst/go-dedupe
* https://github.com/harperreed/mlx_clip
* https://github.com/ml-explore/mlx