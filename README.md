# go-embeddings-mlxclip

Go package to implement the `whosonfirst/go-dedupe/embeddings.Embedder` interface using the [MLX_CLIP](https://github.com/harperreed/mlx_clip) Python and Apple's `MLX` libraries.

## Usage

## embeddings.py

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