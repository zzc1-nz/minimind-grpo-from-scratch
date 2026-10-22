# Third-party notices

## MiniMind

This project builds on [MiniMind](https://github.com/jingyaogong/minimind),
distributed under the **Apache License 2.0**.

The MiniMind source is **not vendored** into this repository: `setup.py` clones it into
`src/minimind-master` at install time. Any redistribution of MiniMind itself must retain
its Apache-2.0 license and NOTICE requirements. The `custom/` directory contains thin
derivatives of MiniMind's trainers; those files are likewise subject to Apache-2.0.

## Datasets

The corpora downloaded by `download_data.py` come from the MiniMind dataset release on
ModelScope (`gongjy/minimind_dataset`). They are not redistributed here; consult that
release for the upstream licenses of the constituent sources.

## Trained weights

The trained checkpoints and GGUF files are not distributed in this repository.

## llama.cpp

GGUF conversion and local inference use [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT). Prebuilt Windows binaries are fetched from its GitHub releases; none are committed
here.