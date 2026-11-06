# LOCAL-LLM-CHAT

`LOCAL-LLM-CHAT` is an experimental project for running local language
models directly on a personal computer.

The project is not intended to be another hosted chatbot or a replacement for
ChatGPT. Its purpose is to explore how far small, locally hosted models can go
when they are given a usable context window, access to project files, and a
small set of controlled tools.

The longer-term goal is to grow this experiment into a simple local coding
agent, conceptually similar to tools such as OpenCode, while keeping the model
and inference process local.

## Current state

The application currently provides a terminal user interface for:

- Loading GGUF models through `llama.cpp`.
- Sending prompts with conversation history.
- Selecting project files and including them in the model context.
- Persisting multiple local sessions.
- Showing context usage, generation statistics, and debug information.
- Allowing the model to list directories, change its working directory, read
  files, write files, and request shell commands.

Shell commands require explicit approval from the user. File operations are
restricted to the project directory.

The agent functionality is intentionally small and experimental. The main
focus of this repository is local inference, context handling, and finding a
practical combination of model, quantization, and hardware.

## How it works

The application is written in Rust and uses
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2), a Rust binding for
[`llama.cpp`](https://github.com/ggerganov/llama.cpp).

The runtime loads a local `.gguf` file, applies the chat template provided by
the model, evaluates the prompt, and streams the generated response. CUDA is
supported through the `cuda` feature, but the application can also run on the
CPU.

The project currently detects the tool-call format used by the loaded model.
For example, Qwen 3.5 uses XML-style tool calls, while Qwen 2.5 commonly uses
JSON inside a `<tool_call>` block.

## Requirements

You need:

- Rust and Cargo.
- CMake and a C/C++ compiler.
- A local GGUF model.
- CUDA and a compatible NVIDIA GPU if GPU inference is desired.

On NixOS, the repository includes `flake.nix` and `shell.nix` with the
development dependencies needed to build `llama.cpp`.

## Running the application

The model is deliberately not included in this repository. Set `MODEL_PATH`
to the GGUF file you want to use:

```bash
MODEL_PATH=./models/Qwen3.5-2B-Q4_K_M.gguf cargo run --release
```

The application starts in the terminal interface. Press `Ctrl-Q` to exit.
The help tab inside the application contains the available keyboard shortcuts
and commands.

Useful configuration variables include:

```bash
MODEL_PATH=./models/model.gguf
N_CTX=16384
MAX_TOKENS=256
N_GPU_LAYERS=999
TEMPERATURE=0.35
TOP_P=0.9
cargo run --release
```

Set `N_GPU_LAYERS=0` to force CPU inference. The default context size is
limited to 16,384 tokens unless `N_CTX` is set explicitly.

## Models used during development

The model files in the local `models/` directory are ignored by Git and must
be downloaded separately.

The main model used for this experiment is:

- **Qwen3.5 2B, Q4_K_M** — based on
  [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B), using the GGUF
  distribution published by
  [`unsloth/Qwen3.5-2B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF).

Another model used for comparison is:

- **Qwen2.5 1.5B Instruct, Q4_K_M** — available from the
  [`Qwen/Qwen2.5-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
  repository.

Model licenses and usage conditions belong to their respective authors and
distributors. Check the original Hugging Face repository before using or
redistributing a model.

## Local data and generated files

Sessions are stored locally under `.llm-cheap/sessions/`. Build output is
stored under `target/`, and downloaded model files are expected under
`models/`. These paths are excluded from version control.

## Project direction

This repository is an experiment rather than a finished product. Possible
future work includes improving context management, making tool execution more
reliable, adding better edit and diff workflows, supporting more model
templates, and making the local agent easier to use across different
projects and hardware configurations.

The project is intentionally being developed from the bottom up: first the
local inference runtime and context handling, then the agent behavior built on
top of them.
