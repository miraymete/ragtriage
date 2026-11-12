# Troubleshooting

Common problems when running ragtriage, and what to do about them. Errors from `ragtriage run`
start with `error:` and exit with code 2.

## A file is not found

```text
error: [Errno 2] No such file or directory: 'examples/turkish_quickstart/config.yaml'
```

Paths are resolved in two different places:

| Path | Relative to |
|---|---|
| The config file given to `ragtriage run` | The folder you run the command from |
| `--out` and `--html` | The folder you run the command from |
| `documents` and `dataset` inside the config | The config file's own folder |

Run the examples from the repository root, or pass an absolute path to the config. Moving a config
file to another folder breaks its relative `documents` and `dataset` paths, so update them or keep
the file where it is.

## Setting up on Windows PowerShell

Create the virtual environment and install from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ragtriage run examples\turkish_quickstart\config.yaml --failures
```

Calling `.venv\Scripts\...` directly works without activating the environment. If you prefer to
activate it, run `.venv\Scripts\Activate.ps1`; when PowerShell's execution policy blocks that
script, keep using the direct paths above instead.

On Linux or macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ragtriage run examples/turkish_quickstart/config.yaml --failures
```

If the `ragtriage` command is not found, the environment is not the one you installed into.
`python -m ragtriage.cli run ...` with that environment's Python runs the same command.

## XQuAD-tr examples need their data first

The configs in `examples/xquad_tr/` read `data/xquad_tr/`, which is not in the repository. Without
it, a run stops with an error such as:

```text
error: no .txt files in ...\examples\xquad_tr\..\..\data\xquad_tr\dev\docs
```

Build the data once from the repository root. It downloads about 712 KB and checks its SHA-256:

```bash
python examples/xquad_tr/prepare.py
```

The script stops if the hash or the recorded split does not match. Source, license and the full
list of commands are in [`examples/xquad_tr/SOURCE.md`](../examples/xquad_tr/SOURCE.md).

## An optional dependency is missing

Embedding models and the Anthropic provider are optional extras, so a plain install does not
include them. A config that needs one stops before any work with one of these messages:

```text
error: this embedder needs: pip install "ragtriage[embeddings]"
error: generation not started: the anthropic provider needs: pip install "ragtriage[anthropic]"
```

In a clone of this repository, install the extra in editable mode:

```bash
pip install -e ".[embeddings]"
pip install -e ".[anthropic]"
```

On Windows, use `.venv\Scripts\python -m pip install -e ".[embeddings]"` as above. Before you
run:

- The first run with a `sentence_transformers` embedder downloads the model from Hugging Face.
  `examples/turkish_quickstart/config_embeddings.yaml` states the size of its model.
- The Anthropic provider makes paid API calls and reads its credentials from the environment
  through the SDK. Never put a key in a config file. See
  [Answer generation](../README.md#answer-generation-experimental).

## The HTML report is too large

`--html` writes a compact page by default: gold evidence and retrieved passages only for questions
with a problem. Two options change only the page, never the metrics or the `--out` JSON:

```bash
# Every question with its passages (the largest page)
ragtriage run CONFIG --html report.html --html-detail full

# At most 5 retrieved results per question
ragtriage run CONFIG --html report.html --html-depth 5
```

- `--html-depth` must be between 1 and the config's `depth`; otherwise the run stops before
  retrieval with `error: the HTML display depth must be between 1 and the run's depth ...`.
- `--html-detail` and `--html-depth` need `--html`; otherwise:
  `error: --html-detail and --html-depth need --html`.
- When generation ran, the context sent to the model stays complete whatever `--html-depth` is, so
  those reports stay larger.

What the compact page keeps and leaves out is described in
[Report detail and size](../README.md#report-detail-and-size).
