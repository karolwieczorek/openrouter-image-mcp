# openrouter-image-mcp

An MCP server that generates and edits images through OpenRouter's Images
API (`POST /api/v1/images`). It is meant to be started by a project, which
hands it that project's API key and the one folder it may write to.

## Tools

- `generate_image`: text to image, or image-guided (`references`: paths
  inside the output root or HTTP(S) URLs). Saves each image plus a `.json`
  sidecar with the prompt, model, parameters and actual cost, and returns
  small previews. It never replaces an existing file unless `overwrite` is
  set: an `out_path` that is taken is refused before anything is paid for.
- `list_image_models`: models, their parameters and, on request, prices
  (per token for input text, input images and output images).
- `get_key_status`: the key's limit, remaining credit, usage per day, week
  and month, and enforced budgets. The key itself is never returned.

Spending limits live on OpenRouter (set a limit on the key); the server
has none of its own.

## Environment

| Variable | Meaning |
| --- | --- |
| `OPENROUTER_API_KEY` | The key (required for generating and key status). |
| `IMAGE_GEN_OUTPUT_ROOT` | Absolute folder; all writes and file references stay inside it. |
| `IMAGE_GEN_DEFAULT_DIR` | Default output folder relative to the root (default `generated`). |
| `IMAGE_GEN_DEFAULT_MODEL` | Model used when a call names none (optional). |
| `IMAGE_GEN_APP_TITLE` | `X-Title` header for OpenRouter's app attribution (optional). |

## Running

```
uv --directory <this folder> run openrouter-image-mcp
```

over stdio, from a launcher that sets the environment above. A project
launcher should keep its own key (for example in the project's gitignored
`.env`) and pass it as `OPENROUTER_API_KEY` only to this process.

## Checks

```
uv run pytest
```

The tests need no network or key.

## License

MIT, see [LICENSE](LICENSE).
