from __future__ import annotations

import argparse
import json
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .data_pipeline_agent import LLMBuildOrchestrator
from .enterprise_agent import EnterpriseLLMAgent, ValidationError, WeightBundle


INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Enterprise LLM Agent UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 1000px; }
    h1 { margin-bottom: 8px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
    label { display: block; margin-top: 8px; font-weight: 600; }
    input, select, textarea, button { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
    textarea { min-height: 180px; font-family: monospace; }
    button { margin-top: 12px; cursor: pointer; }
    pre { background: #111; color: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Enterprise LLM Agent</h1>
  <p>Production-style UI for both onboarding flows.</p>

  <div class=\"grid\">
    <div class=\"card\">
      <h2>Weight-first assessment</h2>
      <label>Has full checkpoint</label>
      <select id=\"has_full_checkpoint\"><option value=\"false\">false</option><option value=\"true\">true</option></select>
      <label>Architecture match</label>
      <select id=\"architecture_match\"><option value=\"true\">true</option><option value=\"false\">false</option></select>
      <label>Tokenizer included</label>
      <select id=\"tokenizer_included\"><option value=\"false\">false</option><option value=\"true\">true</option></select>
      <label>Provided layers</label>
      <input id=\"provided_layers\" type=\"number\" value=\"20\" min=\"0\" />
      <label>Total layers</label>
      <input id=\"total_layers\" type=\"number\" value=\"40\" min=\"1\" />
      <button onclick=\"runWeights()\">Run weight assessment</button>
    </div>

    <div class=\"card\">
      <h2>Data-first orchestration</h2>
      <label>Dataset format</label>
      <select id=\"dataset_format\"><option value=\"jsonl\">jsonl</option><option value=\"csv\">csv</option></select>
      <label>Total layers</label>
      <input id=\"pipeline_total_layers\" type=\"number\" value=\"40\" min=\"1\" />
      <label>Dataset content (JSONL or CSV)</label>
      <textarea id=\"dataset_content\">{"text":"policy doc one","label":"policy"}\n{"text":"runbook entry","label":"ops"}</textarea>
      <button onclick=\"runPipeline()\">Run dataset orchestration</button>
    </div>
  </div>

  <h2>Result</h2>
  <pre id=\"result\">{}</pre>

<script>
async function postJson(path, payload) {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return await resp.json();
}

function show(data) {
  document.getElementById('result').textContent = JSON.stringify(data, null, 2);
}

async function runWeights() {
  const payload = {
    has_full_checkpoint: document.getElementById('has_full_checkpoint').value,
    architecture_match: document.getElementById('architecture_match').value,
    tokenizer_included: document.getElementById('tokenizer_included').value,
    provided_layers: Number(document.getElementById('provided_layers').value),
    total_layers: Number(document.getElementById('total_layers').value)
  };
  show(await postJson('/api/assess-weights', payload));
}

async function runPipeline() {
  const payload = {
    dataset_format: document.getElementById('dataset_format').value,
    total_layers: Number(document.getElementById('pipeline_total_layers').value),
    dataset_content: document.getElementById('dataset_content').value
  };
  show(await postJson('/api/orchestrate-dataset', payload));
}
</script>
</body>
</html>
"""


def assess_weights(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = WeightBundle.from_dict(payload)
    return EnterpriseLLMAgent().assess(bundle)


def orchestrate_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_content = str(payload.get("dataset_content", "")).strip()
    dataset_format = str(payload.get("dataset_format", "jsonl")).lower().strip()
    total_layers = int(payload.get("total_layers", 40))

    if not dataset_content:
        raise ValidationError("dataset_content must not be empty")
    if dataset_format not in {"jsonl", "csv"}:
        raise ValidationError("dataset_format must be either 'jsonl' or 'csv'")

    suffix = ".jsonl" if dataset_format == "jsonl" else ".csv"
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / f"dataset{suffix}"
        path.write_text(dataset_content + "\n", encoding="utf-8")
        return LLMBuildOrchestrator().run(str(path), total_layers=total_layers)


class WebHandler(BaseHTTPRequestHandler):
    def _json_response(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            data = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValidationError("request body must be a JSON object")

            if self.path == "/api/assess-weights":
                result = assess_weights(payload)
            elif self.path == "/api/orchestrate-dataset":
                result = orchestrate_dataset(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json_response(HTTPStatus.OK, {"ok": True, "result": result})
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            self._json_response(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web frontend for enterprise LLM agents")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"Serving Enterprise LLM Agent UI at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
