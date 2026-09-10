package openai

import (
	"encoding/json"
	"testing"
)

// Regression test (2026-09-10). Codex declares tools the local runner cannot
// express, and forwarding them used to break everything downstream:
//
//   - {"type":"web_search"} carries NO name. Ollama then returned EVERY
//     function_call with an empty name for template-parsed models (qwen3,
//     mistral), so Codex reported "unsupported call:" and the model retried
//     forever. gpt-oss hid this because it uses Ollama's native parser.
//   - {"type":"namespace"} (Codex sub-agent container) made llama.cpp reject
//     the whole tools array: HTTP 500 "Failed to parse tools: Unsupported
//     tool type".
//
// Neither can be called by a local model, so FromResponsesRequest must drop
// them while keeping the usable function/custom tools.
func TestFromResponsesRequestDropsUnsupportedTools(t *testing.T) {
	reqJSON := `{
		"model": "gpt-oss:20b",
		"input": "hello",
		"tools": [
			{
				"type": "function",
				"name": "exec_command",
				"description": "Runs a command",
				"strict": false,
				"parameters": {
					"type": "object",
					"properties": {"cmd": {"type": "string"}},
					"required": ["cmd"]
				}
			},
			{
				"type": "custom",
				"name": "apply_patch",
				"description": "FREEFORM apply_patch",
				"strict": null,
				"parameters": null
			},
			{
				"type": "web_search",
				"name": "",
				"description": null,
				"strict": null,
				"parameters": null
			},
			{
				"type": "namespace",
				"name": "collaboration",
				"description": "Tools for spawning sub-agents",
				"strict": null,
				"parameters": null
			},
			{
				"type": "function",
				"name": "",
				"description": "nameless function tool",
				"strict": false,
				"parameters": {"type": "object", "properties": {}}
			}
		]
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	if len(req.Tools) != 5 {
		t.Fatalf("expected 5 declared tools, got %d", len(req.Tools))
	}

	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	var got []string
	for _, tool := range chatReq.Tools {
		got = append(got, tool.Function.Name)
	}
	want := []string{"exec_command", "apply_patch"}
	if len(got) != len(want) {
		t.Fatalf("expected tools %v, got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tool[%d]: expected %q, got %q", i, want[i], got[i])
		}
	}
}
