package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// namespaceToolsRequest mirrors what Codex sends for an MCP server: one
// Responses "namespace" container whose nested function tools carry the real
// schemas. Chat-template runners cannot express that shape, so the converter
// flattens it to "<namespace>__<tool>" function names.
func namespaceToolsRequest() ResponsesRequest {
	return ResponsesRequest{
		Model: "test-model",
		Input: ResponsesInput{Text: "hello"},
		Tools: []ResponsesTool{
			{
				Type:        "namespace",
				Name:        "mcp__files",
				Description: ptrStr("File tools"),
				Tools: []ResponsesTool{
					{
						Type:        "function",
						Name:        "read_file",
						Description: ptrStr("Read a file"),
						Parameters: map[string]any{
							"type":       "object",
							"properties": map[string]any{"path": map[string]any{"type": "string"}},
							"required":   []any{"path"},
						},
					},
					{
						Type:        "function",
						Name:        "write_file",
						Description: ptrStr("Write a file"),
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"path":    map[string]any{"type": "string"},
								"content": map[string]any{"type": "string"},
							},
							"required": []any{"path", "content"},
						},
					},
				},
			},
		},
	}
}

// TestNamespaceToolFromResponsesRequest ensures nested namespace tools reach the
// runner as flat function tools with their schemas intact.
func TestNamespaceToolFromResponsesRequest(t *testing.T) {
	chatReq, err := FromResponsesRequest(namespaceToolsRequest())
	if err != nil {
		t.Fatalf("FromResponsesRequest: %v", err)
	}
	if len(chatReq.Tools) != 2 {
		t.Fatalf("expected 2 flattened tools, got %d", len(chatReq.Tools))
	}
	byName := map[string]api.Tool{}
	for _, tool := range chatReq.Tools {
		byName[tool.Function.Name] = tool
	}
	for _, want := range []string{"mcp__files__read_file", "mcp__files__write_file"} {
		if _, ok := byName[want]; !ok {
			t.Fatalf("missing flattened tool %q", want)
		}
	}
	b, _ := json.Marshal(byName["mcp__files__write_file"].Function.Parameters)
	for _, want := range []string{`"path"`, `"content"`} {
		if !strings.Contains(string(b), want) {
			t.Errorf("write_file schema missing %s: %s", want, b)
		}
	}
}

// TestNamespaceToolToResponse ensures a flat function call comes back as the
// {name, namespace} pair Codex's router accepts.
func TestNamespaceToolToResponse(t *testing.T) {
	req := namespaceToolsRequest()
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "note.txt")
	args.Set("content", "hi")
	chatResp := api.ChatResponse{
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{ID: "call_1", Function: api.ToolCallFunction{Name: "mcp__files__write_file", Arguments: args}},
			},
		},
	}
	resp := ToResponse("m", "resp_1", "item_1", chatResp, req)
	if len(resp.Output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(resp.Output))
	}
	item := resp.Output[0]
	if item.Type != "function_call" {
		t.Fatalf("expected function_call, got %q", item.Type)
	}
	if item.Name != "write_file" || item.Namespace != "mcp__files" {
		t.Errorf("expected name=write_file namespace=mcp__files, got name=%q namespace=%q", item.Name, item.Namespace)
	}
}

// TestNamespaceToolNameTolerance covers the spellings local models actually
// produce: full flat name, single-underscore collapse, and the bare tool name.
func TestNamespaceToolNameTolerance(t *testing.T) {
	req := namespaceToolsRequest()
	for _, name := range []string{"mcp__files__read_file", "mcp__files_read_file", "read_file"} {
		chatResp := api.ChatResponse{
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{ID: "call_1", Function: api.ToolCallFunction{Name: name, Arguments: api.NewToolCallFunctionArguments()}},
				},
			},
		}
		resp := ToResponse("m", "resp_1", "item_1", chatResp, req)
		if len(resp.Output) != 1 {
			t.Fatalf("%s: expected 1 output item, got %d", name, len(resp.Output))
		}
		item := resp.Output[0]
		if item.Name != "read_file" || item.Namespace != "mcp__files" {
			t.Errorf("%s: expected read_file/mcp__files, got name=%q namespace=%q", name, item.Name, item.Namespace)
		}
	}
}

// TestNamespaceToolHistory ensures namespaced calls replayed in the conversation
// history are offered to the runner under their flat name.
func TestNamespaceToolHistory(t *testing.T) {
	req := namespaceToolsRequest()
	req.Input = ResponsesInput{Items: []ResponsesInputItem{
		ResponsesInputMessage{Role: "user", Content: []ResponsesContent{ResponsesTextContent{Type: "input_text", Text: "go"}}},
		ResponsesFunctionCall{ID: "item_1", Type: "function_call", CallID: "call_1", Name: "write_file", Namespace: "mcp__files", Arguments: `{"path":"a","content":"b"}`},
		ResponsesFunctionCallOutput{Type: "function_call_output", CallID: "call_1", Output: "ok"},
	}}
	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("FromResponsesRequest: %v", err)
	}
	if len(chatReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
	}
	assistant := chatReq.Messages[1]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 1 {
		t.Fatalf("expected assistant message with 1 tool call")
	}
	if got := assistant.ToolCalls[0].Function.Name; got != "mcp__files__write_file" {
		t.Errorf("expected flattened history tool call mcp__files__write_file, got %q", got)
	}
}

// TestNamespaceToolStream ensures the streaming item events (added/done) and the
// final response.completed output all carry {name, namespace}.
func TestNamespaceToolStream(t *testing.T) {
	req := namespaceToolsRequest()
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "note.txt")
	args.Set("content", "hi")
	chatResp := api.ChatResponse{
		Done: true,
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{ID: "call_1", Function: api.ToolCallFunction{Name: "mcp__files__write_file", Arguments: args}},
			},
		},
	}
	conv := NewResponsesStreamConverter("resp_1", "item_1", "test-model", req)
	events := conv.Process(chatResp)

	var sawAdded, sawDone, sawCompleted bool
	for _, ev := range events {
		data, _ := ev.Data.(map[string]any)
		switch ev.Event {
		case "response.output_item.added", "response.output_item.done":
			item, _ := data["item"].(map[string]any)
			if item == nil || item["type"] != "function_call" {
				continue
			}
			if item["name"] != "write_file" || item["namespace"] != "mcp__files" {
				t.Errorf("%s item = %v", ev.Event, item)
			}
			if ev.Event == "response.output_item.added" {
				sawAdded = true
			} else {
				sawDone = true
			}
		case "response.completed":
			response, _ := data["response"].(map[string]any)
			output, _ := response["output"].([]any)
			for _, raw := range output {
				item, _ := raw.(map[string]any)
				if item["type"] == "function_call" && item["name"] == "write_file" && item["namespace"] == "mcp__files" {
					sawCompleted = true
				}
			}
		}
	}
	if !sawAdded || !sawDone || !sawCompleted {
		t.Errorf("missing namespaced function_call events: added=%v done=%v completed=%v", sawAdded, sawDone, sawCompleted)
	}
}

// TestNamespaceToolJSONUnmarshal pins the wire shape: without the nested Tools
// field the sub-tools were silently dropped by encoding/json.
func TestNamespaceToolJSONUnmarshal(t *testing.T) {
	reqJSON := `{
		"model": "test-model",
		"input": "hi",
		"tools": [
			{
				"type": "namespace",
				"name": "mcp__files",
				"description": "File tools",
				"strict": null,
				"parameters": null,
				"tools": [
					{
						"type": "function",
						"name": "write_file",
						"description": "Write a file",
						"strict": false,
						"parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
					}
				]
			}
		]
	}`
	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	if len(req.Tools) != 1 || len(req.Tools[0].Tools) != 1 {
		t.Fatalf("nested tools were not decoded: %+v", req.Tools)
	}
	if req.Tools[0].Tools[0].Name != "write_file" {
		t.Fatalf("expected nested write_file, got %q", req.Tools[0].Tools[0].Name)
	}
	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("FromResponsesRequest: %v", err)
	}
	if len(chatReq.Tools) != 1 || chatReq.Tools[0].Function.Name != "mcp__files__write_file" {
		t.Fatalf("expected one flattened tool, got %+v", chatReq.Tools)
	}
}
