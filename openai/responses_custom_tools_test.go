package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func ptrStr(s string) *string { return &s }

func customRequestWithTools(tools []ResponsesTool) ResponsesRequest {
	return ResponsesRequest{
		Model: "test-model",
		Input: ResponsesInput{Text: "hello"},
		Tools: tools,
	}
}

// TestCustomToolFromResponsesRequest ensures a type:"custom" tool is surfaced to
// the model as a regular function with a synthesized "input" string parameter.
func TestCustomToolFromResponsesRequest(t *testing.T) {
	req := customRequestWithTools([]ResponsesTool{
		{Type: "custom", Name: "apply_patch", Description: ptrStr("applies patches")},
	})
	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("FromResponsesRequest: %v", err)
	}
	if len(chatReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(chatReq.Tools))
	}
	tool := chatReq.Tools[0]
	if tool.Type != "function" {
		t.Errorf("expected api tool type 'function', got %q", tool.Type)
	}
	if tool.Function.Name != "apply_patch" {
		t.Errorf("expected function name apply_patch, got %q", tool.Function.Name)
	}
	// The synthesized schema must declare a required string "input" property.
	b, _ := json.Marshal(tool.Function.Parameters)
	s := string(b)
	if !strings.Contains(s, `"input"`) || !strings.Contains(s, `"string"`) {
		t.Errorf("synthesized parameters missing input/string schema: %s", s)
	}
}

// TestCustomToolToResponse ensures a model function call against a custom tool is
// emitted back as a Responses custom_tool_call item carrying the raw input.
func TestCustomToolToResponse(t *testing.T) {
	req := customRequestWithTools([]ResponsesTool{
		{Type: "custom", Name: "apply_patch", Description: ptrStr("applies patches")},
	})
	args := api.NewToolCallFunctionArguments()
	args.Set("input", "*** Begin Patch\n*** End Patch")
	chatResp := api.ChatResponse{
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{ID: "call_1", Function: api.ToolCallFunction{Name: "apply_patch", Arguments: args}},
			},
		},
	}
	resp := ToResponse("m", "resp_1", "item_1", chatResp, req)
	if len(resp.Output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(resp.Output))
	}
	item := resp.Output[0]
	if item.Type != "custom_tool_call" {
		t.Fatalf("expected output type custom_tool_call, got %q", item.Type)
	}
	if item.Name != "apply_patch" {
		t.Errorf("expected name apply_patch, got %q", item.Name)
	}
	if item.Input != "*** Begin Patch\n*** End Patch" {
		t.Errorf("expected raw input in custom_tool_call item, got %q", item.Input)
	}
}

// TestCustomToolStream ensures streaming emits the custom_tool_call events that
// Codex parses (response.custom_tool_call_input.done / output_item.done).
func TestCustomToolStream(t *testing.T) {
	req := customRequestWithTools([]ResponsesTool{
		{Type: "custom", Name: "apply_patch", Description: ptrStr("applies patches")},
	})
	args := api.NewToolCallFunctionArguments()
	args.Set("input", "patch-body")
	chatResp := api.ChatResponse{
		Done: true,
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{ID: "call_1", Function: api.ToolCallFunction{Name: "apply_patch", Arguments: args}},
			},
		},
	}
	conv := NewResponsesStreamConverter("resp_1", "item_1", "test-model", req)
	events := conv.Process(chatResp)
	sawDone := false
	sawOutputItemDone := false
	for _, ev := range events {
		data, _ := ev.Data.(map[string]any)
		switch ev.Event {
		case "response.custom_tool_call_input.done":
			if data["input"] != "patch-body" {
				t.Errorf("custom_tool_call_input.done input = %v", data["input"])
			}
			sawDone = true
		case "response.output_item.done":
			if item, ok := data["item"].(map[string]any); ok && item["type"] == "custom_tool_call" {
				if item["input"] != "patch-body" {
					t.Errorf("output_item.done input = %v", item["input"])
				}
				sawOutputItemDone = true
			}
		}
	}
	if !sawDone || !sawOutputItemDone {
		t.Errorf("missing custom_tool_call events: done=%v output_item.done=%v", sawDone, sawOutputItemDone)
	}
}

// TestCustomToolHistory ensures prior custom_tool_call items sent back by the
// client convert into assistant tool calls + tool results for the model.
func TestCustomToolHistory(t *testing.T) {
	req := ResponsesRequest{
		Model: "test-model",
		Input: ResponsesInput{Items: []ResponsesInputItem{
			ResponsesInputMessage{Role: "user", Content: []ResponsesContent{ResponsesTextContent{Type: "input_text", Text: "edit"}}},
			ResponsesCustomToolCall{ID: "item_1", Type: "custom_tool_call", CallID: "call_1", Name: "apply_patch", Input: "patch-body"},
			ResponsesCustomToolCallOutput{Type: "custom_tool_call_output", CallID: "call_1", Output: "ok"},
		}},
		Tools: []ResponsesTool{{Type: "custom", Name: "apply_patch", Description: ptrStr("applies patches")}},
	}
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
	if assistant.ToolCalls[0].Function.Name != "apply_patch" {
		t.Errorf("expected apply_patch tool call")
	}
	if v, ok := assistant.ToolCalls[0].Function.Arguments.Get("input"); !ok || v != "patch-body" {
		t.Errorf("expected input arg 'patch-body', got %v ok=%v", v, ok)
	}
	tool := chatReq.Messages[2]
	if tool.Role != "tool" || tool.Content != "ok" || tool.ToolCallID != "call_1" {
		t.Errorf("unexpected tool result message: role=%q content=%q call_id=%q", tool.Role, tool.Content, tool.ToolCallID)
	}
}
