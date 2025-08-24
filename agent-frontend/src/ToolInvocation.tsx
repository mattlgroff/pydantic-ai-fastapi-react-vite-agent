import { useState } from "react";

interface ToolInvocationProps {
  invocation: {
    state?: string;
    toolCallId?: string;
    toolName?: string;
    input?: unknown;
    output?: unknown;
    errorText?: string;
  };
}

export const ToolInvocation = ({ invocation }: ToolInvocationProps) => {
  const { toolName, input, output, errorText } = invocation;
  const [isOpen, setIsOpen] = useState(false);

  const hasInput = input !== undefined && input !== null;
  const hasOutput = output !== undefined && output !== null;
  const hasError = errorText !== undefined && errorText !== null;

  const prettyJson = (data: unknown) =>
    typeof data === "string" ? data : JSON.stringify(data, null, 2);

  return (
    <div style={{
      border: "1px solid #ccc",
      borderRadius: "8px",
      padding: "12px",
      margin: "8px 0",
      backgroundColor: "#f9f9f9",
      fontFamily: "monospace",
      fontSize: "12px"
    }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "none",
          border: "none",
          cursor: "pointer",
          fontSize: "14px",
          fontWeight: "bold"
        }}
      >
        <span>🛠️ {toolName || "Tool"}</span>
        <span>{isOpen ? "▲" : "▼"}</span>
      </button>

      {isOpen && (
        <div style={{ marginTop: "8px" }}>
          {hasInput && (
            <div style={{ marginBottom: "8px" }}>
              <div style={{ fontWeight: "bold", marginBottom: "4px" }}>Input:</div>
              <pre style={{ 
                whiteSpace: "pre-wrap", 
                wordBreak: "break-word",
                background: "#fff",
                padding: "8px",
                borderRadius: "4px",
                border: "1px solid #ddd"
              }}>
                {prettyJson(input)}
              </pre>
            </div>
          )}

          {hasOutput && (
            <div style={{ marginBottom: "8px" }}>
              <div style={{ fontWeight: "bold", marginBottom: "4px" }}>Output:</div>
              <pre style={{ 
                whiteSpace: "pre-wrap", 
                wordBreak: "break-word",
                background: "#fff",
                padding: "8px",
                borderRadius: "4px",
                border: "1px solid #ddd"
              }}>
                {prettyJson(output)}
              </pre>
            </div>
          )}

          {hasError && (
            <div>
              <div style={{ fontWeight: "bold", marginBottom: "4px", color: "red" }}>Error:</div>
              <pre style={{ 
                whiteSpace: "pre-wrap", 
                wordBreak: "break-word",
                background: "#ffe6e6",
                padding: "8px",
                borderRadius: "4px",
                border: "1px solid #ffcccc",
                color: "red"
              }}>
                {errorText}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};