import type { UIMessage } from "ai";
import { ToolInvocation } from "./ToolInvocation";

// Helper functions to extract content from AI SDK v5 messages
function getTextFromMessage(message: UIMessage): string {
  if (!message.parts || !Array.isArray(message.parts)) {
    return '';
  }
  
  return message.parts
    .filter(part => part.type === 'text')
    .map(part => part.text)
    .join(' ');
}

function getToolsFromMessage(message: UIMessage) {
  if (!message.parts || !Array.isArray(message.parts)) {
    return [];
  }
  
  return message.parts.filter(part => 
    part.type && typeof part.type === "string" && 
    (part.type.startsWith('tool-') || part.type === 'dynamic-tool')
  );
}

interface MessageProps {
  message: UIMessage;
}

export const Message = ({ message }: MessageProps) => {
  const text = getTextFromMessage(message);
  const tools = getToolsFromMessage(message);

  return (
    <div style={{
      margin: "16px 0",
      padding: "12px",
      borderRadius: "8px",
      backgroundColor: message.role === "user" ? "#e3f2fd" : "#f5f5f5"
    }}>
      <div style={{
        fontWeight: "bold",
        marginBottom: "8px",
        color: message.role === "user" ? "#1976d2" : "#666"
      }}>
        {message.role === "user" ? "👤 You" : "🤖 Assistant"}
      </div>
      
      {tools.length > 0 && (
        <div style={{ marginBottom: text ? "12px" : "0" }}>
          <div style={{ 
            fontSize: "12px", 
            fontWeight: "bold", 
            color: "#666", 
            marginBottom: "8px",
            textTransform: "uppercase"
          }}>
            Tools Used
          </div>
          {tools.map((tool, idx) => {
            const key = `tool-${idx}`;
            return (
              <ToolInvocation
                key={key}
                invocation={tool as any}
              />
            );
          })}
        </div>
      )}
      
      {text && (
        <div>
          {text}
        </div>
      )}
    </div>
  );
};