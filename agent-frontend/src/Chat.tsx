import { useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { Message } from './Message';

export function Chat() {
  const [input, setInput] = useState('');
  
  const {
    messages,
    sendMessage,
    status,
  } = useChat({
    transport: new DefaultChatTransport({
      api: 'http://localhost:8000/agent',
    }),
  });

  const isLoading = status === 'streaming' || status === 'submitted';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    
    sendMessage({ text: input });
    setInput('');
  };

  return (
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'system-ui, sans-serif'
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>
        🧮 Math Agent Chat
      </h1>
      
      <div style={{
        height: '500px',
        overflowY: 'auto',
        border: '1px solid #ccc',
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
        backgroundColor: '#fafafa'
      }}>
        {messages.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            color: '#666',
            padding: '40px 0'
          }}>
            👋 Hi! I'm a math assistant with tools. Try asking me to add some numbers!
            <br /><br />
            For example: "What's 15 + 27 + 8?"
          </div>
        )}
        
        {messages.map((message) => (
          <Message
            key={message.id}
            message={message}
          />
        ))}
        
        {isLoading && (
          <div style={{
            padding: '12px',
            color: '#666',
            fontStyle: 'italic'
          }}>
            🤖 Thinking...
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me to do some math..."
          disabled={isLoading}
          style={{
            flex: 1,
            padding: '12px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            fontSize: '16px'
          }}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          style={{
            padding: '12px 24px',
            backgroundColor: isLoading || !input.trim() ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
            fontSize: '16px'
          }}
        >
          {isLoading ? '⏳' : 'Send'}
        </button>
      </form>
      
      <div style={{
        marginTop: '16px',
        fontSize: '14px',
        color: '#666',
        textAlign: 'center'
      }}>
        Powered by AI SDK v5 + Pydantic-AI
      </div>
    </div>
  );
}