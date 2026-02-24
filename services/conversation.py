from datetime import datetime

class ConversationManager:
    def __init__(self):
        self.conversations = {}
    
    def add_message(self, session_id, role, content, metadata=None):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self, session_id, limit=10):
        if session_id not in self.conversations:
            return []
        return self.conversations[session_id][-limit:]
    
    def get_context(self, session_id):
        history = self.get_history(session_id, limit=5)
        context = ""
        for msg in history:
            context += f"{msg['role']}: {msg['content']}\n"
        return context

conversation_manager = ConversationManager()
