class ConversationMemory:

    def __init__(self):

        self.history = []

    def add_user_message(self, message: str):

        self.history.append({
            "role": "user",
            "content": message
        })

    def add_assistant_message(self, message: str):

        self.history.append({
            "role": "assistant",
            "content": message
        })

    def get_history(self):

        return self.history

    def clear(self):

        self.history = []