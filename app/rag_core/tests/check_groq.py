from groq import Groq
from config import GROQ_API_KEY


client = Groq(api_key=GROQ_API_KEY)

models = client.models.list()

print("\nAvailable Groq models:\n")

for model in models.data:
    print(model.id)