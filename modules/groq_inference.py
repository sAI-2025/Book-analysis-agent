from groq import Groq
import json, os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_groq_inference(passage: str, mode="book"):
    messages=[
        {"role": "system", "content": "You are a literary analysis assistant."},
        {"role": "user", "content": passage}
    ]
    model = "llama-3.3-70b-versatile"

    if mode == "book":
        system_prompt = """Identify 2-3 possible books this passage might belong to.
        Respond only with a JSON object containing {"possible_books": ["Book1", "Book2"]}."""
    else:
        system_prompt = """Summarize this passage in 2-3 meaningful sentences."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages
        ],
        response_format={"type": "json_object" if mode == "book" else "text"}
    )

    content = response.choices[0].message.content
    return json.loads(content) if mode == "book" else content
