# from modules.wordcount import count_words
# from modules.emotion import detect_emotion
# from modules.groq_inference import get_groq_inference

# def analyze_passage(passage):
#     print("Analyzing passage...\n")
#     total_words = count_words(passage)
#     emotion, confidence = detect_emotion(passage)
#     books = get_groq_inference(passage, mode="book")
#     summary = get_groq_inference(passage, mode="summary")

#     result = {
#         "Total Words": total_words,
#         "Emotion": f"{emotion} ({confidence:.2f})",
#         "Possible Books": books,
#         "Summary": summary
#     }
#     return result

# if __name__ == "__main__":
#     import pprint
#     passage = input("Enter passage text:\n")
#     results = analyze_passage(passage)
#     print(results)
# Import necessary modules
from transformers import pipeline  # For emotion detection using a pre-trained transformer model
from groq import Groq              # GROQ API client for accessing LLaMA-3 model
import json, os                    # Standard libraries for JSON handling and environment variable access

import re
import re

def count_words(passage: str) -> int:
    """
    Count the number of words in a given text passage.

    This function uses a regular expression to identify and count
    word-like tokens, ignoring punctuation and extra whitespace.

    Args:
        passage (str): The input text passage to analyze.

    Returns:
        int: The total number of words found in the passage.
    """
    words = re.findall(r'\b\w+\b', passage) # using default .split is better but it fails with triple white spaces
    return len(words)

# Function to detect the main emotion in the text using a pre-trained emotion classification model
def detect_emotion(passage: str):
    # Create a Hugging Face pipeline for emotion classification
    emotion_pipeline = pipeline(
        "text-classification",  # Task type
        model="j-hartmann/emotion-english-distilroberta-base",  # Pre-trained model for emotion detection
        return_all_scores=False  # Only return the most likely emotion label
    )

    # The model has a 512-token limit, so we truncate the input to 512 characters
    result = emotion_pipeline(passage[:512])[0]

    # Return the predicted emotion label and its confidence score
    return result["label"], result["score"]

# Create a client instance for interacting with the GROQ LLaMA-3 API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  #   GROQ_API_KEY is set in environment variables


# Function to perform inference using GROQ's LLaMA-3 model
def get_groq_inference(passage: str, mode="book"):
    # Construct the chat messages for the model
    messages = [
        {"role": "system", "content": "You are a literary analysis assistant."},  # System prompt
        {"role": "user", "content": passage}  # User input: the actual passage
    ]

    model = "llama-3.3-70b-versatile"  # GROQ model to use

    # Decide the system prompt based on the mode (book or summary)
    if mode == "book":
        # If mode is 'book', ask the model to identify possible book sources
        system_prompt = """You are a literary analysis assistant. Based on the following passage, identify 2-3 possible books it could have come from.

                        Limit your guesses to well-known literary works with emotional or philosophical themes. Some examples include:
                        - The Alchemist by Paulo Coelho
                        - Man’s Search for Meaning by Viktor Frankl
                        - To Kill a Mockingbird by Harper Lee

                        Respond only with a JSON object:
                        {"possible_books": ["Book1", "Book2", "Book3"]}"""

    else:
        # If mode is 'summary', ask the model to summarize the passage
        system_prompt = "Summarize the following literary passage in 2-3 sentences. Focus on its emotional tone, theme, and key message."

    # Perform chat completion (inference) using GROQ API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},  # Add the prompt about what the model should do
            *messages  # Add the conversation messages
        ],
        response_format={"type": "json_object" if mode == "book" else "text"}  # Decide response format
    )

    content = response.choices[0].message.content  # Get the text content of the model's response

    # Return parsed JSON (if book mode) or plain text (if summary)
    return json.loads(content) if mode == "book" else content

# Main function to run full analysis on a passage
def analyze_passage(passage):
    print("Analyzing passage...\n")

    # Step 1: Count the total number of words in the passage
    total_words = count_words(passage)

    # Step 2: Detect the dominant emotion in the passage
    emotion, confidence = detect_emotion(passage)

    # Step 3: Identify possible books that the passage could belong to
    books = get_groq_inference(passage, mode="book")

    # Step 4: Generate a short summary of the passage
    summary = get_groq_inference(passage, mode="summary")

    # Collect and return all results in a dictionary
    result = {
        "Total Words": total_words,
        "Emotion": f"{emotion} ({confidence:.2f})",  # Format score to 2 decimal places
        "Possible Books": books,
        "Summary": summary
    }
    return result

# Main entry point for the script
if __name__ == "__main__":
    # Prompt user for passage input
    passage = input("Enter passage text:\n")

    # Run analysis and print results
    results = analyze_passage(passage)
    print(results)
