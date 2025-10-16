from modules.wordcount import count_words
from modules.emotion import detect_emotion
from modules.groq_inference import get_groq_inference

def analyze_passage(passage):
    print("Analyzing passage...\n")
    total_words = count_words(passage)
    emotion, confidence = detect_emotion(passage)
    books = get_groq_inference(passage, mode="book")
    summary = get_groq_inference(passage, mode="summary")

    result = {
        "Total Words": total_words,
        "Emotion": f"{emotion} ({confidence:.2f})",
        "Possible Books": books,
        "Summary": summary
    }
    return result

if __name__ == "__main__":
    import pprint
    passage = input("Enter passage text:\n")
    results = analyze_passage(passage)
    print(results)
