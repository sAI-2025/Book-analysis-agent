import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest
from modules.wordcount import count_words
from modules.emotion import detect_emotion
from modules.groq_inference import get_groq_inference

def analyze_passage(passage):
    total_words = count_words(passage)
    emotion, confidence = "Good",96 #detect_emotion(passage)
    books = get_groq_inference(passage, mode="book")
    summary = get_groq_inference(passage, mode="summary")

    result = {
        "total_words": total_words,
        "emotion": f"{emotion} ({confidence:.2f})",
        "possible_books": books,
        "summary": summary
    }
    return result

from django.shortcuts import render

def chat_view(request):
    return render(request, 'chat/chat.html')

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            passage = data.get("passage", "")
            if passage.strip() == "":
                return HttpResponseBadRequest("Passage text is empty.")
            analysis = analyze_passage(passage)
            return JsonResponse({"success": True, "data": analysis})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)
    else:
        return HttpResponseBadRequest("Only POST method allowed.")
