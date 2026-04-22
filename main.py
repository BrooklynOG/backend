from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag import load_data, query_data
from intent import detect_intent
from llm import generate_response
from tasks import add_expense, add_income, get_summary

app = FastAPI()

# Essential for React deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

load_data()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/chat")
def chat(query: str):
    intent = detect_intent(query)
    if intent == "expense":
        try:
            amount = int(query.split()[-1])
            return {"response": add_expense(amount)}
        except:
            return {"response": "Please use format: add expense 500"}
    elif intent == "income":
        try:
            amount = int(query.split()[-1])
            return {"response": add_income(amount)}
        except:
            return {"response": "Please use format: add income 2000"}
    else:
        context = query_data(query)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        answer = generate_response(prompt)
        return {"response": answer}

@app.get("/summary")
def summary():
    return get_summary()
