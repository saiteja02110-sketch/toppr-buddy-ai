from typing import List
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ai import ask_ai
from database import save_chat, get_chat_history
app = FastAPI(title="Topper Buddy AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {
        "message": "Topper Buddy AI Backend Running"
    }
@app.post("/chat")
async def chat(
    message: str = Form(...),
    tool: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    print("========== NEW CHAT REQUEST ==========")
    final_message = message.strip()
    print("1. Message:", final_message)
    # Read uploaded files
    if files:
        file_contents = []
        for file in files:
            try:
                content = await file.read()
                try:
                    text = content.decode("utf-8", errors="ignore")
                except Exception:
                    text = f"[Binary file uploaded: {file.filename}]"
                file_contents.append(
                    f"\nFilename: {file.filename}\n{text}\n"
                )
            except Exception:
                file_contents.append(
                    f"[Unable to read file: {file.filename}]"
                )
        final_message += "\n\nAttached Files:\n"
        final_message += "\n\n".join(file_contents)
    print("2. Files processed")
    user_id = 5
    print("3. Loading chat history...")
    history = get_chat_history(user_id=user_id)
    print("History loaded.")
    print("4. Calling ask_ai()...")
    print("========== BEFORE AI ==========")
    print("Message:", final_message)
    print("Tool:", tool)
    answer = ask_ai(
        question=final_message,
        tool=tool,
        history=history,
        user_id=user_id
    )
    print("========== AFTER AI ==========")
    print(answer)
    print("6. Saving chat...")
    save_chat(
        user_id=user_id,
        tool=tool,
        user_message=final_message,
        ai_response=answer
    )
    print("7. Chat saved successfully.")
    print("========== REQUEST COMPLETE ==========")
    return {
        "response": answer
    }
 
