from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import  OAuth2PasswordRequestForm

from fastapi.middleware.cors import CORSMiddleware

import json
import chess


from chess_engine.src.model.classes.move_picker import move_picker
from chess_engine.src.model.config.config import Settings
import uvicorn









# FastAPI instance
app = FastAPI()

# OAuth2

mp = move_picker()

@app.post("/aimove")
async def login(fen:str):
    board = chess.Board(fen=fen)
    move = mp.full_engine_get_move(board=board)

    return {"move" : str(move)}




# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
