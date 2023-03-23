import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import router as nlp_router

app = FastAPI()

origins = [
  "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return {"message":"welcome to Auto-Utility Backend !"}

app.include_router(nlp_router, tags=["nlp"], prefix="/nlp")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        reload = True
    )
