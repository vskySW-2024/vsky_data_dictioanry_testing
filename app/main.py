from fastapi import FastAPI
from app.api import routes
import uvicorn

app = FastAPI()

## Include the endpoints
app.include_router(routes.router)

@app.get("/")
def index():
    return "Welcome to the Fuzzy Matcher API dashboard! "