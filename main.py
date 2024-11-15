from fastapi import FastAPI

import contextualization
import goal_model_generation
import middle_goal
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Knowledge Graph for Goal Model Generation")

app.include_router(contextualization.router)
app.include_router(goal_model_generation.router)
app.include_router(middle_goal.router)

# TODO: goal relaxation
