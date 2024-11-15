from fastapi import Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from sqlalchemy.orm import aliased

from transformers.utils import logging
import torch
import os

import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
logging.set_verbosity_error()

# Templates (Jinja2)
templates = Jinja2Templates(directory="templates/")

# Router
router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


models.Base.metadata.create_all(bind=engine)



def process_goal_hierarchy(db: Session):
    # Fetch all goals
    all_goals = db.query(models.Goal).all()

    # Fetch all hierarchies
    hierarchies = db.query(models.Hierarchy).all()

    # Fetch all triple filtered data
    triple_filtered_data = db.query(models.Triple_Filtered).all()

    # Map of goal id to goal object
    goal_map = {goal.id: goal for goal in all_goals}

    # Map of goal id to its corresponding triple filtered data
    #triple_filtered_map = {}
    #for triple in triple_filtered_data:
    #    print("\nTriple Filtered:")
    #    print(triple)
    #    if triple.subgoal_id in triple_filtered_map:
    #        triple_filtered_map[triple.subgoal_id].append(triple.triple_filtered_from_hlg)
    #        print(triple_filtered_map)
    #    else:
    #        triple_filtered_map[triple.subgoal_id] = [triple.triple_filtered_from_hlg]

    #print("\nMAP:")
    #print(triple_filtered_map)

    # Process and map the goals with their hierarchies
    hierarchy_data = []

    for hierarchy in hierarchies:
        subgoal = goal_map[hierarchy.subgoal_id]
        high_level_goal = goal_map.get(hierarchy.high_level_goal_id)  # Can be None

        # Get the triple filtered data for this subgoal, if any
        #triples_for_subgoal = triple_filtered_map.get(subgoal.id, [])

        #print("\ntriples_for_subgoal:")
        #print(triples_for_subgoal)


        hierarchy_data.append({
            'hierarchy_id': hierarchy.id,
            'subgoal_id': subgoal.id,
            'subgoal_name': subgoal.goal_name,
            'subgoal_goal_type': subgoal.goal_type,
            'high_level_goal_id': high_level_goal.id if high_level_goal else None,
            'high_level_goal_name': high_level_goal.goal_name if high_level_goal else None,
            'high_level_goal_goal_type': high_level_goal.goal_type if high_level_goal else None,
            'refinement': hierarchy.refinement,
            #'triple_filtered': json.dumps(triples_for_subgoal) if triples_for_subgoal else None  # Add triple data
        })

    print("\nHIERARCHY DATA:")
    print(hierarchy_data)

    # Add single goals with no hierarchy
    for goal in all_goals:
        if not any(goal.id == h['subgoal_id'] or goal.id == h['high_level_goal_id'] for h in hierarchy_data):
            #triples_for_goal = triple_filtered_map.get(goal.id, [])

            hierarchy_data.append({
                'hierarchy_id': None,
                'subgoal_id': goal.id,
                'subgoal_name': goal.goal_name,
                'subgoal_goal_type': goal.goal_type,
                'high_level_goal_id': None,
                'high_level_goal_name': None,
                'high_level_goal_goal_type': None,
                'refinement': None,
                #'triple_filtered': json.dumps(triples_for_goal) if triples_for_goal else None  # Add triple data
            })

    return hierarchy_data



@router.get("/goal_model_generation")
async def goal_model_generation(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse('goal_model_generation.html', context={'request': request, 'hierarchy_data': process_goal_hierarchy(db)})



@router.post("/goal_model_generation")
async def deleteGoal(request: Request, subgoal_id: int = Form(...), db: Session = Depends(get_db)):
    # Log
    print("Subgoal ID to delete:", subgoal_id)

    # Query the database to find the subgoal
    subgoal = db.query(models.Goal).filter(models.Goal.id == subgoal_id).first()

    if subgoal:
        # Delete the subgoal and all related hierarchies
        db.query(models.Hierarchy).filter(
            (models.Hierarchy.subgoal_id == subgoal_id) | (models.Hierarchy.high_level_goal_id == subgoal_id)).delete()
        db.delete(subgoal)
        db.commit()

        print("The subgoal ID " + str(subgoal_id) + " is deleted!")

    return templates.TemplateResponse('goal_model_generation.html',
                                      context={'request': request, 'hierarchy_data': process_goal_hierarchy(db)})