from fastapi import Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

from typing import List, Optional

from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from database import SessionLocal, engine
import models

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


@router.get("/middle_goal/{high_level_goal_id}")
async def middle_goal(request: Request, high_level_goal_id: int, refinement_type: str = "AND",  db: Session = Depends(get_db)):
    goal_with_outputs = db.query(models.Goal).filter(models.Goal.id == high_level_goal_id).first()

    if not goal_with_outputs:
        return RedirectResponse("/")

    highlevelgoal = goal_with_outputs.goal_name

    # Get all current subgoals linked to the high-level goal
    subgoals = db.query(models.Goal).join(models.Hierarchy, models.Goal.id == models.Hierarchy.subgoal_id).filter(
        models.Hierarchy.high_level_goal_id == high_level_goal_id,
        models.Hierarchy.refinement == refinement_type
    ).all()

    return templates.TemplateResponse('middle_goal.html', context={
        'request': request,
        'hlg_id': high_level_goal_id,
        'refinement_type': refinement_type,
        'highlevelgoal': highlevelgoal,
        'subgoals': subgoals  # Pass the subgoals to the template
    })


@router.post("/middle_goal")
async def middle_goal(request: Request, goal_name: str = Form(...), goal_type: str = Form(...), hlg_id: int = Form(...),
                      refinement_type: str = Form(...), subgoal_id: List[str] = Form([]), db: Session = Depends(get_db)):
    if hlg_id == -1:
        return RedirectResponse(f"/goal_model_generation", status_code=302)

    # Create the new mid-goal (MG)
    new_middle_goal = models.Goal(goal_type=goal_type, goal_name=goal_name)
    db.add(new_middle_goal)
    db.commit()
    db.refresh(new_middle_goal)

    # If a subgoal (SG) is provided, link the MG between the high-level goal (HLG) and the SG
    for sg in subgoal_id:
        # --> Update the goal hierarchy <--
        # Break the current hierarchy between the HLG and the SG

        # Remove any existing relationships between the HLG and the SGs
        old_hierarchy = db.query(models.Hierarchy).filter(
            models.Hierarchy.high_level_goal_id == hlg_id,
            models.Hierarchy.subgoal_id == sg,
            models.Hierarchy.refinement == refinement_type
        ).first()

        if old_hierarchy:
            db.delete(old_hierarchy)
            db.commit()

        # Check if the new hierarchy (HLG -> MG) already exists to avoid redundancy
        existing_hierarchy_hlg = db.query(models.Hierarchy).filter(
            models.Hierarchy.high_level_goal_id == hlg_id,
            models.Hierarchy.subgoal_id == new_middle_goal.id,
            models.Hierarchy.refinement == refinement_type
        ).first()

        if not existing_hierarchy_hlg:
            # Insert the new relationship (HLG -> MG)
            new_hierarchy_hlg = models.Hierarchy(
                refinement=refinement_type,
                high_level_goal_id=hlg_id,
                subgoal_id=new_middle_goal.id
            )
            db.add(new_hierarchy_hlg)

        # Check if the new hierarchy (MG -> SG) already exists to avoid redundancy
        existing_hierarchy_sub = db.query(models.Hierarchy).filter(
            models.Hierarchy.high_level_goal_id == new_middle_goal.id,
            models.Hierarchy.subgoal_id == sg,
            models.Hierarchy.refinement == refinement_type
        ).first()

        if not existing_hierarchy_sub:
            # Insert the new relationship (MG -> SG)
            new_hierarchy_sub = models.Hierarchy(
                refinement=refinement_type,
                high_level_goal_id=new_middle_goal.id,
                subgoal_id=sg
            )
            db.add(new_hierarchy_sub)

    # Commit all changes after the loop
    db.commit()

    return RedirectResponse(f"/goal_model_generation", status_code=302)

