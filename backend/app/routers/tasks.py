"""Task status for the frontend. Tasks are executed by services/dispatcher.py, which pushes them to the GPU server."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Edit, Task, User, utcnow
from ..schemas import TaskOut, task_out
from ..security import get_current_user, get_owned_project
from ..services import dispatcher

router = APIRouter(tags=["tasks"])


def _get_task(db: Session, task_id: int, user: User) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    get_owned_project(db, task.project_id, user)
    return task


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return task_out(_get_task(db, task_id, user))


@router.post("/tasks/{task_id}/cancel", response_model=TaskOut)
def cancel_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = _get_task(db, task_id, user)
    if task.status in ("pending", "running"):
        was_pending = task.status == "pending"
        task.status, task.finished_at, task.detail = "cancelled", utcnow(), None
        if was_pending and task.edit_id:  # never started: put the edit back where it was
            edit = db.get(Edit, task.edit_id)
            revert = dispatcher.REVERT_STATE.get(task.type)
            if edit and revert:
                edit.state = revert
        db.commit()
        dispatcher.wake()
    return task_out(task)
