from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..quiz import engine, scoring, sessions
from ..quiz.word_provider import get_provider
from ..storage import db
from .schemas import AnswerRequest, StartRequest

router = APIRouter(prefix="/api")

provider = get_provider()


@router.post("/session/start")
def start_session(req: StartRequest):
    session = sessions.store.create(req.mode, req.k)
    q = engine.next_question(session, provider)
    session.current = q
    return {"session_id": session.id, "question": q.public()}


@router.post("/session/answer")
def answer(req: AnswerRequest):
    session = sessions.store.get(req.session_id)
    if session is None:
        raise HTTPException(404, "сессия не найдена")
    if session.finished:
        return {"finished": True, "result": session.result}
    q = session.current
    if q is None or q.question_id != req.question_id:
        raise HTTPException(409, "вопрос не совпадает с текущим")

    result = engine.grade(q, req.answer)

    if result["is_fake"]:
        session.fake_total += 1
        if result["caught_lie"]:
            session.fake_caught += 1

    engine.apply_result(session, result)
    session.index += 1
    db.save_answer(session.id, session.index, q, result, req.answer, sessions.now())

    if session.index >= session.k:
        session.finished = True
        session.result = scoring.compute_result(session, db.percentile_pool())
        session.current = None
        db.save_session(session, session.result, sessions.now())
        return {"finished": True, "result": session.result}

    nxt = engine.next_question(session, provider)
    session.current = nxt
    return {"finished": False, "question": nxt.public()}


@router.get("/session/{session_id}/result")
def get_result(session_id: str):
    session = sessions.store.get(session_id)
    if session is None or not session.finished:
        raise HTTPException(404, "результат недоступен")
    return session.result


@router.get("/admin/stats")
def admin_stats():
    return db.admin_stats()
