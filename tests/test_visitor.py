"""Exercise the visitor callbacks with speech and retrieval providers replaced."""

from src.app import gradio_app as app


def test_recordings_share_history_and_sources_across_tabs(monkeypatch):
    visit = app.Visit(active=True)
    questions = {"first.wav": "Where did you live?", "next.wav": "Who lived there?"}
    monkeypatch.setattr(app, "transcribe", questions.__getitem__)
    calls = []

    def answer(question, history):
        calls.append((question, history.copy()))
        return f"Answer {len(calls)}", f"Record {len(calls)}"

    monkeypatch.setattr(app, "generate_response", answer)
    first = list(app.ask_voice("first.wav", False, visit))[-1]
    second = list(app.ask_avatar("next.wav", False, visit))
    first_turn = (questions["first.wav"], "Answer 1")
    assert calls == [(questions["first.wav"], []), (questions["next.wav"], [first_turn])]
    expected = [
        {"role": "user", "content": questions["first.wav"]},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": questions["next.wav"]},
        {"role": "assistant", "content": "Answer 2"},
    ]
    for panel in (app.conversation, app.avatar):
        assert first[panel.chat] == second[0][panel.chat] == expected[:2]
        assert first[panel.sources] == second[0][panel.sources] == "Record 1"
        assert second[-1][panel.chat] == expected
        assert second[-1][panel.sources] == "Record 2"


def test_speech_failure_preserves_text_and_sources(monkeypatch):
    visit = app.Visit(active=True)
    monkeypatch.setattr(app, "generate_response", lambda *_: ("At the bakery.", "Interview 4"))

    def unavailable(text, *, cancel):
        raise app.SpeechError("Voice service unavailable.")

    monkeypatch.setattr(app, "synthesize", unavailable)
    result = list(app.ask_text("Where did you work?", True, visit))[-1]
    assert visit.history == [("Where did you work?", "At the bakery.")]
    for panel in (app.conversation, app.avatar):
        assert result[panel.chat][-1] == {"role": "assistant", "content": "At the bakery."}
        assert result[panel.sources] == "Interview 4"
        assert result[panel.player] is None
        assert result[panel.status] == "Voice service unavailable."


def test_end_visit_discards_inflight_reply(monkeypatch):
    visit = app.Visit(active=True)
    ended = {}

    def end_during_search(question, history):
        ended.update(app.end_visit(visit))
        return "A late reply", "A late source"

    monkeypatch.setattr(app, "generate_response", end_during_search)
    turn = app.ask_text("What happened?", False, visit)
    pending = next(turn)
    assert pending[app.conversation.chat][-1] == {"role": "user", "content": "What happened?"}
    assert list(turn) == []
    assert not visit.active and visit.history == [] and visit.sources == ""
    for panel in (app.conversation, app.avatar):
        assert ended[panel.chat] == []
        assert ended[panel.sources] == ""
        assert ended[panel.player] is None
