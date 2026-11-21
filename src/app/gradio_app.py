"""Local visitor interface for the AI Ancestor memory companion."""

from dataclasses import dataclass, field
import hmac
import os
from pathlib import Path
import sys
import threading

import gradio as gr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.settings import load_environment
from src.app.speech import SpeechError, synthesize, transcribe
from src.rag.chatbot import generate_response

load_environment()

RELATIONSHIPS = ["Grandchild", "Child", "Spouse", "Sibling", "Family Friend", "Co-Worker", "Visitor"]
DEFAULT_CODE = "ANCESTOR2025"
ACCESS_CODE = os.environ.get("ANCESTOR_ACCESS_CODE") or DEFAULT_CODE
STATE_LOCK = threading.Lock()


@dataclass
class Visit:
    """Server-side state belonging to one browser session."""

    active: bool = False
    revision: int = 0
    history: list[tuple[str, str]] = field(default_factory=list)
    sources: str = ""
    speech_cancel: threading.Event | None = None

    def _invalidate_turn(self):
        self.revision += 1
        if self.speech_cancel is not None:
            self.speech_cancel.set()
            self.speech_cancel = None

    def reset(self, active):
        with STATE_LOCK:
            self._invalidate_turn()
            self.active = active
            self.history.clear()
            self.sources = ""

    def begin_turn(self):
        with STATE_LOCK:
            self.require_access()
            self._invalidate_turn()
            self.speech_cancel = threading.Event()
            return self.revision, self.history.copy(), self.sources, self.speech_cancel

    def require_access(self):
        if not self.active:
            raise gr.Error("Open Access & Consent and verify your visit first.")

    def current(self, revision):
        with STATE_LOCK:
            return self.active and self.revision == revision

    def remember(self, revision, question, answer, sources):
        with STATE_LOCK:
            if not self.active or self.revision != revision:
                return False
            self.history.append((question, answer))
            self.sources = sources
            return True


@dataclass
class Panel:
    chat: gr.Chatbot
    recording: gr.Audio
    submit_recording: gr.Button
    player: gr.Audio
    sources: gr.Textbox
    status: gr.Markdown
    stop: gr.Button
    clear: gr.Button
    end: gr.Button


def conversation_panel(with_text):
    chat = gr.Chatbot(
        label="Conversation transcript", height=280, render_markdown=False,
        buttons=["copy"], elem_classes=["visit-transcript"],
        placeholder="Your questions and the companion’s replies will appear here.",
    )
    message = send = None
    if with_text:
        with gr.Row():
            message = gr.Textbox(
                label="Your question", placeholder="What was your very first job?", scale=5,
            )
            send = gr.Button("Send", variant="primary", scale=1)
    with gr.Row():
        with gr.Column():
            recording = gr.Audio(
                label="Your recording", sources=["microphone", "upload"],
                type="filepath", format="wav",
            )
            submit = gr.Button("Send voice question", variant="secondary" if with_text else "primary")
        with gr.Column():
            audio = gr.Audio(
                label="Spoken reply", autoplay=True, interactive=False, format="wav",
                elem_classes=["spoken-reply"],
            )
            status = gr.Markdown()
    with gr.Accordion("Memory sources for the latest reply", open=False):
        sources = gr.Textbox(label="Interview records", interactive=False, lines=4, max_lines=12)
    with gr.Row():
        stop = gr.Button("Stop reply", size="sm")
        clear = gr.Button("Clear conversation", size="sm")
        end = gr.Button("End visit", size="sm")
    return Panel(chat, recording, submit, audio, sources, status, stop, clear, end), message, send


def messages(history, pending=None):
    result = []
    for question, answer in history:
        result.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    if pending:
        result.append({"role": "user", "content": pending})
    return result


def display(history, *, pending=None, sources="", status="", player=None, audio=None):
    result = {}
    transcript = messages(history, pending)
    for panel in panels:
        result[panel.chat] = transcript
        result[panel.sources] = sources
        result[panel.status] = status
        result[panel.player] = audio if panel.player is player else None
    return result


def run_turn(visit, panel, message, recording, read_aloud):
    """Display the recorded question and grounded reply as each stage finishes."""
    visit.require_access()
    question = message.strip() if message else ""
    if not question and not recording:
        prompt = "Type a question first." if message is not None else "Record or upload a question first."
        yield {panel.status: prompt}
        return
    revision, history, previous_sources, cancel = visit.begin_turn()

    if recording:
        yield display(history, sources=previous_sources, status="Transcribing your recording…")
        if cancel.is_set():
            return
        try:
            question = transcribe(recording).strip()
        except Exception as error:
            if visit.current(revision):
                detail = str(error) if isinstance(error, SpeechError) else "Transcription failed. Try recording again or type your question."
                yield display(history, sources=previous_sources, status=detail)
            return
        if not visit.current(revision):
            return
        if not question:
            yield display(history, sources=previous_sources, status="No speech was detected. Try recording again or type your question.")
            return

    searching = display(history, pending=question, sources=previous_sources, status="Looking through the interview records…")
    if recording:
        searching[panel.recording] = None
    else:
        searching[text_input] = ""
    yield searching
    if cancel.is_set():
        return
    try:
        answer, sources = generate_response(question, history)
    except Exception:
        if visit.current(revision):
            yield display(history, pending=question, sources=previous_sources, status="The memory search could not finish. Please try again.")
        return
    if not visit.remember(revision, question, answer, sources):
        return
    history.append((question, answer))

    def completed(status="", audio=None):
        return display(history, sources=sources, status=status, audio=audio, player=panel.player)

    if not read_aloud:
        yield completed()
        return
    yield completed("Preparing the spoken reply…")
    if cancel.is_set():
        return
    try:
        audio = synthesize(answer, cancel=cancel)
    except Exception as error:
        if visit.current(revision):
            detail = str(error) if isinstance(error, SpeechError) else "Voice playback is unavailable. Your reply is in the transcript."
            yield completed(detail)
        return
    if visit.current(revision):
        yield completed(audio=audio)


def ask_text(message, read_aloud, visit):
    yield from run_turn(visit, conversation, message or "", None, read_aloud)


def ask_voice(recording, read_aloud, visit):
    yield from run_turn(visit, conversation, None, recording, read_aloud)


def ask_avatar(recording, read_aloud, visit):
    yield from run_turn(visit, avatar, None, recording, read_aloud)


def clear_display():
    return {**display([]), text_input: "", conversation.recording: None, avatar.recording: None}


def clear_conversation(visit):
    with STATE_LOCK:
        visit.require_access()
        visit._invalidate_turn()
        visit.history.clear()
        visit.sources = ""
    return clear_display()


def stop_reply(visit):
    with STATE_LOCK:
        visit.require_access()
        visit._invalidate_turn()
        return display(visit.history, sources=visit.sources)


def navigate(tab_id, active, finished=False):
    updates = {tab: gr.update(visible=active) for tab in protected_tabs}
    updates.update({
        access_tab: gr.update(visible=not active and not finished),
        end_tab: gr.update(visible=finished),
        tabs: gr.update(selected=tab_id),
    })
    return updates


def verify_access(name, relationship, access_code, consent, visit):
    if not name or not name.strip() or relationship not in RELATIONSHIPS:
        raise gr.Error("Enter your name and select your relationship to the person.")
    if not consent:
        raise gr.Error("Please acknowledge the memorial disclosure to enter.")
    if not hmac.compare_digest((access_code or "").encode(), ACCESS_CODE.encode()):
        raise gr.Error("That access code was not recognized.")
    visit.reset(True)
    return {
        **navigate("welcome", True), **clear_display(),
        code_input: "", welcome_name: f"Welcome, {name.strip()}",
    }


def end_visit(visit):
    visit.reset(False)
    return {
        **navigate("end_visit", False, finished=True), **clear_display(),
        name_input: "", relationship_input: None, code_input: "", consent_input: False,
        welcome_name: "", read_aloud_input: True,
    }


def new_visit(visit):
    return {**end_visit(visit), **navigate("access", False)}


CSS = """
.gradio-container { max-width: 1040px !important; margin: auto; }
.memorial-header { padding: clamp(1.4rem, 5vw, 3.5rem) 0 1.4rem; border-bottom: 1px solid #aabab1; }
.memorial-header h1 { font: 500 clamp(2.4rem, 6.5vw, 4.4rem)/1.03 Georgia, serif;
  letter-spacing: -.055em; margin: 0 0 .6rem; color: var(--body-text-color); }
.memorial-header p { max-width: 60ch; color: var(--body-text-color-subdued); margin: 0; }
.visit-transcript .icon-button-wrapper.top-panel { display: none; }
"""


with gr.Blocks(title="AI Ancestor Interaction", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
    visit_state = gr.State(Visit())
    gr.HTML(
        '<header class="memorial-header"><h1>AI Ancestor Interaction</h1>'
        '<p>An AI memorial built from recorded stories and family memories.</p></header>'
    )
    with gr.Tabs(selected="access") as tabs:
        with gr.Tab("Access & Consent", id="access") as access_tab:
            gr.Markdown("## Begin your visit\nEnter your visitor details to open the family interview.")
            with gr.Row():
                name_input = gr.Textbox(label="Your name")
                relationship_input = gr.Dropdown(RELATIONSHIPS, value=None, label="Relationship")
            code_input = gr.Textbox(
                label="Access code", type="password",
                info="Local demo access code: ANCESTOR2025" if ACCESS_CODE == DEFAULT_CODE else None,
            )
            consent_input = gr.Checkbox(
                label="I understand the replies come from an AI memorial and I have permission to explore these memories."
            )
            gr.Markdown(
                "Depending on the configured services, interview excerpts and questions may be processed by "
                "OpenAI and Pinecone, recordings by Deepgram, and reply text by ElevenLabs."
            )
            verify = gr.Button("Verify access", variant="primary")

        with gr.Tab("Home", id="welcome", visible=False) as home_tab:
            welcome_name = gr.Textbox(label="Visitor", interactive=False)
            gr.Markdown("## Explore the interview\nStart with a memory about family, childhood, or a first job.")
            with gr.Row():
                open_chat = gr.Button("Begin conversation", variant="primary")
                open_avatar = gr.Button("Begin voice visit")

        with gr.Tab("Conversation", id="conversation", visible=False) as chat_tab:
            gr.Markdown("## Conversation\nType a question or send a recording. Ask a follow-up to explore a memory in more detail.")
            conversation, text_input, send_text = conversation_panel(with_text=True)

        with gr.Tab("Avatar", id="avatar", visible=False) as avatar_tab:
            gr.Markdown(
                "## Voice visit\nRecord a question and press **Send voice question**. "
                "This audio-only view shares the transcript with Conversation."
            )
            avatar, _, _ = conversation_panel(with_text=False)

        with gr.Tab("Settings & Privacy", id="settings", visible=False) as settings_tab:
            gr.Markdown("## Playback")
            read_aloud_input = gr.Checkbox(value=True, label="Read replies aloud")
            gr.Markdown(
                "## About this visit\nThe assistant uses the interview records to answer. "
                "Open **Memory sources** beneath a reply to review the retrieved excerpts. "
                "Ending the visit clears its conversation history."
            )
            end_settings = gr.Button("End visit")

        with gr.Tab("End Visit", id="end_visit", visible=False) as end_tab:
            gr.Markdown("## Thank you for visiting\nYour conversation has been cleared.")
            restart = gr.Button("Start a new visit", variant="primary")

    panels = [conversation, avatar]
    protected_tabs = [home_tab, chat_tab, avatar_tab, settings_tab]
    page_outputs = [*protected_tabs, access_tab, end_tab, tabs]
    turn_outputs = [
        component for panel in panels
        for component in [panel.chat, panel.sources, panel.status, panel.player, panel.recording]
    ] + [text_input]
    session_outputs = [
        *page_outputs, *turn_outputs, name_input, relationship_input, code_input,
        consent_input, welcome_name, read_aloud_input,
    ]

    verify.click(
        verify_access, [name_input, relationship_input, code_input, consent_input, visit_state],
        [*page_outputs, *turn_outputs, code_input, welcome_name], api_name="verify_access", queue=False,
    )
    open_chat.click(lambda: gr.update(selected="conversation"), outputs=tabs, api_visibility="private", queue=False)
    open_avatar.click(lambda: gr.update(selected="avatar"), outputs=tabs, api_visibility="private", queue=False)
    # Pause the previous reply immediately, including while the browser is recording the next question.
    pause_audio_js = """(...values) => {
        document.querySelectorAll('.spoken-reply audio').forEach(audio => audio.pause());
        return values;
    }"""
    text_event = gr.on(
        [text_input.submit, send_text.click], ask_text, [text_input, read_aloud_input, visit_state],
        turn_outputs, api_name="ask_text", js=pause_audio_js, concurrency_limit=8,
    )
    voice_events = [
        panel.submit_recording.click(
            handler, [panel.recording, read_aloud_input, visit_state], turn_outputs,
            api_name=handler.__name__, js=pause_audio_js, concurrency_limit=8,
        )
        for panel, handler in zip(panels, (ask_voice, ask_avatar))
    ]
    turns = [text_event, *voice_events]
    gr.on(
        [conversation.stop.click, avatar.stop.click,
         conversation.recording.start_recording, avatar.recording.start_recording],
        stop_reply, [visit_state], turn_outputs, api_name="stop_reply",
        queue=False, cancels=turns, js=pause_audio_js,
    )
    gr.on(
        [conversation.clear.click, avatar.clear.click, conversation.chat.clear, avatar.chat.clear],
        clear_conversation, [visit_state], turn_outputs,
        api_name="clear_chat", queue=False, cancels=turns, js=pause_audio_js,
    )
    gr.on(
        [conversation.end.click, avatar.end.click, end_settings.click], end_visit, [visit_state],
        session_outputs,
        api_name="end_visit", queue=False, cancels=turns, js=pause_audio_js,
    )
    restart.click(new_visit, [visit_state], session_outputs, api_name="new_visit", queue=False)


def launch():
    demo.queue().launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("ANCESTOR_INBROWSER", "0").lower() in {"1", "true", "yes"},
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="stone", font=["Arial", "sans-serif"]),
        css=CSS, share=False, run_history=False, footer_links=[],
    )


if __name__ == "__main__":
    launch()
