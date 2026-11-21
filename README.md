# AI Ancestor Interaction

Talk with the AI memorial built from Walter Bennett’s fictional life-archive interview. The visitor interface accepts text and recorded questions, shows the interview records retrieved for each reply, and reads answers aloud with a stock voice.

## Start a visit on this Mac

Double-click **Start Visitor App.command** to open the app. The prepared copy uses its installed Python environment and reads the provider keys from the original backend’s local `.env` file. Keep that file in place or update `AFTERLIFE_ENV_FILE` in this copy’s `.env`.

1. Open the local visitor app at [127.0.0.1:7861](http://127.0.0.1:7861). Enter a display name, select **Visitor** or a relationship, enter the demo code `ANCESTOR2025`, and acknowledge the memorial disclosure.
2. Choose **Begin conversation**. Try **What was your very first job?** Then ask **What did you learn there?**
3. For a spoken question, select the microphone, allow access in your browser, record, stop, and press **Send voice question**. You can also upload a WAV recording. Your transcript appears first, followed by the reply, its sources, and the audio controls.
4. Expand **Memory sources for the latest reply** to inspect the retrieved passages. Use **Settings & Privacy** to toggle spoken replies, **Stop reply** to interrupt a turn, or **End visit** to clear the conversation.

The Avatar tab offers an audio-only visit with the same conversation history. The app binds to the loopback address on your computer. If your embedded browser has no microphone, open the same URL in Safari or Chrome.

## Set up another checkout

Use Python 3.11. From the project directory, run the following commands on macOS or Linux.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m src.app.gradio_app
```

New checkouts use port 7860 by default. Set `GRADIO_SERVER_PORT` in `.env` to select a different port. On Windows, create the environment with `py -3.11 -m venv .venv`, activate it with `.venv\Scripts\activate`, and use `copy .env.example .env`. On Linux, the system-voice fallback uses eSpeak NG. For uploads in formats other than WAV, install FFmpeg or convert the recording to WAV first.

## Configure the services

Set keys on the server in `.env`, or set `AFTERLIFE_ENV_FILE` to an existing backend environment file. Environment variables set in the shell take precedence. Restart after changing settings.

| Setting | What it enables |
| --- | --- |
| `OPENAI_API_KEY` | Generates an answer from retrieved records and uses recent turns to interpret follow-up questions. |
| `PINECONE_API_KEY` | Adds semantic search of the existing Pinecone index to local BM25 keyword search. Defaults are `ancestor-life-story` and namespace `person-001`. |
| `DEEPGRAM_API_KEY` | Transcribes submitted recordings using Deepgram Nova-3. Typed questions work independently. |
| `ELEVENLABS_API_KEY` | Reads replies aloud with the configured stock voice. The default `TTS_MODE=auto` can fall back to local speech synthesis. |
| `ANCESTOR_ACCESS_CODE` | Replaces the shared local demo code. |
| `RAG_PDF` | Selects the interview file. The bundled `questions.pdf` contains Walter’s fictional interview. |

`RAG_MODE=auto` uses the configured cloud services and falls back to verbatim interview excerpts if generation is unavailable. Set `RAG_MODE=local` for excerpt-only answers, or `RAG_MODE=cloud` to require the configured hybrid pipeline. The sources panel identifies how a reply was produced. For a fully local typed visit, set `RAG_MODE=local` and `TTS_MODE=offline`. Speech input uses Deepgram when submitted.

For another person’s interview, supply its PDF and set `RAG_INDEX` and `RAG_NAMESPACE` to the matching, pre-indexed archive. The visitor app reads the index. The separate backend indexing CLI can rebuild namespaces, so use a dedicated namespace when preparing a different archive.

The browser processes one submitted recording at a time. It sends audio to Deepgram, sends the question for retrieval and answer generation, and sends generated text to ElevenLabs when cloud speech is enabled. Browser audio files pass through Gradio’s temporary cache. The Gradio launcher disables run history, and each visitor’s access state and chat history live in that browser session.

## Source projects

The integration combines the visitor screens from [AI-Ancestor-Interaction](https://github.com/vm-zx/AI-Ancestor-Interaction) with the retrieval pipeline and team speech engine from [ai-afterlife](https://github.com/az-tayar/ai-afterlife). The original command-line RAG and voice-chat entry points remain available.
