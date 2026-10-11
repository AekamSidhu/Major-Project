# System Architecture

Component view of OpenMeet, an interview copilot for interviewers and
workplace teams with built-in cheating detection. Top to bottom: the two React webviews, the Tauri
desktop shell (`src-tauri`), the real-time engine, the library crates, and
everything outside the OpenMeet process. Names match the code
(`docs/ARCHITECTURE.md` §1, `docs/INTERNALS.md`).

Legend: solid arrow = in-process call or data hand-off · dashed arrow = IPC or
network (HTTP on loopback or to a remote host) · cylinder = data store.

```mermaid
flowchart TB
    subgraph Web["Webviews (React + TypeScript)"]
        Overlay["Overlay window<br/>src/overlay · live HUD"]
        Main["Main window<br/>src/main · onboarding, settings,<br/>privacy, history, diagnostics"]
    end

    subgraph Shell["src-tauri (Rust desktop shell)"]
        Cmd["Typed commands<br/>(capability-scoped per window)"]
        Sink["TauriSink<br/>MeetingEvent → webviews"]
        AppState["AppState<br/>settings · lifecycle · retention"]
        Capture["CaptureManager<br/>mic · system audio · screen sampler"]
        OverlayCtl["OverlayController<br/>position · click-through ·<br/>capture exclusion"]
        Secrets["SecretStore"]
        LocalAi["LocalAiManager<br/>starts / stops the sidecar"]
    end

    subgraph Engine["openmeet-engine (tokio)"]
        Trans["VAD transcription<br/>sessions"]
        Conv["Conversation loop<br/>detection · turns ·<br/>topics · claims"]
        Req["RequestManager<br/>slots · cancellation"]
        Assist["Assist flows<br/>follow-ups · ask ·<br/>refine · summary"]
        Integrity["Integrity analyzer<br/>AI-echo score · delivery ·<br/>reading gaze"]
        Ctx["Context engine<br/>window · summary · retrieval"]
        Screen["Screen worker<br/>OCR · vision · observations"]
        Persist["Persistence worker"]
    end

    subgraph Libs["Library crates"]
        Intel["openmeet-intel<br/>VAD · detection · prompts · export"]
        Providers["openmeet-providers<br/>HTTP adapters ·<br/>local-only enforcement"]
        Platform["openmeet-platform<br/>cpal · xcap · OCR ·<br/>capture privacy"]
        Store["openmeet-store<br/>SQLite · migrations · retention"]
    end

    subgraph Ext["Outside the OpenMeet process"]
        OS["Operating system<br/>audio devices · displays ·<br/>Windows.Media.Ocr"]
        Keychain["OS credential store"]
        Sidecar["local-ai/server.py<br/>faster-whisper · fastembed ·<br/>MediaPipe gaze (127.0.0.1)"]
        Ollama["Ollama<br/>(local LLM)"]
        Cloud["Cloud AI providers<br/>OpenAI · Anthropic · compatible<br/>(optional, blocked in local-only mode)"]
        Disk[("SQLite database")]
        Shots[("Kept screenshots<br/>PNG files, opt-in")]
    end

    Overlay <-. "invoke" .-> Cmd
    Main <-. "invoke" .-> Cmd
    Sink -. "events" .-> Overlay & Main
    Cmd --> AppState
    AppState --> Capture & OverlayCtl & Secrets & LocalAi

    Capture -- "AudioFrame" --> Trans
    Capture -- "ScreenFrame" --> Screen
    Capture -- "analyzed frames (opt-in)" --> Shots
    Trans --> Conv --> Req --> Assist --> Ctx
    Screen --> Ctx
    Conv -- "question + answer turn" --> Integrity
    Trans -- "word timing" --> Integrity
    Screen -- "candidate video tile" --> Integrity
    Conv & Assist & Screen & Integrity & Persist -- "MeetingEvent" --> Sink

    Conv & Assist & Integrity --> Intel
    Trans & Assist & Screen & Integrity --> Providers
    Persist --> Store
    Capture & OverlayCtl --> Platform

    Platform --> OS
    Secrets --> Keychain
    LocalAi -. "spawn / health" .-> Sidecar
    Providers -. "HTTP" .-> Sidecar & Ollama & Cloud
    Store --> Disk
```
