# Gaze detection — `feature/gaze-detection`

Work branch for building, training and evaluating the **reading-gaze** ("eye
detection") signal of OpenMeet's answer-integrity check (cheating detection).
The other signal, AI-echo answer analysis, is developed separately on
`feature/echo-detection`. The two are combined only after both are merged.

> Status: research and training. Nothing from this branch is wired into the
> app yet. The `IntegrityAnalyzer` and `GazeProvider` exist only in the design
> diagrams (process 11.0 "Assess answer integrity" in the DFDs, steps
> 11.5–11.6).

## What we're detecting

Eyes move in a recognisable way when someone reads text off a screen. They
step left to right along a line, then snap back to the start of the next
line. Plotted over time, horizontal gaze looks like a sawtooth.

While the candidate answers, OpenMeet watches their video tile in the meeting
window and asks: are the eyes doing that reading pattern, or holding on one
off-camera spot for long stretches?

Looking away on its own is **not** the signal. People naturally look up or
away when they think. We look specifically for *reading*.

## How the signal works (planned)

1. **Find the face.** Capture the candidate's video tile from the meeting
   window at about 10 fps. This is the same picture the interviewer already
   sees, and nothing is installed on the candidate's machine. Crop to the
   face.
2. **Landmarks.** Run MediaPipe Face Mesh with iris refinement (478
   landmarks, including the irises) in the local AI sidecar.
3. **Gaze direction.** Compute the iris position relative to the eye corners
   and lids, giving horizontal and vertical gaze ratios. Estimate head pose
   from the same landmarks (e.g. `solvePnP`) so head turns aren't mistaken
   for eye movement.
4. **Reading pattern.** Build a gaze time series over each answer and
   extract features:
   - **return sweeps**: large, fast right-to-left jumps after slower
     left-to-right drift
   - **regularity**: whether the sweeps repeat at a similar rhythm, since
     lines of text are similar lengths
   - **off-camera fixation**: share of the answer spent looking at one
     off-screen spot, compared with the same person while listening
5. **Score.** Turn the features into a per-answer reading score.

Start rule-based: a sawtooth detector plus thresholds, which is easy to
explain. Then try a learned model, first logistic regression on windowed
features, and a small 1D-CNN or GRU only if there is enough data.

## Known hard parts

Be ready to explain these in the report and the viva:

- **Frame rate.** At 10 fps we can't see individual saccades, which last only
  tens of milliseconds. Return sweeps happen once per line, every few
  seconds, which 10 fps captures easily (it resolves motion up to 5 Hz).
  The detector must rely on line-level patterns, not saccades.
- **Resolution and compression.** In gallery view the face may be only
  100–200 px wide, which leaves each eye just a few pixels. Measure how
  accuracy drops as the tile shrinks.
- **Glasses, reflections, lighting, dark irises and camera position** (a
  laptop camera below the screen versus an external webcam).
- **Text right next to the camera.** A cheater can place the answer just
  under the webcam, which makes the sweeps small. Record this condition and
  report the result honestly.

## Data we need

Mock interviews with volunteers, held over a real call so the video goes
through real compression.

| Condition | Label |
|---|---|
| Answers naturally, looking at the screen or camera | not reading |
| Thinks with natural gaze aversion (up, away) | not reading (hard negative) |
| Glances at their own notes now and then | separate label (whether this counts depends on the interview's rules) |
| Reads an AI answer: near the camera, side of the screen, second monitor, phone | reading |

Vary glasses or no glasses, lighting, webcams, and tile size (gallery view
versus speaker view).

**Data rules**

- Face video is sensitive personal data. Get written consent from every
  participant. Anyone can withdraw and have their data deleted.
- **Never commit video, frames, landmark dumps or anything else tied to a
  participant.** Keep them in the team's private storage. Commit only code
  and aggregate results. Add `training/*/data/` to `.gitignore` before the
  first recording lands in this folder.
- We never identify anyone: no face recognition, no stored face templates.
  Use anonymous session ids.

## How we'll measure it

- Split train and test **by person**, so no participant appears in both.
- Report ROC-AUC, precision and recall, and the **false-positive rate at the
  chosen threshold**.
- Robustness curves: accuracy against face size in pixels and against frame
  rate (5, 10, 15 fps).
- Per-subgroup results: glasses, lighting, camera position.
- Per-frame processing time on a laptop CPU. It must keep up with 10 fps.

## Planned layout

```
training/gaze-detection/
  README.md      this file
  data/          local only, never committed
  landmarks/     MediaPipe wrapper, gaze ratios, head pose
  features/      sweep and fixation features
  notebooks/     exploration
  train.py       detector / classifier training
  evaluate.py    metrics and plots for the report
  results/       aggregate metrics only
```

## Where it plugs into the app (later)

- `local-ai/server.py`: a new loopback-only endpoint (e.g. `POST /v1/gaze`)
  that takes a face crop and returns landmarks and gaze. MediaPipe goes into
  `local-ai/requirements.txt`.
- `openmeet-core`: a `GazeProvider` trait, alongside the existing provider
  traits.
- `openmeet-providers`: a `LocalAiGaze` adapter for the sidecar endpoint.
- `openmeet-engine`: the screen worker crops the candidate's tile, and the
  `IntegrityAnalyzer` builds the gaze series and scores it when the answer
  turn ends.
- Off unless the interviewer turns it on (the "gaze on / off" setting in the
  design).

## Ground rules

- A flag is a **signal for the interviewer, never a verdict**. It always
  comes with evidence (timestamps, the gaze trace), and the human decides.
- Missing a cheater is better than flagging an honest candidate. Choose
  thresholds accordingly.
- Frames are processed in memory and discarded. They never go to a cloud
  provider.

## Done when

- [ ] Sidecar gaze endpoint, with tests
- [ ] Dataset collected under the data rules above
- [ ] Reading detector with results on held-out participants
- [ ] Robustness report (face size, frame rate, glasses, lighting)
- [ ] Threshold chosen and its false-positive rate documented
- [ ] Integrated into the engine behind a setting
- [ ] Fused with the echo signal (after both branches are merged)
