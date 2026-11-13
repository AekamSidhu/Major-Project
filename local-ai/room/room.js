// Shared by the host and guest pages: media, signaling and the peer connection.
// Video and audio go browser to browser (WebRTC); the server only relays the
// offer, answer and network candidates between the two pages.

const ICE = [{ urls: "stun:stun.l.google.com:19302" }]; // finds a route across NATs; no media passes through it

// Camera and microphone, or with ?fake=1 a generated picture and tone (for testing without devices).
export async function getMedia() {
  if (new URLSearchParams(location.search).has("fake")) return fakeMedia();
  return navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: { echoCancellation: true, noiseSuppression: true },
  });
}

function fakeMedia() {
  const canvas = Object.assign(document.createElement("canvas"), { width: 640, height: 360 });
  const g = canvas.getContext("2d");
  const label = location.pathname.startsWith("/host") ? "interviewer (test video)" : "candidate (test video)";
  (function draw(t) {
    g.fillStyle = "#1d2b3a"; g.fillRect(0, 0, 640, 360);
    g.fillStyle = "#6c9cf0"; g.beginPath(); g.arc(320 + 150 * Math.sin(t / 900), 180, 40, 0, 7); g.fill();
    g.fillStyle = "#fff"; g.font = "20px sans-serif"; g.fillText(label, 20, 340);
    requestAnimationFrame(draw);
  })(0);
  const stream = canvas.captureStream(15);
  const ctx = new AudioContext(), osc = ctx.createOscillator(), gain = ctx.createGain(), dest = ctx.createMediaStreamDestination();
  osc.frequency.value = 220; gain.gain.value = 0.05; osc.connect(gain).connect(dest); osc.start();
  stream.addTrack(dest.stream.getAudioTracks()[0]);
  return stream;
}

// One WebSocket to the room; messages are JSON objects with a "type".
export function connect(room, role, key, name, onMessage) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams({ role, k: key, name: name || "" });
  const ws = new WebSocket(`${proto}://${location.host}/ws/${room}?${q}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return { send: (m) => ws.readyState === 1 && ws.send(JSON.stringify(m)), ws };
}

// A peer connection carrying `local` tracks; `onTrack(stream)` gets the other side's media.
export function makePeer(local, signal, onTrack, onState) {
  const pc = new RTCPeerConnection({ iceServers: ICE });
  local.getTracks().forEach((t) => pc.addTrack(t, local));
  pc.ontrack = (e) => onTrack(e.streams[0]);
  pc.onicecandidate = (e) => e.candidate && signal.send({ type: "ice", candidate: e.candidate });
  pc.onconnectionstatechange = () => onState && onState(pc.connectionState);
  return pc;
}

export async function handleSignal(pc, signal, m) {
  if (m.type === "offer") {
    await pc.setRemoteDescription(m.sdp);
    await pc.setLocalDescription(await pc.createAnswer());
    signal.send({ type: "answer", sdp: pc.localDescription });
  } else if (m.type === "answer") {
    await pc.setRemoteDescription(m.sdp);
  } else if (m.type === "ice" && m.candidate) {
    try { await pc.addIceCandidate(m.candidate); } catch (e) { console.warn("ice", e); }
  }
}

export function toggleTrack(stream, kind) {
  const track = stream.getTracks().find((t) => t.kind === kind);
  if (track) track.enabled = !track.enabled;
  return track ? track.enabled : false;
}
