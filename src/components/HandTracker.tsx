/**
 * HandTracker — UI Rendering Layer
 *
 * Wires together all pipeline layers:
 *
 *   Webcam frames
 *     ↓  useHandTracking    (camera + MediaPipe + canvas drawing)
 *   Raw landmarks (21 pts)
 *     ↓  extractFeatures    (geometry → HandFeatures struct)
 *   HandFeatures
 *     ↓  scoreAllLetters    (heuristic scorer for each letter)
 *   ScoreMap (9 confidence values)
 *     ↓  useStableSign      (EMA smoothing + commit/release hysteresis)
 *   StableResult (sign, confidence, isStable)
 *     ↓  useWordBuilder     (hold-to-confirm + word buffer)
 *   Word + holdProgress
 *     ↓  React render       (SignBadge + WordPanel)
 */

import { useRef, useCallback, useEffect } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { useHandTracking } from "../hooks/useHandTracking";
import { useStableSign }   from "../hooks/useStableSign";
import { useWordBuilder }  from "../hooks/useWordBuilder";
import { extractFeatures } from "../classifier/features";
import { scoreAllLetters, ZERO_SCORES, type Sign } from "../classifier/classifySign";

export function HandTracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { result, update: updateStable } = useStableSign();
  const { word, holdProgress, tick: wordTick, addSpace, backspace, clear } = useWordBuilder();

  /**
   * onLandmarks — called every animation frame by useHandTracking.
   *
   * This is the entry point of the recognition pipeline.  The entire chain
   * (feature extraction → scoring → smoothing → word building) runs here
   * synchronously so all updates happen in the same animation frame.
   */
  const onLandmarks = useCallback(
    (landmarkSets: NormalizedLandmark[][]) => {
      const lm       = landmarkSets[0]; // classify first detected hand only
      const features = lm ? extractFeatures(lm) : null;
      const scores   = features ? scoreAllLetters(features) : ZERO_SCORES;
      const stable   = updateStable(scores); // returns new result synchronously
      wordTick(stable.sign, stable.isStable);
    },
    [updateStable, wordTick]
  );

  const status = useHandTracking(videoRef, canvasRef, onLandmarks);

  // Keyboard shortcuts — Space, Backspace, Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ")         { e.preventDefault(); addSpace();  }
      if (e.key === "Backspace") { e.preventDefault(); backspace(); }
      if (e.key === "Escape")    { e.preventDefault(); clear();     }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [addSpace, backspace, clear]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>

      {/* ── Camera feed ─────────────────────────────────────────────────── */}
      <div style={cameraContainerStyle}>
        {/* Hidden video — MediaPipe reads frames from this element */}
        <video ref={videoRef} style={{ display: "none" }} width={640} height={480} playsInline />

        {/* Canvas: mirrored with CSS scaleX(-1) for a selfie-camera feel.
            The CSS transform is purely visual — MediaPipe reads the original
            (un-mirrored) video, so landmarks are in original coordinates.  */}
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{ display: "block", transform: "scaleX(-1)" }}
        />

        {/* Sign badge — bottom-right overlay showing detected letter */}
        {result.sign && (
          <SignBadge
            sign={result.sign}
            confidence={result.confidence}
            isStable={result.isStable}
            holdProgress={holdProgress}
          />
        )}

        {status === "loading" && (
          <CenterOverlay>
            <span style={{ color: "#aaa", fontSize: 14 }}>Loading MediaPipe...</span>
          </CenterOverlay>
        )}

        {status === "error" && (
          <CenterOverlay>
            <span style={{ color: "#f87171", fontSize: 14 }}>
              Camera error — check browser permissions and refresh.
            </span>
          </CenterOverlay>
        )}
      </div>

      {/* ── Word panel ─────────────────────────────────────────────────── */}
      <WordPanel
        word={word}
        onSpace={addSpace}
        onBackspace={backspace}
        onClear={clear}
      />
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface SignBadgeProps {
  sign:         Sign;
  confidence:   number;
  isStable:     boolean;
  holdProgress: number; // 0–1
}

function SignBadge({ sign, confidence, isStable, holdProgress }: SignBadgeProps) {
  const pct          = Math.round(confidence * 100);
  const dotColor     = isStable ? "#4ade80" : "#fbbf24";
  const statusText   = isStable ? "STABLE" : "...";
  const progressFill = Math.round(holdProgress * 100);
  const barColor     = holdProgress >= 1 ? "#4ade80" : "#fbbf24";

  return (
    <div style={{
      position:       "absolute",
      bottom:         16,
      right:          16,
      display:        "flex",
      flexDirection:  "column",
      alignItems:     "center",
      gap:            4,
      background:     "rgba(0,0,0,0.65)",
      backdropFilter: "blur(8px)",
      borderRadius:   14,
      padding:        "10px 22px 12px",
      userSelect:     "none",
      minWidth:       90,
      // Dim while accumulating (not yet stable)
      opacity: isStable ? 1 : 0.72,
    }}>

      {/* Detected letter */}
      <span style={{ fontSize: 80, fontWeight: 700, lineHeight: 1, color: "#fff", letterSpacing: "0.02em" }}>
        {sign}
      </span>

      {/* Confidence percentage */}
      <span style={{ fontSize: 13, color: "#bbb", letterSpacing: "0.05em" }}>
        {pct}%
      </span>

      {/* Stability indicator */}
      <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
        <span style={{
          display:     "inline-block",
          width:       8,
          height:      8,
          borderRadius: "50%",
          background:  dotColor,
          boxShadow:   `0 0 6px ${dotColor}`,
        }} />
        <span style={{ fontSize: 10, color: dotColor, letterSpacing: "0.08em", fontWeight: 600 }}>
          {statusText}
        </span>
      </div>

      {/* Hold progress bar — fills up over HOLD_MS; 100% → letter committed */}
      <div style={{
        width:        "100%",
        height:       4,
        borderRadius: 2,
        background:   "rgba(255,255,255,0.15)",
        marginTop:    4,
        overflow:     "hidden",
      }}>
        <div style={{
          width:      `${progressFill}%`,
          height:     "100%",
          background: barColor,
          borderRadius: 2,
          transition: "width 0.05s linear, background 0.2s",
        }} />
      </div>

    </div>
  );
}

interface WordPanelProps {
  word:        string;
  onSpace:     () => void;
  onBackspace: () => void;
  onClear:     () => void;
}

function WordPanel({ word, onSpace, onBackspace, onClear }: WordPanelProps) {
  const displayWord = word || "";
  const isEmpty     = displayWord.length === 0;

  return (
    <div style={wordPanelStyle}>
      {/* Current word */}
      <div style={{
        fontSize:    36,
        fontFamily:  "monospace",
        fontWeight:  600,
        color:       isEmpty ? "#444" : "#fff",
        letterSpacing: "0.12em",
        minHeight:   52,
        lineHeight:  "52px",
        textAlign:   "left",
        width:       "100%",
        overflowX:   "auto",
        whiteSpace:  "nowrap",
      }}>
        {isEmpty ? "─" : displayWord}
        {/* Blinking cursor */}
        {!isEmpty && <span style={{ animation: "blink 1s step-end infinite", opacity: 1 }}>▎</span>}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <PanelButton onClick={onSpace}     label="Space"     kbd="Space" />
        <PanelButton onClick={onBackspace} label="⌫"         kbd="⌫"     />
        <PanelButton onClick={onClear}     label="Clear"     kbd="Esc"   />
      </div>

      {/* Hint */}
      <p style={{ margin: "10px 0 0", fontSize: 11, color: "#555", letterSpacing: "0.04em" }}>
        Hold a sign for 1.5 s to add a letter &nbsp;·&nbsp; Space / ⌫ Backspace / Esc to clear
      </p>
    </div>
  );
}

function PanelButton({ onClick, label, kbd }: { onClick: () => void; label: string; kbd: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        background:   "rgba(255,255,255,0.07)",
        border:       "1px solid rgba(255,255,255,0.15)",
        borderRadius: 6,
        color:        "#ccc",
        fontSize:     13,
        padding:      "5px 14px",
        cursor:       "pointer",
        display:      "flex",
        alignItems:   "center",
        gap:          6,
      }}
    >
      {label}
      <kbd style={{
        fontSize:     10,
        background:   "rgba(255,255,255,0.08)",
        borderRadius: 3,
        padding:      "1px 5px",
        color:        "#777",
      }}>
        {kbd}
      </kbd>
    </button>
  );
}

function CenterOverlay({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      position:       "absolute",
      inset:          0,
      display:        "flex",
      alignItems:     "center",
      justifyContent: "center",
      background:     "#1a1a1a",
    }}>
      {children}
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const cameraContainerStyle: React.CSSProperties = {
  position:     "relative",
  width:        640,
  height:       480,
  borderRadius: "8px 8px 0 0",
  overflow:     "hidden",
  border:       "2px solid #2a2a2a",
  borderBottom: "none",
  background:   "#1a1a1a",
};

const wordPanelStyle: React.CSSProperties = {
  width:        636,    // accounts for 2px border on each side
  background:   "#141414",
  border:       "2px solid #2a2a2a",
  borderRadius: "0 0 8px 8px",
  padding:      "14px 20px 16px",
  display:      "flex",
  flexDirection: "column",
  alignItems:   "flex-start",
};
