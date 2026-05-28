import { useRef, useCallback } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { useHandTracking } from "../hooks/useHandTracking";
import { useStableSign, ZERO_SCORES } from "../hooks/useStableSign";
import { scoreAllLetters } from "../classifier/classifySign";

export function HandTracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { result, update } = useStableSign();

  /**
   * Called every frame by useHandTracking after MediaPipe runs.
   * landmarkSets[0] is the first detected hand (we only classify one hand).
   * When no hand is present the array is empty → pass zeros so all letter
   * scores decay naturally through the EMA filter.
   */
  const onLandmarks = useCallback(
    (landmarkSets: NormalizedLandmark[][]) => {
      const lm = landmarkSets[0];
      update(lm ? scoreAllLetters(lm) : ZERO_SCORES);
    },
    [update]
  );

  const status = useHandTracking(videoRef, canvasRef, onLandmarks);

  return (
    <div style={containerStyle}>
      {/* Hidden video element — MediaPipe reads frames from here */}
      <video ref={videoRef} style={{ display: "none" }} width={640} height={480} playsInline />

      {/* Canvas shows the mirrored camera feed + drawn landmarks */}
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{ display: "block", transform: "scaleX(-1)" }}
      />

      {/* Sign badge — shown as soon as a candidate passes the threshold,
          fully locked (green dot) once it's been held for COMMIT_FRAMES. */}
      {result.sign && <SignBadge sign={result.sign} confidence={result.confidence} isStable={result.isStable} />}

      {status === "loading" && (
        <div style={overlayStyle}>
          <p style={{ color: "#aaa", fontSize: 14 }}>Loading MediaPipe...</p>
        </div>
      )}

      {status === "error" && (
        <div style={overlayStyle}>
          <p style={{ color: "#f87171", fontSize: 14 }}>
            Camera error — check browser permissions and refresh.
          </p>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

interface BadgeProps {
  sign: string;
  confidence: number;
  isStable: boolean;
}

function SignBadge({ sign, confidence, isStable }: BadgeProps) {
  const pct = Math.round(confidence * 100);

  return (
    <div style={{
      position: "absolute",
      bottom: 16,
      right: 16,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 4,
      background: "rgba(0,0,0,0.60)",
      backdropFilter: "blur(6px)",
      borderRadius: 14,
      padding: "10px 22px 12px",
      userSelect: "none",
      // Dim the badge while the sign is still accumulating stability.
      opacity: isStable ? 1 : 0.65,
    }}>
      {/* The detected letter */}
      <span style={{ fontSize: 80, fontWeight: 700, lineHeight: 1, color: "#fff", letterSpacing: "0.02em" }}>
        {sign}
      </span>

      {/* Confidence percentage */}
      <span style={{ fontSize: 13, color: "#ccc", letterSpacing: "0.04em" }}>
        {pct}%
      </span>

      {/* Stability indicator */}
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 2 }}>
        {/* Dot: green = stable/locked, amber = still accumulating */}
        <span style={{
          display: "inline-block",
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: isStable ? "#4ade80" : "#fbbf24",
          boxShadow: isStable ? "0 0 6px #4ade80" : "0 0 6px #fbbf24",
        }} />
        <span style={{ fontSize: 11, color: isStable ? "#4ade80" : "#fbbf24", letterSpacing: "0.06em" }}>
          {isStable ? "STABLE" : "..."}
        </span>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  position: "relative",
  width: 640,
  height: 480,
  borderRadius: 8,
  overflow: "hidden",
  border: "2px solid #333",
  background: "#1a1a1a",
};

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "#1a1a1a",
};
