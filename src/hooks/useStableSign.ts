/**
 * useStableSign — Prediction Smoothing Layer
 *
 * ── Why stability matters in real-time vision ────────────────────────────────
 * Raw per-frame scores are noisy.  Even for a perfectly held sign, MediaPipe
 * jitters slightly between frames, and transitional hand poses during sign
 * changes can briefly score high for a wrong letter.  Without smoothing you get:
 *   A → flicker to B for 2 frames → back to A → feels broken
 *
 * ── Two-stage filter ─────────────────────────────────────────────────────────
 *
 * Stage 1 — Exponential Moving Average (EMA) per letter:
 *
 *   smoothed[L] = α × raw[L] + (1−α) × smoothed[L]
 *
 *   α = 0.22 ("learning rate").  A single-frame spike takes ~5 frames to
 *   decay to 50 % of its original value.  A genuinely held sign reaches
 *   95 % of its true score in ~12 frames (~0.4 s at 30 fps).
 *   Letters not detected this frame get raw = 0, so they decay automatically.
 *
 * Stage 2 — Commit / Release hysteresis:
 *
 *   Commit:  winner's smoothed score > COMMIT_THRESHOLD for COMMIT_FRAMES
 *            consecutive frames.
 *   Release: currently displayed sign's score drops below RELEASE_THRESHOLD.
 *
 *   The gap between thresholds (0.58 → 0.33) prevents rapid on/off at the
 *   boundary — a classic "Schmitt trigger" pattern from electronics.
 *
 * ── Synchronous return value ─────────────────────────────────────────────────
 * update() returns the new StableResult immediately (in addition to queuing
 * a React state update for rendering).  This lets the caller (HandTracker)
 * pass the result to the word builder in the SAME animation frame, with zero
 * lag.  The React state is only needed to trigger a re-render.
 */

import { useRef, useState, useCallback } from "react";
import { ALL_SIGNS, type Sign, type ScoreMap } from "../classifier/classifySign";

export interface StableResult {
  sign:       Sign | null; // currently displayed letter (null = nothing confident)
  confidence: number;      // smoothed score of the displayed sign, 0–1
  isStable:   boolean;     // true once held for COMMIT_FRAMES (green dot in UI)
}

const EMA_ALPHA        = 0.22;
const COMMIT_THRESHOLD = 0.58; // smoothed score needed to start displaying / accumulating
const RELEASE_THRESHOLD = 0.33; // committed sign is cleared below this
const COMMIT_FRAMES    = 7;    // consecutive frames above threshold before locking

const INITIAL: StableResult = { sign: null, confidence: 0, isStable: false };

export function useStableSign() {
  // Smoothed scores live in a ref — they update every frame but don't need
  // to trigger React renders by themselves.
  const smoothed    = useRef<ScoreMap>(
    Object.fromEntries(ALL_SIGNS.map(s => [s, 0])) as ScoreMap
  );
  const stableCount = useRef(0);
  const pendingSign = useRef<Sign | null>(null);

  // resultRef holds the current logical result (used for synchronous reads).
  // result (state) is the React-reactive copy used for rendering.
  const resultRef = useRef<StableResult>(INITIAL);
  const [result, setResult] = useState<StableResult>(INITIAL);

  /**
   * update — call once per animation frame with raw per-letter scores.
   *
   * Returns the new StableResult synchronously so callers in the same
   * frame (e.g., the word builder) don't need to wait for a React re-render.
   */
  const update = useCallback((scores: ScoreMap): StableResult => {
    const sm   = smoothed.current;
    const prev = resultRef.current;

    // ── Stage 1: EMA update ─────────────────────────────────────────────────
    for (const sign of ALL_SIGNS) {
      sm[sign] = EMA_ALPHA * scores[sign] + (1 - EMA_ALPHA) * sm[sign];
    }

    // ── Find smoothed winner ────────────────────────────────────────────────
    let topSign: Sign | null = null;
    let topConf = 0;
    for (const sign of ALL_SIGNS) {
      if (sm[sign] > topConf) { topConf = sm[sign]; topSign = sign; }
    }
    if (topConf < COMMIT_THRESHOLD) topSign = null; // not confident enough

    // ── Stage 2: consecutive-frame counter ─────────────────────────────────
    if (topSign === pendingSign.current) {
      stableCount.current++;
    } else {
      stableCount.current = 1;
      pendingSign.current = topSign;
    }
    const isNowStable = stableCount.current >= COMMIT_FRAMES;

    // ── Compute new result ──────────────────────────────────────────────────
    let next: StableResult;

    if (isNowStable && topSign !== null && topSign !== prev.sign) {
      // Lock in a newly committed sign
      next = { sign: topSign, confidence: topConf, isStable: true };

    } else if (prev.sign !== null) {
      const currentConf = sm[prev.sign];

      if (currentConf < RELEASE_THRESHOLD) {
        // Committed sign has decayed — clear it
        next = INITIAL;
      } else if (prev.sign === topSign) {
        // Refresh confidence of the current sign
        next = { ...prev, confidence: topConf, isStable: isNowStable };
      } else {
        // A different sign is accumulating; keep showing the committed one
        next = { ...prev, confidence: currentConf };
      }

    } else if (topSign !== null && topConf >= COMMIT_THRESHOLD) {
      // Nothing committed yet — show candidate dimly while accumulating
      next = { sign: topSign, confidence: topConf, isStable: false };

    } else {
      next = prev; // nothing to show, nothing changed
    }

    // ── Sync to React state only when something meaningful changed ───────────
    if (next !== prev) {
      resultRef.current = next;
      setResult(next);
    }

    return next; // synchronous return for same-frame consumers
  }, []);

  return { result, update };
}
