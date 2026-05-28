/**
 * useStableSign — temporal smoothing + hysteresis for ASL letter output.
 *
 * ── Problem ──────────────────────────────────────────────────────────────────
 * Raw per-frame scores fluctuate even for a held gesture because MediaPipe
 * jitters slightly between frames, and a partially-formed hand shape can flip
 * between two high-scoring letters rapidly.
 *
 * ── Solution: two-stage filter ───────────────────────────────────────────────
 *
 *  Stage 1 — EMA smoothing (per letter)
 *    smoothed[l] = α × raw[l] + (1−α) × smoothed[l]
 *    α = 0.22  → a spike lasts ~4–5 frames before decaying; a held sign
 *    reaches 95 % of its true score in ~12 frames (~0.4 s at 30 fps).
 *    Letters not detected this frame get raw = 0 and decay automatically.
 *
 *  Stage 2 — commit / release hysteresis
 *    Commit:  smoothed winner > COMMIT_THRESHOLD for COMMIT_FRAMES consecutive frames.
 *    Release: currently displayed sign drops below RELEASE_THRESHOLD.
 *    The gap between thresholds (0.58 → 0.33) prevents rapid on/off flickering
 *    when confidence hovers at the boundary.
 *
 * ── Return value ─────────────────────────────────────────────────────────────
 *   sign         — the currently committed letter, or null if none.
 *   confidence   — smoothed score of the displayed sign (0–1).
 *   isStable     — true once the sign has been held for COMMIT_FRAMES.
 *                  false = "candidate" — a sign is accumulating but not yet locked.
 */

import { useRef, useState, useCallback } from "react";
import { ALL_SIGNS, type Sign, type ScoreMap } from "../classifier/classifySign";

export interface StableResult {
  sign: Sign | null;
  confidence: number;
  isStable: boolean;
}

const EMA_ALPHA       = 0.22;  // smoothing weight
const COMMIT_THRESHOLD = 0.58; // smoothed score required to start committing
const RELEASE_THRESHOLD = 0.33; // committed sign cleared below this
const COMMIT_FRAMES   = 7;     // consecutive frames above threshold before lock

const ZERO_SCORES = Object.fromEntries(ALL_SIGNS.map(s => [s, 0])) as ScoreMap;

export function useStableSign() {
  // Smoothed score per letter — lives in a ref so the animation loop never
  // sees a stale closure (unlike useState, ref mutations are synchronous).
  const smoothed    = useRef<ScoreMap>({ ...ZERO_SCORES });
  const stableCount = useRef(0);           // consecutive frames same candidate is winning
  const pendingSign = useRef<Sign | null>(null); // sign currently accumulating

  const [result, setResult] = useState<StableResult>({
    sign: null,
    confidence: 0,
    isStable: false,
  });

  /**
   * update — call this once per animation frame with the raw per-letter scores.
   * Pass all-zeros (ZERO_SCORES) when no hand is in frame so letters decay.
   */
  const update = useCallback((scores: ScoreMap) => {
    const sm = smoothed.current;

    // ── Stage 1: EMA ──────────────────────────────────────────────────────
    for (const sign of ALL_SIGNS) {
      sm[sign] = EMA_ALPHA * scores[sign] + (1 - EMA_ALPHA) * sm[sign];
    }

    // ── Find the smoothed winner ───────────────────────────────────────────
    let topSign: Sign | null = null;
    let topConf = 0;
    for (const sign of ALL_SIGNS) {
      if (sm[sign] > topConf) {
        topConf = sm[sign];
        topSign = sign;
      }
    }
    // Suppress if below threshold — nothing confident enough to consider.
    if (topConf < COMMIT_THRESHOLD) topSign = null;

    // ── Stage 2: commit counter ────────────────────────────────────────────
    if (topSign === pendingSign.current) {
      stableCount.current++;
    } else {
      // A different sign (or null) is now leading — restart the counter.
      stableCount.current = 1;
      pendingSign.current = topSign;
    }

    const isNowStable = stableCount.current >= COMMIT_FRAMES;

    // ── Update display state ───────────────────────────────────────────────
    setResult(prev => {
      // Case A: we have a stable new winner → commit to it.
      if (isNowStable && topSign !== null && topSign !== prev.sign) {
        return { sign: topSign, confidence: topConf, isStable: true };
      }

      // Case B: currently showing a committed sign.
      if (prev.sign !== null) {
        const currentConf = sm[prev.sign];

        // Release: the committed sign's smoothed score fell below the release floor.
        if (currentConf < RELEASE_THRESHOLD) {
          return { sign: null, confidence: 0, isStable: false };
        }

        // Refresh confidence for the current sign (update % in UI).
        if (prev.sign === topSign) {
          return { ...prev, confidence: topConf, isStable: true };
        }

        // Keep showing the current sign while a new one is accumulating.
        return { ...prev, confidence: currentConf };
      }

      // Case C: nothing committed yet — show candidate dimly once it passes threshold.
      if (topSign !== null && topConf >= COMMIT_THRESHOLD) {
        return { sign: topSign, confidence: topConf, isStable: false };
      }

      return prev;
    });
  }, []);

  return { result, update };
}

/** Convenience: zero scores for when no hand is detected. */
export { ZERO_SCORES };
