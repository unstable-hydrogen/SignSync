/**
 * useWordBuilder — Word Building Layer
 *
 * Converts a stream of stable sign detections into a typed word by applying:
 *   1. Hold-to-confirm   — a sign must be held stably for HOLD_MS before adding
 *   2. Duplicate guard   — same letter can't be added again for SAME_COOLDOWN_MS
 *   3. Word buffer       — accumulates letters into a string
 *
 * ── Hold-to-confirm ──────────────────────────────────────────────────────────
 * Simply detecting a letter isn't enough — the user might be mid-transition
 * between two signs.  Requiring a deliberate hold (1.5 s) before committing
 * means only intentional signs end up in the word.
 *
 * The UI shows a progress bar filling up during the hold.  When it reaches
 * 100 %, the letter is appended (if not in cooldown).
 *
 * ── Duplicate guard ──────────────────────────────────────────────────────────
 * Without this, holding a sign past the commit point would add the same letter
 * repeatedly.  The `hasCommittedThisHold` ref prevents re-adding on the same
 * continuous hold.  SAME_COOLDOWN_MS prevents re-adding immediately after a
 * sign release + re-hold of the same letter.
 *
 * To intentionally type "LL", the user must: show L → wait for commit →
 * remove hand briefly → wait 1.5 s → show L again.
 *
 * ── tick() call pattern ──────────────────────────────────────────────────────
 * tick() is called synchronously every animation frame (from HandTracker's
 * onLandmarks callback, after useStableSign.update() returns the new result).
 * It uses performance.now() for timing, so it works correctly at any frame rate.
 */

import { useRef, useState, useCallback } from "react";
import type { Sign } from "../classifier/classifySign";

const HOLD_MS         = 1500;  // hold stably for this long to add a letter
const SAME_COOLDOWN_MS = 2000; // minimum gap before the SAME letter can be added again

export interface WordBuilderResult {
  word:         string;       // the text built so far
  holdProgress: number;       // 0–1, how far through the hold we are (for progress bar)
  lastCommitted: Sign | null; // the most recently committed letter
}

export function useWordBuilder() {
  const [word,          setWord]          = useState("");
  const [holdProgress,  setHoldProgress]  = useState(0);
  const [lastCommitted, setLastCommitted] = useState<Sign | null>(null);

  // Mutable refs — don't need to trigger re-renders, must be synchronously
  // readable inside the animation-frame callback.
  const currentSignRef       = useRef<Sign | null>(null); // sign being tracked this hold
  const holdStartRef         = useRef<number | null>(null); // when the current hold began
  const hasCommittedThisHold = useRef(false);              // prevent re-commit on same hold
  const lastCommittedRef     = useRef<Sign | null>(null);  // for same-letter cooldown check
  const lastCommitTimeRef    = useRef<number>(0);          // timestamp of last commit

  /**
   * tick — call every animation frame with the current stable sign and whether
   * it has passed the stability threshold.
   *
   * Internal state machine:
   *
   *   IDLE (no sign)
   *     → sign appears + isStable  ↓  START hold timer
   *   HOLDING (sign stable, timer running)
   *     → sign changes / lost      ↓  RESET
   *     → timer reaches HOLD_MS    ↓  COMMIT (if not in cooldown)
   *   COMMITTED (letter added, still holding)
   *     → sign changes / lost      ↓  RESET (allow next hold)
   */
  const tick = useCallback((sign: Sign | null, isStable: boolean) => {
    const now = performance.now();

    // ── Sign changed or disappeared — reset the hold tracker ───────────────
    if (sign !== currentSignRef.current) {
      currentSignRef.current       = sign;
      holdStartRef.current         = null;
      hasCommittedThisHold.current = false;
      setHoldProgress(0);
    }

    // ── No sign, or sign not yet stable — nothing to do ────────────────────
    if (!sign || !isStable) {
      if (holdStartRef.current !== null) {
        holdStartRef.current = null;
        setHoldProgress(0);
      }
      return;
    }

    // ── Start the hold timer the moment the sign becomes stable ────────────
    if (holdStartRef.current === null) {
      holdStartRef.current = now;
    }

    // ── Already committed on this hold — stay at 100 %, wait for release ──
    if (hasCommittedThisHold.current) {
      setHoldProgress(1);
      return;
    }

    // ── Update hold progress ────────────────────────────────────────────────
    const elapsed  = now - holdStartRef.current;
    const progress = Math.min(1, elapsed / HOLD_MS);
    setHoldProgress(progress);

    // ── Threshold reached — try to commit ──────────────────────────────────
    if (progress >= 1) {
      const isSameLetter   = sign === lastCommittedRef.current;
      const timeSinceLast  = now - lastCommitTimeRef.current;
      const cooldownNeeded = isSameLetter ? SAME_COOLDOWN_MS : 0;

      if (timeSinceLast >= cooldownNeeded) {
        // ✓ Commit the letter
        setWord(prev => prev + sign);
        setLastCommitted(sign);
        lastCommittedRef.current  = sign;
        lastCommitTimeRef.current = now;
        hasCommittedThisHold.current = true;
      }
      // If in cooldown: do nothing — user must release and re-hold
    }
  }, []);

  /** Append a space (keyboard shortcut: Space). */
  const addSpace = useCallback(() => {
    setWord(prev => (prev.endsWith(" ") ? prev : prev + " "));
  }, []);

  /** Remove the last character (keyboard shortcut: Backspace). */
  const backspace = useCallback(() => {
    setWord(prev => prev.slice(0, -1));
  }, []);

  /** Clear the entire word (keyboard shortcut: Escape). */
  const clear = useCallback(() => {
    setWord("");
    setLastCommitted(null);
    lastCommittedRef.current = null;
  }, []);

  return {
    word,
    holdProgress,
    lastCommitted,
    tick,
    addSpace,
    backspace,
    clear,
  };
}
