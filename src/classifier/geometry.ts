/**
 * geometry.ts — low-level landmark math
 *
 * All inputs are MediaPipe NormalizedLandmark objects:
 *   x, y ∈ [0, 1]  (x=0 left, y=0 top of frame)
 *   z           depth relative to wrist (less reliable, not used here)
 *
 * Key coordinate insight:
 *   y INCREASES downward, so a raised fingertip has a SMALLER y than its knuckle.
 *   That's why "tip.y < mcp.y" means the finger is pointing UP.
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
export type NL = NormalizedLandmark;

/** 2-D Euclidean distance between two landmarks (normalized coords). */
export function dist(a: NL, b: NL): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Palm width — horizontal span from index MCP (5) to pinky MCP (17).
 * Used to normalize distances so they're hand-size and camera-distance
 * independent.  A typical hand takes up ~20 % of frame width.
 */
export function palmWidth(lm: NL[]): number {
  return Math.max(Math.abs(lm[5].x - lm[17].x), 0.01);
}

/**
 * Palm height — vertical span from wrist (0) to middle MCP (9).
 * Used to normalize finger extension so a fully extended finger scores ≈ 1
 * regardless of how close/far the hand is from the camera.
 */
export function palmHeight(lm: NL[]): number {
  return Math.max(Math.abs(lm[0].y - lm[9].y), 0.01);
}

/**
 * Finger extension score  [0, 1].
 *   0 = completely curled (tip is at or below the knuckle)
 *   1 = fully extended upward (tip is one full palm-height above the knuckle)
 *
 * Formula: (mcp.y − tip.y) / palmHeight
 *   When the finger points straight up, tip.y << mcp.y → positive ratio → high score.
 *   When curled, tip.y ≥ mcp.y → zero or negative ratio → clamped to 0.
 */
export function fingerExt(tip: NL, mcp: NL, palmH: number): number {
  return Math.max(0, Math.min(1, (mcp.y - tip.y) / palmH));
}

/**
 * Thumb horizontal spread ratio  [0, ∞), normalised by palmWidth.
 *   Measures how far the thumb tip (4) is from the index knuckle (5) sideways.
 *   < 0.30  → thumb is tucked / folded (A, B, D, I, F)
 *   > 0.55  → thumb extended to the side (L, Y)
 *
 * Works for both left and right hands because we take the absolute value.
 * The CSS mirror on the canvas does NOT affect landmarks — MediaPipe sees
 * the original (un-mirrored) video, so landmark coordinates are consistent.
 */
export function thumbSpread(lm: NL[], palmW: number): number {
  return Math.abs(lm[4].x - lm[5].x) / palmW;
}

/**
 * Maps a raw value linearly from [lo, hi] → [0, 1], clamped at both ends.
 * Use this to turn a geometric measurement into a 0–1 confidence component.
 *
 * Examples:
 *   linearScore(thumbSpread, 0.30, 0.70) → 0 when spread ≤ 0.30, 1 when ≥ 0.70
 *   1 - linearScore(spread, 0.10, 0.50) → 1 when spread is tiny, 0 when large
 */
export function linearScore(v: number, lo: number, hi: number): number {
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}
