/**
 * features.ts — Feature Extraction Layer
 *
 * ── Why a separate layer? ────────────────────────────────────────────────────
 * Raw landmarks are just 21 (x, y, z) points.  Before any letter can be
 * recognised, we need to derive *geometric meaning* from those points:
 * "how curled is this finger?", "how far is the thumb from the palm?", etc.
 *
 * Extracting these once per frame into a single HandFeatures struct means:
 *   • Each letter scorer reads pre-computed, scale-invariant values.
 *   • No geometric calculation is repeated across multiple scorers.
 *   • Adding a new feature (e.g., wrist angle) only requires touching this file.
 *   • Features can be logged / visualised independently of classification.
 *
 * ── Scale invariance ─────────────────────────────────────────────────────────
 * All distances are divided by palmWidth or palmHeight (computed from the same
 * frame's landmarks).  This makes every value independent of:
 *   • How large the user's hand is.
 *   • How far the hand is from the camera.
 *   • The camera's field of view or resolution.
 *
 * ── Two complementary curl signals ───────────────────────────────────────────
 * 1. fingerExt  (continuous, 0–1)  — (mcp.y − tip.y) / palmHeight
 *      Robust when the hand is upright.  Degrades at oblique angles.
 *
 * 2. tipBelowPip  (binary)  — tip.y > pip.y
 *      Works even when the hand is tilted sideways, because it only checks
 *      the relative ordering of two joints on the SAME finger, not an absolute
 *      vertical reference.  Used as a cross-check on the continuous score.
 *
 * ── Landmark indices ─────────────────────────────────────────────────────────
 *   0  WRIST
 *   1  THUMB_CMC   2  THUMB_MCP   3  THUMB_IP    4  THUMB_TIP
 *   5  INDEX_MCP   6  INDEX_PIP   7  INDEX_DIP   8  INDEX_TIP
 *   9  MIDDLE_MCP 10  MIDDLE_PIP 11  MIDDLE_DIP  12  MIDDLE_TIP
 *  13  RING_MCP   14  RING_PIP   15  RING_DIP    16  RING_TIP
 *  17  PINKY_MCP  18  PINKY_PIP  19  PINKY_DIP   20  PINKY_TIP
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { dist, palmWidth, palmHeight, fingerExt, thumbSpread } from "./geometry";
import { normalizeHand } from "./normalize";

type NL = NormalizedLandmark;

export interface HandFeatures {
  // ── Palm reference dimensions (used for normalisation) ───────────────────
  palmH: number;   // wrist-to-middle-MCP vertical span
  palmW: number;   // index-MCP-to-pinky-MCP horizontal span

  // ── Finger extension  [0 = fully curled, 1 = fully extended upward] ──────
  // Formula: (mcp.y − tip.y) / palmH  →  clamped to [0, 1].
  indexExt:  number;
  middleExt: number;
  ringExt:   number;
  pinkyExt:  number;

  // ── Finger curl  [inverse of extension] ──────────────────────────────────
  indexCurl:  number;
  middleCurl: number;
  ringCurl:   number;
  pinkyCurl:  number;

  // ── PIP-based binary curl check ───────────────────────────────────────────
  // True when the fingertip is geometrically BELOW its PIP (middle) joint.
  // Independent of absolute y values, so robust when hand is tilted sideways.
  indexTipBelowPip:  boolean;
  middleTipBelowPip: boolean;
  ringTipBelowPip:   boolean;
  pinkyTipBelowPip:  boolean;

  // ── Thumb geometry ────────────────────────────────────────────────────────
  thumbSpreadRatio: number; // |thumbTip.x − indexMcp.x| / palmW  — lateral splay
  thumbIndexPinch:  number; // dist(thumbTip, indexTip)  / palmW  — F-pinch gap
  thumbMiddleDist:  number; // dist(thumbTip, middlePip) / palmW  — D closeness

  // ── Inter-finger geometry ─────────────────────────────────────────────────
  indexMiddleSpread: number; // |indexTip.x − middleTip.x| / palmW — V-spread
}

/**
 * extractFeatures — convert 21 raw landmarks into a HandFeatures struct.
 * Returns null if the landmark array is incomplete (< 21 points).
 */
export function extractFeatures(rawLm: NL[]): HandFeatures | null {
  if (rawLm.length < 21) return null;

  // Z-score normalize per axis before any geometric computation.
  // Adapted from Kaggle ASL 1st-place: per-body-part normalization in model forward().
  const lm = normalizeHand(rawLm);

  const palmH = palmHeight(lm);
  const palmW  = palmWidth(lm);

  const indexExt  = fingerExt(lm[8],  lm[5],  palmH);
  const middleExt = fingerExt(lm[12], lm[9],  palmH);
  const ringExt   = fingerExt(lm[16], lm[13], palmH);
  const pinkyExt  = fingerExt(lm[20], lm[17], palmH);

  return {
    palmH, palmW,

    indexExt, middleExt, ringExt, pinkyExt,
    indexCurl:  1 - indexExt,
    middleCurl: 1 - middleExt,
    ringCurl:   1 - ringExt,
    pinkyCurl:  1 - pinkyExt,

    // PIP binary checks — compare tip to the SAME finger's PIP joint only
    indexTipBelowPip:  lm[8].y  > lm[6].y,
    middleTipBelowPip: lm[12].y > lm[10].y,
    ringTipBelowPip:   lm[16].y > lm[14].y,
    pinkyTipBelowPip:  lm[20].y > lm[18].y,

    thumbSpreadRatio: thumbSpread(lm, palmW),
    thumbIndexPinch:  dist(lm[4], lm[8])  / palmW,
    thumbMiddleDist:  dist(lm[4], lm[10]) / palmW,

    indexMiddleSpread: Math.abs(lm[8].x - lm[12].x) / palmW,
  };
}
