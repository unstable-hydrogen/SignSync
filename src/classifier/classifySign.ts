/**
 * classifySign.ts — heuristic ASL letter recognition
 *
 * Supported letters: A  B  D  F  I  L  V  W  Y
 * These were chosen because their hand shapes are geometrically distinct,
 * making heuristic separation reliable without an ML model.
 *
 * ── Landmark map (MediaPipe 21-point hand model) ────────────────────────────
 *
 *   0  WRIST
 *   1  THUMB_CMC   2  THUMB_MCP   3  THUMB_IP    4  THUMB_TIP
 *   5  INDEX_MCP   6  INDEX_PIP   7  INDEX_DIP   8  INDEX_TIP
 *   9  MIDDLE_MCP  10 MIDDLE_PIP  11 MIDDLE_DIP  12 MIDDLE_TIP
 *  13  RING_MCP    14 RING_PIP    15 RING_DIP    16 RING_TIP
 *  17  PINKY_MCP   18 PINKY_PIP   19 PINKY_DIP   20 PINKY_TIP
 *
 * ── Coordinate system ───────────────────────────────────────────────────────
 *   x ∈ [0,1]  left → right of the raw (un-mirrored) video frame
 *   y ∈ [0,1]  top → bottom  (larger y = lower on screen)
 *
 *   Therefore: a raised fingertip has tip.y < its mcp.y.
 *
 * ── Confidence formula ──────────────────────────────────────────────────────
 *   Each scorer returns 0–1.  Sub-scores are computed with geometry helpers
 *   (fingerExt, thumbSpread, dist, linearScore) and combined as weighted
 *   averages.  A "soft veto" pattern — "if primary feature < 0.25, cap the
 *   whole score" — prevents false positives without using hard binary logic.
 */

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import {
  dist, palmWidth, palmHeight, fingerExt, thumbSpread, linearScore,
} from "./geometry";

type NL = NormalizedLandmark;

export type Sign = "A" | "B" | "D" | "F" | "I" | "L" | "V" | "W" | "Y";
export const ALL_SIGNS: Sign[] = ["A", "B", "D", "F", "I", "L", "V", "W", "Y"];
export type ScoreMap = Record<Sign, number>;

// ── Letter scorers ────────────────────────────────────────────────────────────
//
// Every scorer receives the full 21-landmark array plus pre-computed
// palmHeight and palmWidth so those aren't recalculated per letter.
//
// Return value: confidence in [0, 1].

// ─── A ───────────────────────────────────────────────────────────────────────
/**
 * A — Closed fist, thumb resting alongside the index finger.
 *
 * Visual: All four fingers curl tightly into the palm.
 *         Thumb is on the side (not wrapped over the top, not splayed out).
 *
 * Key landmarks:
 *   fingertip.y > pip.y for every finger  → tip has dropped below mid-joint
 *   |thumbTip.x − indexMcp.x| small       → thumb is not spread to the side
 *
 * Math:
 *   curl(finger) = 1 − fingerExt(tip, mcp, palmH).
 *   fingerExt returns 0 when tip is at or below the knuckle → curl = 1 (good).
 *   Thumb spread < 0.30 normalised → thumb is alongside, not in L/Y territory.
 *
 * Avoids overlap with:
 *   B / V / W — those need extended fingers (extScore would be high, curlScore low).
 *   I / Y     — those need pinky extended.
 *   L         — L needs thumb spread far to the side.
 */
function scoreA(lm: NL[], palmH: number, palmW: number): number {
  const curlIdx = 1 - fingerExt(lm[8],  lm[5],  palmH);
  const curlMid = 1 - fingerExt(lm[12], lm[9],  palmH);
  const curlRng = 1 - fingerExt(lm[16], lm[13], palmH);
  const curlPky = 1 - fingerExt(lm[20], lm[17], palmH);

  // Soft veto: if any finger is visibly extended this can't be a fist.
  const minCurl = Math.min(curlIdx, curlMid, curlRng, curlPky);
  if (minCurl < 0.25) return minCurl * 0.4;

  const fingerScore = (curlIdx + curlMid + curlRng + curlPky) / 4;

  // Thumb should not splay — penalise if spread is large.
  const tSpread     = thumbSpread(lm, palmW);
  const thumbScore  = 1 - linearScore(tSpread, 0.15, 0.55);

  return 0.85 * fingerScore + 0.15 * thumbScore;
}

// ─── B ───────────────────────────────────────────────────────────────────────
/**
 * B — Four fingers straight up, thumb folded across the palm.
 *
 * Visual: Index, middle, ring, pinky all point up and are held together.
 *         Thumb crosses over the palm toward the index side (not visible from front).
 *
 * Key landmarks:
 *   All four fingertips clearly above their MCPs → high extension scores.
 *   thumbTip.x ≈ indexMcp.x → thumb hasn't moved sideways.
 *
 * Math:
 *   fingerExt = (mcp.y − tip.y) / palmHeight.
 *   For B, this should be ≥ 0.5 for all four fingers.
 *   Thumb spread < 0.30 normalised → thumb is tucked in, not opened.
 *
 * Avoids overlap with:
 *   W  — W has pinky curled; B's pinkyExt check separates them.
 *   V  — V has ring + pinky curled; B's ringExt + pinkyExt eliminate V.
 *   Open-5 — open hand has thumb spread; B's thumbScore penalises that.
 */
function scoreB(lm: NL[], palmH: number, palmW: number): number {
  const extIdx = fingerExt(lm[8],  lm[5],  palmH);
  const extMid = fingerExt(lm[12], lm[9],  palmH);
  const extRng = fingerExt(lm[16], lm[13], palmH);
  const extPky = fingerExt(lm[20], lm[17], palmH);

  const minExt = Math.min(extIdx, extMid, extRng, extPky);
  if (minExt < 0.30) return minExt * 0.5;

  const fingerScore = (extIdx + extMid + extRng + extPky) / 4;

  const tSpread    = thumbSpread(lm, palmW);
  const thumbScore = 1 - linearScore(tSpread, 0.10, 0.45);

  return 0.80 * fingerScore + 0.20 * thumbScore;
}

// ─── D ───────────────────────────────────────────────────────────────────────
/**
 * D — Index pointing up, middle/ring/pinky curled into a ball, thumb alongside them.
 *
 * Visual: Index finger extends upward (slightly curved in practice).
 *         Middle, ring, pinky curl together.
 *         Thumb touches or rests against the side of the middle finger, closing
 *         a circular shape with the index — the "D" profile.
 *
 * Key landmarks:
 *   indexTip well above indexMcp → high extension.
 *   middle/ring/pinkyTips below their PIPs → curled.
 *   dist(thumbTip, middlePip) small → thumb is near the middle finger.
 *
 * Math:
 *   thumbNearMiddle = 1 − linearScore(dist(lm[4], lm[10]) / palmW, 0.10, 0.45)
 *   This gives 1 when thumb tip is very close to middle PIP, 0 when far.
 *
 * Avoids overlap with:
 *   L — L requires thumb spread FAR to the side; D requires thumb NEAR middle.
 *   I — I has pinky up (not index) and thumb close to fist (different geometry).
 */
function scoreD(lm: NL[], palmH: number, palmW: number): number {
  const extIdx  = fingerExt(lm[8],  lm[5],  palmH);
  const curlMid = 1 - fingerExt(lm[12], lm[9],  palmH);
  const curlRng = 1 - fingerExt(lm[16], lm[13], palmH);
  const curlPky = 1 - fingerExt(lm[20], lm[17], palmH);

  if (extIdx < 0.35) return extIdx * 0.4;

  const otherCurl = (curlMid + curlRng + curlPky) / 3;
  if (otherCurl < 0.35) return otherCurl * 0.4;

  // Thumb tip distance to middle PIP, normalised.
  const thumbToMidPip    = dist(lm[4], lm[10]) / palmW;
  const thumbNearMiddle  = 1 - linearScore(thumbToMidPip, 0.10, 0.45);

  return 0.30 * extIdx + 0.40 * otherCurl + 0.30 * thumbNearMiddle;
}

// ─── F ───────────────────────────────────────────────────────────────────────
/**
 * F — Index + thumb pinch (touching tips), middle/ring/pinky extend upward.
 *
 * Visual: Index finger curls down and thumb comes up to meet it, forming a
 *         small circle (OK-like pinch) at the front.
 *         Middle, ring, and pinky fingers point straight up.
 *
 * Key landmarks:
 *   dist(thumbTip, indexTip) very small relative to palmWidth → pinch closed.
 *   middle/ring/pinky extension high.
 *   Index is NOT fully extended (it bends toward the thumb).
 *
 * Math:
 *   pinchDist = dist(lm[4], lm[8]) / palmW
 *   pinchScore = 1 − linearScore(pinchDist, 0.05, 0.30)
 *   High when tips are touching (< 5 % of palm width).
 *
 * Avoids overlap with:
 *   B — B has index fully extended, no pinch → pinchScore near 0 for B.
 *   W — W has index extended, ring also extended → different pattern.
 */
function scoreF(lm: NL[], palmH: number, palmW: number): number {
  const extMid  = fingerExt(lm[12], lm[9],  palmH);
  const extRng  = fingerExt(lm[16], lm[13], palmH);
  const extPky  = fingerExt(lm[20], lm[17], palmH);
  const extIdx  = fingerExt(lm[8],  lm[5],  palmH);

  const threeFingers = (extMid + extRng + extPky) / 3;
  if (threeFingers < 0.35) return threeFingers * 0.3;

  const pinchDist  = dist(lm[4], lm[8]) / palmW;
  const pinchScore = 1 - linearScore(pinchDist, 0.05, 0.30);
  if (pinchScore < 0.20) return pinchScore * 0.3;

  // Index should be curled (bent toward thumb, not straight up).
  const indexCurl = 1 - linearScore(extIdx, 0.0, 0.55);

  return 0.35 * threeFingers + 0.45 * pinchScore + 0.20 * indexCurl;
}

// ─── I ───────────────────────────────────────────────────────────────────────
/**
 * I — Only the pinky finger extended ("pinky power").
 *
 * Visual: Pinky points straight up.  Index, middle, ring curl into the palm.
 *         Thumb is held alongside the fist (not spread to the side).
 *
 * Key landmarks:
 *   pinkyTip well above pinkyMcp → high pinky extension.
 *   index/middle/ring curled → low extension scores.
 *   thumbSpread low → thumb not in Y-territory.
 *
 * Avoids overlap with:
 *   Y — Y also has pinky up BUT thumb is spread far to the side.
 *       The thumbScore term (penalising high spread) is the key separator.
 */
function scoreI(lm: NL[], palmH: number, palmW: number): number {
  const extPky  = fingerExt(lm[20], lm[17], palmH);
  const curlIdx = 1 - fingerExt(lm[8],  lm[5],  palmH);
  const curlMid = 1 - fingerExt(lm[12], lm[9],  palmH);
  const curlRng = 1 - fingerExt(lm[16], lm[13], palmH);

  if (extPky < 0.35) return extPky * 0.4;

  const otherCurl = (curlIdx + curlMid + curlRng) / 3;
  if (otherCurl < 0.30) return otherCurl * 0.4;

  const tSpread    = thumbSpread(lm, palmW);
  const thumbClose = 1 - linearScore(tSpread, 0.10, 0.55);

  return 0.35 * extPky + 0.50 * otherCurl + 0.15 * thumbClose;
}

// ─── L ───────────────────────────────────────────────────────────────────────
/**
 * L — Index pointing up, thumb extending sideways (the letter "L" shape).
 *
 * Visual: Index finger points straight up.  Thumb extends horizontally to the
 *         side forming the corner of an L.  Middle, ring, pinky curl in.
 *
 * Key landmarks:
 *   indexTip well above indexMcp → extension high.
 *   |thumbTip.x − indexMcp.x| / palmWidth large → thumb is far to the side.
 *   middle/ring/pinky curled.
 *
 * Math:
 *   thumbExtended = linearScore(thumbSpread, 0.35, 0.75)
 *   1.0 when the thumb tip is 75 % of a palm-width away from the index knuckle.
 *
 * Why mirroring doesn't matter: we take the absolute value of the x difference,
 * so the score is the same whether the user shows a left or right hand.
 *
 * Avoids overlap with:
 *   D — D has thumb near the middle finger, thumbSpread low → thumbExtended ≈ 0.
 *   Y — Y has pinky up instead of index, and index curled → extIdx low for Y.
 */
function scoreL(lm: NL[], palmH: number, palmW: number): number {
  const extIdx  = fingerExt(lm[8],  lm[5],  palmH);
  const curlMid = 1 - fingerExt(lm[12], lm[9],  palmH);
  const curlRng = 1 - fingerExt(lm[16], lm[13], palmH);
  const curlPky = 1 - fingerExt(lm[20], lm[17], palmH);

  if (extIdx < 0.35) return extIdx * 0.4;

  const otherCurl    = (curlMid + curlRng + curlPky) / 3;
  const tSpread      = thumbSpread(lm, palmW);
  const thumbOut     = linearScore(tSpread, 0.35, 0.75);

  if (thumbOut < 0.20) return thumbOut * 0.4;

  return 0.30 * extIdx + 0.35 * otherCurl + 0.35 * thumbOut;
}

// ─── V ───────────────────────────────────────────────────────────────────────
/**
 * V — Index and middle extended and spread apart (peace sign / scissors).
 *
 * Visual: Index and middle point up, spread into a V shape.
 *         Ring and pinky curl in.  Thumb is tucked or slightly open.
 *
 * Key landmarks:
 *   indexTip and middleTip both above their MCPs → both extended.
 *   ring and pinky curled.
 *   Horizontal gap between indexTip and middleTip > ~15 % of palm width.
 *
 * Math:
 *   fingerSpread = |indexTip.x − middleTip.x| / palmWidth
 *   spreadScore = linearScore(fingerSpread, 0.05, 0.25)
 *   This rewards the V shape's characteristic spread and helps avoid confusion
 *   with U (two fingers together), though U is not in our set.
 *
 * Avoids overlap with:
 *   B — B has ring + pinky extended; V's curlRng + curlPky check handles this.
 *   W — W has ring also extended; curlRng eliminates W when ring is up.
 */
function scoreV(lm: NL[], palmH: number, palmW: number): number {
  const extIdx  = fingerExt(lm[8],  lm[5],  palmH);
  const extMid  = fingerExt(lm[12], lm[9],  palmH);
  const curlRng = 1 - fingerExt(lm[16], lm[13], palmH);
  const curlPky = 1 - fingerExt(lm[20], lm[17], palmH);

  const minExt = Math.min(extIdx, extMid);
  if (minExt < 0.35) return minExt * 0.4;

  const curlScore = (curlRng + curlPky) / 2;
  if (curlScore < 0.25) return curlScore * 0.4;

  // The spread between index and middle distinguishes V from a collapsed 2-finger pose.
  const fingerSpread = Math.abs(lm[8].x - lm[12].x) / palmW;
  const spreadScore  = linearScore(fingerSpread, 0.05, 0.25);

  return 0.35 * ((extIdx + extMid) / 2) + 0.40 * curlScore + 0.25 * spreadScore;
}

// ─── W ───────────────────────────────────────────────────────────────────────
/**
 * W — Index, middle, ring all extended; pinky curled; thumb folded.
 *
 * Visual: Three middle fingers (index, middle, ring) fan open and point up,
 *         slightly spread to show the W shape.  Pinky curls down.
 *         Thumb is folded across the palm.
 *
 * Key landmarks:
 *   Index, middle, ring: high extension scores.
 *   Pinky: curled (low extension).
 *   thumbSpread low (thumb not flaring out).
 *
 * Avoids overlap with:
 *   B — B has pinky extended; W's curlPky check removes that ambiguity.
 *   V — V has ring curled; W's extRng check removes V when ring is up.
 */
function scoreW(lm: NL[], palmH: number, palmW: number): number {
  const extIdx  = fingerExt(lm[8],  lm[5],  palmH);
  const extMid  = fingerExt(lm[12], lm[9],  palmH);
  const extRng  = fingerExt(lm[16], lm[13], palmH);
  const curlPky = 1 - fingerExt(lm[20], lm[17], palmH);

  const minThree = Math.min(extIdx, extMid, extRng);
  if (minThree < 0.30) return minThree * 0.4;
  if (curlPky  < 0.30) return curlPky  * 0.4;

  const threeScore = (extIdx + extMid + extRng) / 3;
  const tSpread    = thumbSpread(lm, palmW);
  const thumbClose = 1 - linearScore(tSpread, 0.10, 0.50);

  return 0.60 * threeScore + 0.30 * curlPky + 0.10 * thumbClose;
}

// ─── Y ───────────────────────────────────────────────────────────────────────
/**
 * Y — Pinky extended up + thumb extended sideways (shaka / hang-loose).
 *
 * Visual: Pinky points up.  Thumb extends horizontally to the side.
 *         Index, middle, ring curl into the palm.
 *         The pinky and thumb together form the Y silhouette.
 *
 * Key landmarks:
 *   pinkyTip well above pinkyMcp → pinky extended.
 *   thumbSpread large → thumb is far to the side.
 *   index/middle/ring curled.
 *
 * Avoids overlap with:
 *   I — I has ONLY pinky extended but thumb NOT spread (thumbOut ≈ 0 for I).
 *       The thumbOut factor is the sole discriminator between I and Y.
 *   L — L has index extended (not pinky) and others curled.
 */
function scoreY(lm: NL[], palmH: number, palmW: number): number {
  const extPky  = fingerExt(lm[20], lm[17], palmH);
  const curlIdx = 1 - fingerExt(lm[8],  lm[5],  palmH);
  const curlMid = 1 - fingerExt(lm[12], lm[9],  palmH);
  const curlRng = 1 - fingerExt(lm[16], lm[13], palmH);

  if (extPky < 0.30) return extPky * 0.4;

  const otherCurl = (curlIdx + curlMid + curlRng) / 3;
  const tSpread   = thumbSpread(lm, palmW);
  const thumbOut  = linearScore(tSpread, 0.35, 0.70);

  if (thumbOut < 0.15) return thumbOut * 0.4;

  return 0.30 * extPky + 0.35 * otherCurl + 0.35 * thumbOut;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * scoreAllLetters — run every letter scorer and return a confidence map.
 *
 * Call this once per frame with the 21 landmarks of ONE hand.
 * Returns a Record mapping each Sign to its 0–1 confidence.
 * All scores will be 0 if the landmark array is incomplete.
 */
export function scoreAllLetters(lm: NL[]): ScoreMap {
  if (lm.length < 21) {
    return Object.fromEntries(ALL_SIGNS.map(s => [s, 0])) as ScoreMap;
  }
  const palmH = palmHeight(lm);
  const palmW = palmWidth(lm);
  return {
    A: scoreA(lm, palmH, palmW),
    B: scoreB(lm, palmH, palmW),
    D: scoreD(lm, palmH, palmW),
    F: scoreF(lm, palmH, palmW),
    I: scoreI(lm, palmH, palmW),
    L: scoreL(lm, palmH, palmW),
    V: scoreV(lm, palmH, palmW),
    W: scoreW(lm, palmH, palmW),
    Y: scoreY(lm, palmH, palmW),
  };
}
