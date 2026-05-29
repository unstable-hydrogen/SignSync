/**
 * classifySign.ts — Gesture Classification Layer
 *
 * Receives a pre-extracted HandFeatures struct (one per frame) and returns a
 * confidence score in [0, 1] for each supported ASL letter.
 *
 * ── Supported letters ───────────────────────────────────────────────────────
 *   A  B  D  F  I  L  V  W  Y
 *   Chosen because their hand shapes are geometrically distinct enough
 *   for heuristic separation without an ML model.
 *
 * ── Confidence scoring pattern ──────────────────────────────────────────────
 * Each scorer computes several sub-scores (0–1) for individual geometric
 * features, then returns their weighted average.
 *
 * "Soft veto" pattern:
 *   if (primaryFeature < LOW_THRESHOLD) return primaryFeature * PENALTY
 *   This caps the whole score when a defining feature is clearly absent,
 *   without forcing a hard binary decision that ignores partial matches.
 *
 * ── Strongest geometric features for ASL ────────────────────────────────────
 *   1. Finger extension (most discriminative — separates A/fist from B/4-up)
 *   2. Thumb spread (separates I from Y, D from L, A from L)
 *   3. Pinch distance (unique to F — thumb+index touching)
 *   4. Which fingers are extended (V=2, W=3, B=4)
 *   5. PIP-based curl (robust cross-check for tilted hands)
 *
 * ── Letters that commonly overlap ───────────────────────────────────────────
 *   I ↔ Y  — identical except thumb position (Y has thumb splayed)
 *   D ↔ L  — identical except thumb position (L spread, D near middle)
 *   V ↔ W  — V has ring curled, W has ring extended; one-finger difference
 *   B ↔ W  — B has pinky extended, W has pinky curled; one-finger difference
 */

import { linearScore } from "./geometry";
import type { HandFeatures } from "./features";

export type Sign =
  "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" |
  "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" |
  "U" | "V" | "W" | "X" | "Y" | "Z";
export const ALL_SIGNS: Sign[] = [
  "A","B","C","D","E","F","G","H","I","J",
  "K","L","M","N","O","P","Q","R","S","T",
  "U","V","W","X","Y","Z",
];
export type ScoreMap = Record<Sign, number>;

/** All-zero scores — used when no hand is detected so EMA decays naturally. */
export const ZERO_SCORES: ScoreMap =
  Object.fromEntries(ALL_SIGNS.map(s => [s, 0])) as ScoreMap;

// ── Letter scorers ────────────────────────────────────────────────────────────

/**
 * A — Closed fist, thumb resting alongside the index finger.
 *
 * Visual logic:
 *   All four fingers curl tightly into the palm.  Thumb rests on the side,
 *   neither wrapped over the top (S) nor splayed outward (L/Y).
 *
 * Key features:
 *   • All four fingerCurl scores high (tipBelowPip adds a robustness cross-check)
 *   • thumbSpreadRatio low — thumb is not flaring to the side
 *
 * Avoids overlap with:
 *   B/V/W  — those require extended fingers (fingerExt > 0.5)
 *   I/Y    — those require pinky extended (pinkyCurl would be low)
 *   L      — L needs thumbSpreadRatio > 0.4 (A penalises high spread)
 */
function scoreA(f: HandFeatures): number {
  const { indexCurl, middleCurl, ringCurl, pinkyCurl,
          indexTipBelowPip, middleTipBelowPip, ringTipBelowPip, pinkyTipBelowPip,
          thumbSpreadRatio, thumbIndexPinch } = f;

  // Primary: all four fingers must be curled
  const minCurl = Math.min(indexCurl, middleCurl, ringCurl, pinkyCurl);
  if (minCurl < 0.20) return minCurl * 0.35; // soft veto

  // If thumb tip is close to index tip the pose is O, not A.
  // A's thumb rests on the side of the index — it does NOT reach the fingertip.
  if (thumbIndexPinch < 0.22) return thumbIndexPinch * 1.8; // hard cap ≈ 0.40

  const curlScore = (indexCurl + middleCurl + ringCurl + pinkyCurl) / 4;

  // Secondary: PIP-based binary check — cross-validates the continuous score
  const pipScore = (
    (indexTipBelowPip  ? 1 : 0) +
    (middleTipBelowPip ? 1 : 0) +
    (ringTipBelowPip   ? 1 : 0) +
    (pinkyTipBelowPip  ? 1 : 0)
  ) / 4;

  // Thumb: should not be splayed sideways (distinguishes A from L)
  const thumbScore = 1 - linearScore(thumbSpreadRatio, 0.15, 0.55);

  return 0.55 * curlScore + 0.30 * pipScore + 0.15 * thumbScore;
}

/**
 * B — Four fingers straight up, thumb folded across the palm.
 *
 * Visual logic:
 *   Index, middle, ring, pinky all point straight up and are held together.
 *   Thumb crosses the palm (not visible from the front).
 *
 * Key features:
 *   • All four fingerExt scores high
 *   • thumbSpreadRatio low — thumb is across the palm, not to the side
 *
 * Avoids overlap with:
 *   W  — W has pinky curled; B's pinkyExt check handles this
 *   V  — V has ring+pinky curled; B's ringExt+pinkyExt eliminate V
 *   Open-5 — open hand has thumb spread outward; B's thumbScore penalises that
 */
function scoreB(f: HandFeatures): number {
  const { indexExt, middleExt, ringExt, pinkyExt, thumbSpreadRatio } = f;

  const minExt = Math.min(indexExt, middleExt, ringExt, pinkyExt);
  if (minExt < 0.28) return minExt * 0.45;

  const fingerScore = (indexExt + middleExt + ringExt + pinkyExt) / 4;
  const thumbScore  = 1 - linearScore(thumbSpreadRatio, 0.10, 0.45);

  return 0.80 * fingerScore + 0.20 * thumbScore;
}

/**
 * D — Index pointing up, others curl into a ball, thumb touches middle finger.
 *
 * Visual logic:
 *   Index extends upward.  Middle, ring, and pinky curl together into a ball.
 *   Thumb tip touches or rests against the middle finger's PIP area,
 *   closing a circle with the base of the index — the "D" profile.
 *
 * Key features:
 *   • indexExt high
 *   • middle/ring/pinky curl high
 *   • thumbMiddleDist low — thumb is near the curled middle finger
 *
 * Avoids overlap with:
 *   L  — L requires thumbSpreadRatio high (far from palm); D requires thumb NEAR middle
 *   I  — I has pinky extended (not index), and different thumb position
 */
function scoreD(f: HandFeatures): number {
  const { indexExt, middleCurl, ringCurl, pinkyCurl, thumbMiddleDist } = f;

  if (indexExt < 0.35) return indexExt * 0.4;

  const otherCurl = (middleCurl + ringCurl + pinkyCurl) / 3;
  if (otherCurl < 0.30) return otherCurl * 0.4;

  // Thumb near middle PIP — the defining circle of D
  const thumbNearMiddle = 1 - linearScore(thumbMiddleDist, 0.10, 0.50);

  return 0.28 * indexExt + 0.42 * otherCurl + 0.30 * thumbNearMiddle;
}

/**
 * F — Index+thumb pinch (touching tips), middle/ring/pinky extend upward.
 *
 * Visual logic:
 *   The index curls down and the thumb meets it, forming a small circle
 *   (like an OK sign but with the other three fingers straight up).
 *
 * Key features:
 *   • thumbIndexPinch low — the two tips are close together
 *   • middle/ring/pinky extended
 *   • index NOT fully extended (it bends toward the thumb)
 *
 * Avoids overlap with:
 *   B  — B has index extended, no pinch; pinchScore near 0 for B
 *   W  — W has index extended with ring also up; different pattern entirely
 */
function scoreF(f: HandFeatures): number {
  const { middleExt, ringExt, pinkyExt, indexExt, thumbIndexPinch } = f;

  const threeFingers = (middleExt + ringExt + pinkyExt) / 3;
  if (threeFingers < 0.35) return threeFingers * 0.3;

  // Pinch: thumb tip close to index tip
  const pinchScore = 1 - linearScore(thumbIndexPinch, 0.04, 0.28);
  if (pinchScore < 0.20) return pinchScore * 0.3;

  // Index should be partially curled (bending toward thumb, not straight up)
  const indexCurled = 1 - linearScore(indexExt, 0.0, 0.60);

  return 0.35 * threeFingers + 0.45 * pinchScore + 0.20 * indexCurled;
}

/**
 * I — Only the pinky extended, others curled, thumb tucked.
 *
 * Visual logic:
 *   Pinky points straight up; all other fingers and thumb are folded into a fist.
 *   Also called "pinky salute" or the letter I in one-handed fingerspelling.
 *
 * Key features:
 *   • pinkyExt high
 *   • index/middle/ring all curled
 *   • thumbSpreadRatio low — critical for separating I from Y
 *
 * Avoids overlap with:
 *   Y  — Y also has pinky extended BUT thumb is splayed sideways.
 *         The (1 − thumbSpreadRatio) term is the sole discriminator.
 */
function scoreI(f: HandFeatures): number {
  const { pinkyExt, indexCurl, middleCurl, ringCurl, thumbSpreadRatio } = f;

  if (pinkyExt < 0.35) return pinkyExt * 0.4;

  const otherCurl = (indexCurl + middleCurl + ringCurl) / 3;
  if (otherCurl < 0.28) return otherCurl * 0.4;

  // Thumb must NOT be spread — this is what separates I from Y
  const thumbClose = 1 - linearScore(thumbSpreadRatio, 0.10, 0.55);

  return 0.35 * pinkyExt + 0.50 * otherCurl + 0.15 * thumbClose;
}

/**
 * L — Index pointing up, thumb extending horizontally to the side.
 *
 * Visual logic:
 *   Index points up, thumb extends to the side, forming a 90° "L" shape.
 *   Middle, ring, and pinky curl into the palm.
 *
 * Key features:
 *   • indexExt high
 *   • thumbSpreadRatio large — thumb is far from the palm
 *   • middle/ring/pinky curled
 *
 * Why mirroring doesn't matter:
 *   thumbSpreadRatio = |thumbTip.x − indexMcp.x| / palmW.
 *   The absolute value makes this invariant to left/right hand orientation.
 *
 * Avoids overlap with:
 *   D  — D has thumbMiddleDist low (thumb near middle); L has thumbSpreadRatio high
 *   Y  — Y has pinky up (not index); indexCurl would be high for Y
 */
function scoreL(f: HandFeatures): number {
  const { indexExt, middleCurl, ringCurl, pinkyCurl, thumbSpreadRatio } = f;

  if (indexExt < 0.35) return indexExt * 0.4;

  const otherCurl  = (middleCurl + ringCurl + pinkyCurl) / 3;
  const thumbOut   = linearScore(thumbSpreadRatio, 0.35, 0.78);

  if (thumbOut < 0.20) return thumbOut * 0.4;

  return 0.28 * indexExt + 0.37 * otherCurl + 0.35 * thumbOut;
}

/**
 * V — Index and middle extended and spread apart (peace / scissors).
 *
 * Visual logic:
 *   Index and middle point up and fan apart into a V shape.
 *   Ring and pinky curl in.  Thumb is tucked or neutral.
 *
 * Key features:
 *   • indexExt and middleExt both high
 *   • ringCurl and pinkyCurl high
 *   • indexMiddleSpread — horizontal gap between the two extended tips
 *     distinguishes V (spread) from U (together), though U is not in our set
 *
 * Avoids overlap with:
 *   B  — B has ring+pinky also extended; V's curlRng+curlPky check handles this
 *   W  — W has ring extended; V's ringCurl check eliminates W
 */
function scoreV(f: HandFeatures): number {
  const { indexExt, middleExt, ringCurl, pinkyCurl, indexMiddleSpread } = f;

  const minTwoExt = Math.min(indexExt, middleExt);
  if (minTwoExt < 0.35) return minTwoExt * 0.4;

  const curlScore   = (ringCurl + pinkyCurl) / 2;
  if (curlScore < 0.25) return curlScore * 0.4;

  // Spread between index and middle — rewards the characteristic V gap
  const spreadScore = linearScore(indexMiddleSpread, 0.05, 0.28);

  return 0.35 * ((indexExt + middleExt) / 2) + 0.40 * curlScore + 0.25 * spreadScore;
}

/**
 * W — Index, middle, ring all extended; pinky curled; thumb folded.
 *
 * Visual logic:
 *   Three fingers (index, middle, ring) fan upward slightly.
 *   Pinky curls in.  Thumb stays close to the palm.
 *
 * Key features:
 *   • indexExt, middleExt, ringExt all high
 *   • pinkyCurl high — the key difference from B (which has all four)
 *   • thumbSpreadRatio low (thumb not flaring)
 *
 * Avoids overlap with:
 *   B  — B has pinky extended; W's pinkyCurl check handles this
 *   V  — V has ring curled; W's ringExt check eliminates V when ring is up
 */
function scoreW(f: HandFeatures): number {
  const { indexExt, middleExt, ringExt, pinkyCurl, thumbSpreadRatio } = f;

  const minThree = Math.min(indexExt, middleExt, ringExt);
  if (minThree < 0.28) return minThree * 0.4;
  if (pinkyCurl  < 0.28) return pinkyCurl  * 0.4;

  const threeScore = (indexExt + middleExt + ringExt) / 3;
  const thumbClose = 1 - linearScore(thumbSpreadRatio, 0.10, 0.50);

  return 0.60 * threeScore + 0.30 * pinkyCurl + 0.10 * thumbClose;
}

/**
 * Y — Pinky extended up + thumb spread sideways (shaka / hang-loose).
 *
 * Visual logic:
 *   Pinky points up, thumb extends horizontally to the side.
 *   Index, middle, ring curl into the palm.
 *   The pinky and thumb together trace the Y shape.
 *
 * Key features:
 *   • pinkyExt high
 *   • thumbSpreadRatio large — this is what distinguishes Y from I
 *   • index/middle/ring curled
 *
 * Avoids overlap with:
 *   I  — I has ONLY pinky up with thumb NOT spread; thumbSpreadRatio low for I
 *   L  — L has index up (not pinky); indexCurl would be high for L hand poses
 */
function scoreY(f: HandFeatures): number {
  const { pinkyExt, indexCurl, middleCurl, ringCurl, thumbSpreadRatio } = f;

  if (pinkyExt < 0.30) return pinkyExt * 0.4;

  const otherCurl = (indexCurl + middleCurl + ringCurl) / 3;
  const thumbOut  = linearScore(thumbSpreadRatio, 0.35, 0.72);

  if (thumbOut < 0.15) return thumbOut * 0.4;

  return 0.30 * pinkyExt + 0.35 * otherCurl + 0.35 * thumbOut;
}

/**
 * C — All four fingers arc into a gentle curve (like holding a ball).
 *
 * Visual logic:
 *   None of the fingers are fully extended or fully curled.
 *   They all rest in the 0.25–0.70 extension range, tracing the arc of the letter C.
 *   The thumb extends outward to complete the curve, giving a medium spread ratio.
 *
 * Key features:
 *   • All four fingerExt in mid-range (not a fist, not a flat hand)
 *   • thumbSpreadRatio moderate — thumb arcs away to match the curve
 *
 * Avoids overlap with:
 *   A/S  — those have all fingers fully curled (ext ≈ 0)
 *   B    — B has all fingers fully extended (ext ≈ 1)
 *   O    — O has all fingers tightly curled AND thumb meeting index
 */
function scoreC(f: HandFeatures): number {
  const { indexExt, middleExt, ringExt, pinkyExt, thumbSpreadRatio } = f;

  const avgExt = (indexExt + middleExt + ringExt + pinkyExt) / 4;
  const maxExt = Math.max(indexExt, middleExt, ringExt, pinkyExt);
  const minExt = Math.min(indexExt, middleExt, ringExt, pinkyExt);

  // Soft veto: too flat (B-like) or too closed (A-like)
  if (maxExt < 0.15) return maxExt * 0.3;
  if (minExt > 0.80) return (1 - minExt) * 0.3;

  // Best C when avg extension is near the midpoint (≈0.40–0.55)
  const midScore = 1 - linearScore(Math.abs(avgExt - 0.45), 0.0, 0.28);

  // Thumb arcs out moderately
  const thumbScore = linearScore(thumbSpreadRatio, 0.18, 0.60);

  return 0.65 * midScore + 0.35 * thumbScore;
}

/**
 * O — All fingers tightly curled, thumb tip meets index tip to close a circle.
 *
 * Visual logic:
 *   Every finger curls inward, and the thumb rounds up to touch the index fingertip,
 *   forming a tight closed circle — the letter O.
 *
 * Key features:
 *   • All four fingerCurl high (≥ 0.55) — tighter than C
 *   • thumbIndexPinch very low — thumb tip is touching index tip
 *
 * Avoids overlap with:
 *   A    — A also has tight curl, but thumb rests on the side (no pinch)
 *   C    — C has fingers in mid-range extension, not tight curl
 *   F    — F only curls the index; middle/ring/pinky extend
 */
function scoreO(f: HandFeatures): number {
  const { indexCurl, middleCurl, ringCurl, pinkyCurl, thumbIndexPinch } = f;

  const minCurl = Math.min(indexCurl, middleCurl, ringCurl, pinkyCurl);
  if (minCurl < 0.28) return minCurl * 0.4; // fingers clearly not curled

  // Thumb rounds up to meet index tip — the defining feature of O.
  // Range widened: thumb doesn't need to be perfectly touching, just clearly close.
  const pinchScore = 1 - linearScore(thumbIndexPinch, 0.04, 0.50);
  if (pinchScore < 0.10) return pinchScore * 0.3;

  const curlScore = (indexCurl + middleCurl + ringCurl + pinkyCurl) / 4;

  return 0.40 * curlScore + 0.60 * pinchScore;
}

/**
 * U — Index and middle extended close together (no V-spread); ring and pinky curled.
 *
 * Visual logic:
 *   Exactly like V but with index and middle held parallel and touching,
 *   rather than fanned apart into a V shape.
 *
 * Key features:
 *   • indexExt and middleExt both high
 *   • ringCurl and pinkyCurl high (same as V)
 *   • indexMiddleSpread very LOW — the defining difference from V
 *
 * Avoids overlap with:
 *   V  — V has indexMiddleSpread high (fingers fanned); the spread term discriminates
 *   B  — B also has ring+pinky extended; U's curlScore check handles this
 */
function scoreU(f: HandFeatures): number {
  const { indexExt, middleExt, ringCurl, pinkyCurl, indexMiddleSpread, thumbSpreadRatio } = f;

  const minTwoExt = Math.min(indexExt, middleExt);
  if (minTwoExt < 0.40) return minTwoExt * 0.4;

  const curlScore = (ringCurl + pinkyCurl) / 2;
  if (curlScore < 0.28) return curlScore * 0.4;

  // Fingers must be held together — this is what separates U from V
  const togetherScore = 1 - linearScore(indexMiddleSpread, 0.03, 0.20);
  if (togetherScore < 0.15) return togetherScore * 0.4;

  // Thumb should not be splayed (distinguishes U from L)
  const thumbClose = 1 - linearScore(thumbSpreadRatio, 0.12, 0.50);

  return 0.30 * ((indexExt + middleExt) / 2)
       + 0.28 * curlScore
       + 0.30 * togetherScore
       + 0.12 * thumbClose;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * scoreAllLetters — run every letter scorer and return a confidence map.
 *
 * Input:  HandFeatures extracted from one hand's 21 landmarks.
 * Output: ScoreMap — confidence in [0, 1] for every supported letter.
 *
 * Call this once per animation frame per detected hand.
 */
export function scoreAllLetters(f: HandFeatures): ScoreMap {
  const zero = 0;
  return {
    A: scoreA(f), B: scoreB(f), C: scoreC(f),
    D: scoreD(f), E: zero,      F: scoreF(f),
    G: zero,      H: zero,      I: scoreI(f),
    J: zero,      K: zero,      L: scoreL(f),
    M: zero,      N: zero,      O: scoreO(f),
    P: zero,      Q: zero,      R: zero,
    S: zero,      T: zero,      U: scoreU(f),
    V: scoreV(f), W: scoreW(f), X: zero,
    Y: scoreY(f), Z: zero,
  };
}
