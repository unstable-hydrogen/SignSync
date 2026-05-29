import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

type NL = NormalizedLandmark;

// Adapted from snrao ASL repo: gestureStatic pattern.
// That code uses cv2.matchShapes on contours — only runs identifyGesture()
// when the contour has been stable for 10 consecutive frames.
//
// Here: track movement of key landmarks between frames.
// Only return true (allow classification) when the hand has been still for
// STABLE_FRAMES in a row. Resets whenever movement exceeds the threshold.

const STABLE_FRAMES = 8;          // ~270 ms at 30 fps (snrao uses 10)
const MOVEMENT_THRESHOLD = 0.014; // max per-landmark delta in MediaPipe coords [0,1]

// Subset of the 21 landmarks that best capture overall hand motion:
// wrist, index/middle/pinky MCP, index/middle/pinky tip
const TRACK = [0, 5, 9, 17, 8, 12, 20];

export class StabilityGate {
  private prev: NL[] | null = null;
  private count = 0;

  /**
   * Feed each frame's raw landmarks here.
   * Returns true only when the hand has been sufficiently still
   * for STABLE_FRAMES consecutive frames — safe to classify.
   */
  check(lm: NL[]): boolean {
    if (!this.prev) {
      this.prev = lm;
      this.count = 0;
      return false;
    }

    const moved = maxMovement(lm, this.prev);
    this.prev = lm;

    if (moved > MOVEMENT_THRESHOLD) {
      this.count = 0;
      return false;
    }

    this.count = Math.min(this.count + 1, STABLE_FRAMES);
    return this.count >= STABLE_FRAMES;
  }

  reset() {
    this.prev = null;
    this.count = 0;
  }
}

function maxMovement(a: NL[], b: NL[]): number {
  let maxSq = 0;
  for (const i of TRACK) {
    if (i >= a.length || i >= b.length) continue;
    const dx = a[i].x - b[i].x;
    const dy = a[i].y - b[i].y;
    const sq = dx * dx + dy * dy;
    if (sq > maxSq) maxSq = sq;
  }
  return Math.sqrt(maxSq);
}
