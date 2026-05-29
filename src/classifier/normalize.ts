import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

type NL = NormalizedLandmark;

/**
 * Z-score normalize all 21 hand landmarks independently per axis.
 *
 * Adapted from the Kaggle ASL 1st-place solution's per-body-part normalization:
 *   for each axis (x, y, z): mean = 0, std = 1 across all 21 points.
 *
 * This makes every downstream geometric ratio (fingerExt, thumbSpreadRatio, etc.)
 * invariant to:
 *   - Hand position on screen  (mean shift removed)
 *   - Distance from camera     (std scaling removed for x and y)
 *   - Hand size differences    (std scaling removed)
 *
 * Ratios like (mcp.y − tip.y) / palmHeight are preserved under uniform scaling,
 * so all existing feature thresholds remain valid after normalization.
 */
export function normalizeHand(lm: NL[]): NL[] {
  const xs = lm.map(p => p.x);
  const ys = lm.map(p => p.y);
  const zs = lm.map(p => p.z ?? 0);

  zscoreInPlace(xs);
  zscoreInPlace(ys);
  zscoreInPlace(zs);

  return lm.map((p, i) => ({ x: xs[i], y: ys[i], z: zs[i], visibility: p.visibility }));
}

function zscoreInPlace(arr: number[]): void {
  const n = arr.length;
  const mean = arr.reduce((s, v) => s + v, 0) / n;
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  if (std < 1e-6) return;
  for (let i = 0; i < n; i++) arr[i] = (arr[i] - mean) / std;
}
