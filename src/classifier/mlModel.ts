/**
 * mlModel.ts — MLP inference engine for the ASL landmark classifier.
 *
 * The model was trained with scikit-learn MLPClassifier on MediaPipe
 * landmarks extracted from the snrao ASL dataset (a-z, ~1260 samples).
 * Weights are stored in public/asl_model/model_weights.json.
 *
 * Forward pass: for each layer apply ReLU(x @ W + b), softmax on final layer.
 * This mirrors sklearn's internal forward pass exactly.
 */

import type { ScoreMap, Sign } from "./classifySign";
import { ALL_SIGNS, ZERO_SCORES } from "./classifySign";

interface Weights {
  coefs:      number[][][];  // [layer][in][out]
  intercepts: number[][];    // [layer][out]
  labels:     string[];      // 26 uppercase letters
}

let weights: Weights | null = null;
let loadPromise: Promise<void> | null = null;

export function loadModel(): Promise<void> {
  if (loadPromise) return loadPromise;
  loadPromise = fetch("/asl_model/model_weights.json")
    .then(r => r.json())
    .then((w: Weights) => { weights = w; console.log("ASL MLP model loaded"); })
    .catch(e => console.warn("ASL model unavailable, using heuristics:", e.message));
  return loadPromise;
}

export function isModelReady(): boolean {
  return weights !== null;
}

/**
 * Run the MLP forward pass on pre-normalised landmarks.
 * Input: Float32Array[63]  — z-scored x0..x20, y0..y20, z0..z20
 * Output: ScoreMap — per-letter confidence in [0, 1]
 */
export function predictLandmarks(flat: Float32Array): ScoreMap {
  if (!weights) return { ...ZERO_SCORES };

  let x: number[] = Array.from(flat);

  const nLayers = weights.coefs.length;
  for (let l = 0; l < nLayers; l++) {
    const W = weights.coefs[l];       // [inDim][outDim]
    const b = weights.intercepts[l];  // [outDim]
    const outDim = b.length;
    const next = new Array<number>(outDim);

    for (let j = 0; j < outDim; j++) {
      let sum = b[j];
      for (let i = 0; i < x.length; i++) sum += x[i] * W[i][j];
      // ReLU on all hidden layers; final layer uses softmax below
      next[j] = l < nLayers - 1 ? Math.max(0, sum) : sum;
    }
    x = next;
  }

  // Softmax on logits
  const maxLogit = Math.max(...x);
  const exps = x.map(v => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sumExp);

  // J and Z are movement-based letters (trace a path in the air) and cannot
  // be detected from a single static frame — exclude them from output.
  const MOVEMENT_ONLY = new Set(["J", "Z"]);

  const result: ScoreMap = { ...ZERO_SCORES };
  for (let i = 0; i < weights.labels.length; i++) {
    const sign = weights.labels[i] as Sign;
    if (ALL_SIGNS.includes(sign) && !MOVEMENT_ONLY.has(sign)) {
      result[sign] = probs[i];
    }
  }
  return result;
}
