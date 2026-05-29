import { useState, useRef, useCallback } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export interface CollectorState {
  active:  boolean;
  target:  string | null;
  counts:  Record<string, number>;
  total:   number;
}

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const CSV_HEADER =
  LETTERS.flatMap(
    (_, i) => [`x${i}`, `y${i}`, `z${i}`]
  ).join(",");

export function useCollector() {
  const [active, setActive]   = useState(false);
  const [target, setTarget]   = useState<string | null>(null);
  const [counts, setCounts]   = useState<Record<string, number>>(
    () => Object.fromEntries(LETTERS.map(l => [l, 0]))
  );
  const [total, setTotal]     = useState(0);

  // Rows accumulated in memory — each is a CSV line "LABEL,x0,y0,z0,...,x20,y20,z20"
  const rowsRef = useRef<string[]>([]);

  const toggle = useCallback(() => {
    setActive(a => !a);
  }, []);

  const pickTarget = useCallback((letter: string) => {
    if (LETTERS.includes(letter.toUpperCase())) {
      setTarget(letter.toUpperCase());
    }
  }, []);

  /** Capture raw landmarks for the current target letter. Returns true on success. */
  const capture = useCallback((lm: NormalizedLandmark[]): boolean => {
    if (!target || lm.length < 21) return false;

    const vals = lm.flatMap(p => [p.x, p.y, p.z ?? 0]);
    rowsRef.current.push(`${target},${vals.map(v => v.toFixed(6)).join(",")}`);

    setCounts(prev => ({ ...prev, [target]: (prev[target] ?? 0) + 1 }));
    setTotal(t => t + 1);
    return true;
  }, [target]);

  /** Download all collected samples as a CSV file. */
  const exportCSV = useCallback(() => {
    if (rowsRef.current.length === 0) return;
    const csv  = `label,${CSV_HEADER}\n${rowsRef.current.join("\n")}`;
    const blob = new Blob([csv], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `asl_landmarks_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  /** Remove the most recently captured sample. */
  const undo = useCallback(() => {
    if (rowsRef.current.length === 0) return;
    const last = rowsRef.current.pop()!;
    const label = last.split(",")[0];
    setCounts(prev => ({ ...prev, [label]: Math.max(0, (prev[label] ?? 1) - 1) }));
    setTotal(t => Math.max(0, t - 1));
  }, []);

  return {
    state: { active, target, counts, total } as CollectorState,
    toggle,
    pickTarget,
    capture,
    exportCSV,
    undo,
  };
}
