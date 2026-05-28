import { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

type Status = "loading" | "ready" | "error";

export function useHandTracking(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  onLandmarks?: (landmarks: NormalizedLandmark[][]) => void
) {
  const [status, setStatus] = useState<Status>("loading");
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);

  // Keep a stable ref to onLandmarks so the animation loop always sees
  // the latest version without needing to re-run the effect.
  const onLandmarksRef = useRef(onLandmarks);
  useEffect(() => { onLandmarksRef.current = onLandmarks; }, [onLandmarks]);

  useEffect(() => {
    let animId: number;
    let active = true;
    let landmarker: HandLandmarker | null = null;
    let stream: MediaStream | null = null;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        });

        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });

        const video = videoRef.current;
        if (!video || !active) return;

        video.srcObject = stream;
        await video.play();

        if (!active) return;
        setStatus("ready");
        renderLoop();
      } catch (err) {
        console.error("Failed to initialize:", err);
        if (active) setStatus("error");
      }
    }

    function renderLoop() {
      if (!active) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || !landmarker) {
        animId = requestAnimationFrame(renderLoop);
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        animId = requestAnimationFrame(renderLoop);
        return;
      }

      if (!drawingUtilsRef.current) {
        drawingUtilsRef.current = new DrawingUtils(ctx);
      }

      const results = landmarker.detectForVideo(video, performance.now());

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      for (const landmarks of results.landmarks) {
        drawingUtilsRef.current.drawConnectors(
          landmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#00FF00", lineWidth: 2 }
        );
        drawingUtilsRef.current.drawLandmarks(landmarks, {
          color: "#FF0000",
          lineWidth: 1,
          radius: 4,
        });
      }

      onLandmarksRef.current?.(results.landmarks);

      animId = requestAnimationFrame(renderLoop);
    }

    init();

    return () => {
      active = false;
      cancelAnimationFrame(animId);
      landmarker?.close();
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [videoRef, canvasRef]);

  return status;
}
