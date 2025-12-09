import React, { useRef, useState } from "react";

export default function CameraCapture() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [streaming, setStreaming] = useState(false);
  const [ocrText, setOcrText] = useState("");
  const [preds, setPreds] = useState([]);
  const [loading, setLoading] = useState(false);

  const backendURL = "http://127.0.0.1:8000/api/predict_camera"; // your FastAPI backend

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      videoRef.current.srcObject = stream;
      setStreaming(true);
    } catch (err) {
      alert("Camera error: " + err.message);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
    }
    setStreaming(false);
  };

  const captureImage = async () => {
    if (!streaming) return;

    setLoading(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob
    canvas.toBlob(
      async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");
        formData.append("topk", "10");

        try {
          const res = await fetch(backendURL, {
            method: "POST",
            body: formData,
          });

          const data = await res.json();

          if (data.error) {
            setOcrText("Error: " + data.error);
            setPreds([]);
          } else {
            setOcrText(data.ocr_text || "");
            setPreds(data.predictions || []);
          }
        } catch (err) {
          setOcrText("Network error: " + err.message);
        }

        setLoading(false);
      },
      "image/jpeg",
      0.9
    );
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Camera Disease Predictor</h2>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "100%", maxWidth: "600px", borderRadius: "8px" }}
      ></video>

      <canvas
        ref={canvasRef}
        style={{ display: "none" }}
      ></canvas>

      <div style={{ marginTop: 10 }}>
        {!streaming ? (
          <button onClick={startCamera}>Start Camera</button>
        ) : (
          <button onClick={stopCamera}>Stop Camera</button>
        )}

        <button onClick={captureImage} disabled={!streaming || loading} style={{ marginLeft: 10 }}>
          {loading ? "Processing..." : "Capture & Predict"}
        </button>
      </div>

      <div style={{ marginTop: 20, background: "#f8f8f8", padding: 15, borderRadius: 8 }}>
        <h3>OCR Text</h3>
        <pre style={{ whiteSpace: "pre-wrap" }}>{ocrText || "No OCR yet"}</pre>

        <h3>Predictions</h3>
        <ul>
          {preds.length === 0 && <li>No predictions yet</li>}
          {preds.map((p, i) => (
            <li key={i}>
              <b>{p.label}</b> — {p.score.toFixed(4)}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
