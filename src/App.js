// frontend/src/App.js (example)
import React from "react";
import CameraCapture from "./components/CameraCapture";

function App() {
  return (
    <div className="App">
      <h1>Lab Camera Predictor</h1>
      <CameraCapture apiUrl="/api/predict_camera" topk={10} />
    </div>
  );
}

export default App;
