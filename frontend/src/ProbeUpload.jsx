import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000"; // FastAPI backend

export default function ProbeUpload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [result, setResult] = useState(null);

  // Upload image to backend
  const uploadProbe = async () => {
    if (!file) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setStatus("Uploading image...");

    await fetch(`${API_BASE}/upload_probe`, {
      method: "POST",
      body: formData
    });

    setStatus("Searching in live feed...");
    setResult(null);
  };

  // Poll backend for result
  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch(`${API_BASE}/probe_status`);
      const data = await res.json();

      if (data.matched) {
        setResult(data);
        setStatus("MATCH FOUND");
      }
    }, 500); // poll every 500ms

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded shadow">
      <h2 className="text-xl font-bold mb-4">Person Search</h2>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-4"
      />

      <button
        onClick={uploadProbe}
        className="bg-blue-600 text-white px-4 py-2 rounded"
      >
        Search Person
      </button>

      <p className="mt-4 font-semibold">{status}</p>

      {result && (
        <div className="mt-4 p-3 border rounded bg-green-100">
          <p>✅ Person ID: {result.person_id}</p>
          <p>Similarity: {result.similarity}</p>
        </div>
      )}
    </div>
  );
}
