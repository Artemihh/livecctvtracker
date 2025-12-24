import { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [result, setResult] = useState(null);

  async function upload() {
    if (!file) return alert("Select image");

    setStatus("Uploading probe...");
    setResult(null);

    const fd = new FormData();
    fd.append("file", file);

    await fetch("http://127.0.0.1:8000/upload_probe", {
      method: "POST",
      body: fd,
    });

    setStatus("Searching in live feed...");

    let elapsed = 0;
    const poll = setInterval(async () => {
      elapsed++;

      const res = await fetch("http://127.0.0.1:8000/probe_status");
      const data = await res.json();

      if (data.matched) {
        setResult(data);
        setStatus("Match found ✅");
        clearInterval(poll);
      }

      if (elapsed > 20) {
        setStatus("No match found ❌");
        clearInterval(poll);
      }
    }, 1000);
  }

  return (
    <div style={{ fontFamily: "sans-serif", padding: 20 }}>
      <h2>📹 Live CCTV Feed</h2>
      <img
        src="http://127.0.0.1:5001/feed"
        width="520"
        style={{ borderRadius: 10, border: "2px solid #ddd" }}
      />

      <h2 style={{ marginTop: 20 }}>🧍 Upload Person Image</h2>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <br /><br />
      <button onClick={upload}>Upload & Search</button>

      <p><b>Status:</b> {status}</p>

      {result && (
        <div style={{
          background: "#e6fffa",
          padding: 15,
          borderRadius: 10,
          border: "1px solid #2dd4bf"
        }}>
          <h3>✅ MATCH FOUND</h3>
          <p>Person ID: <b>{result.person_id}</b></p>
          <p>Similarity: <b>{result.similarity}</b></p>
        </div>
      )}
    </div>
  );
}
