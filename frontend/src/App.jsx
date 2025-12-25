import { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [result, setResult] = useState(null);

  async function upload() {
    if (!file) {
      alert("Please select an image");
      return;
    }

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
    <div style={styles.page}>
      {/* HEADER */}
      <header style={styles.header}>
        <h1>📹 Live CCTV Person Tracker</h1>
        <p>Real‑time person re‑identification system</p>
      </header>

      {/* MAIN CONTENT */}
      <div style={styles.main}>
        {/* LIVE FEED SECTION */}
        <div style={styles.feedCard}>
          <h2>🔴 Live Camera Feed</h2>

          <div style={styles.feedBox}>
            <img
              src="http://127.0.0.1:5001/feed"
              alt="Live CCTV Feed"
              style={styles.feedImg}
            />
          </div>

          <p style={{ color: "#16a34a", marginTop: 10 }}>
            ● Live stream running
          </p>
        </div>

        {/* CONTROL PANEL */}
        <div style={styles.controlCard}>
          <h2>🧍 Upload Probe Image</h2>

          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            style={styles.fileInput}
          />

          <button onClick={upload} style={styles.button}>
            Upload & Search
          </button>

          <div style={styles.statusBox}>
            <b>Status:</b> {status}
          </div>

          {result && (
            <div style={styles.resultBox}>
              <h3>✅ MATCH FOUND</h3>
              <p>
                <b>Person ID:</b> {result.person_id}
              </p>
              <p>
                <b>Similarity:</b> {result.similarity}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* FOOTER */}
      <footer style={styles.footer}>
        Built with OpenCV · YOLO · DeepSORT · FastAPI · React
      </footer>
    </div>
  );
}

/* ================= STYLES ================= */

const styles = {
  page: {
    fontFamily: "Inter, system-ui, sans-serif",
    backgroundColor: "#f8fafc",
    minHeight: "100vh",
  },

  header: {
    padding: "20px",
    background: "linear-gradient(90deg, #0f172a, #020617)",
    color: "white",
    textAlign: "center",
  },

  main: {
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: "20px",
    padding: "20px",
  },

  feedCard: {
    background: "white",
    borderRadius: "12px",
    padding: "16px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
  },

  feedBox: {
    backgroundColor: "#000",
    borderRadius: "10px",
    overflow: "hidden",
    marginTop: "10px",
  },

  feedImg: {
    width: "100%",
    height: "auto",
    display: "block",
  },

  controlCard: {
    background: "white",
    borderRadius: "12px",
    padding: "16px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
  },

  fileInput: {
    marginTop: "10px",
    marginBottom: "10px",
  },

  button: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#2563eb",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
  },

  statusBox: {
    marginTop: "15px",
    padding: "10px",
    backgroundColor: "#f1f5f9",
    borderRadius: "8px",
  },

  resultBox: {
    marginTop: "15px",
    padding: "12px",
    backgroundColor: "#dcfce7",
    border: "1px solid #22c55e",
    borderRadius: "8px",
  },

  footer: {
    textAlign: "center",
    padding: "10px",
    fontSize: "14px",
    color: "#475569",
  },
};
