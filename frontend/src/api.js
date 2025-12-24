const BASE = "http://127.0.0.1:8000";

export async function searchPerson(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE}/upload_probe`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

export async function getProbeStatus() {
  const res = await fetch(`${BASE}/probe_status`);
  return res.json();
}
