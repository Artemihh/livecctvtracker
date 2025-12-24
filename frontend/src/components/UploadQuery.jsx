import { useState } from "react";

export default function UploadQuery({ onUploaded }) {
  const [preview, setPreview] = useState(null);

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    onUploaded(file);
  }

  return (
    <div className="bg-white p-6 rounded-xl shadow">
      <h2 className="text-xl font-semibold mb-4">
        Upload Query Image
      </h2>

      <input type="file" accept="image/*" onChange={handleFile} />

      {preview && (
        <img
          src={preview}
          className="mt-4 w-48 h-48 rounded-xl object-cover"
        />
      )}
    </div>
  );
}
