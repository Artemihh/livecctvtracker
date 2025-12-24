export default function CameraFeeds({ targetCamera }) {
  const cameras = [
    { id: "CAM1", url: "http://127.0.0.1:5001/feed" },  // Your unified stream server
  ];

  return (
    <div className="mt-6">
      <h2 className="text-xl font-semibold mb-4">Camera Feeds</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {cameras.map((cam) => (
          <div
            key={cam.id}
            className={`rounded-xl overflow-hidden border shadow bg-black ${
              targetCamera === cam.id
                ? "ring-4 ring-red-500 scale-[1.02] transition"
                : ""
            }`}
          >
            {/* Camera Label */}
            <div className="p-2 bg-gray-900 text-white text-center">
              {cam.id}
            </div>

            {/* Live Stream */}
            <img
              src={cam.url}
              alt={`${cam.id} feed`}
              className="w-full h-72 object-cover"
            />
          </div>
        ))}
      </div>
    </div>
  );
}


