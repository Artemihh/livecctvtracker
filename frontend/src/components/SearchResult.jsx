export default function SearchResult({ result }) {
  if (!result) return null;

  return (
    <div className="mt-6 bg-green-100 p-4 rounded-xl">
      <h3 className="font-bold">MATCH FOUND</h3>
      <p>Person ID: {result.person_id}</p>
      <p>Similarity: {result.similarity}</p>
    </div>
  );
}
