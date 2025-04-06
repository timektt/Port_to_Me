import { useLocation, Link } from "react-router-dom";
import { useMemo } from "react";
import { keywords } from "../data/keywords";
import Fuse from "fuse.js";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

const SearchResults = () => {
  const rawQuery = useQuery().get("q") || "";
  const query = rawQuery.trim().toLowerCase();

  const fuse = new Fuse(keywords, {
    keys: ["title", "tags"],
    threshold: 0.3,
    includeScore: false,
  });

  const filtered = useMemo(() => {
    if (!query || query.length > 100) return []; // ✅ ป้องกัน input ยาวเกินหรือลองใส่โค้ดแปลกๆ
    return fuse.search(query).map((result) => result.item);
  }, [query]);

  return (
    <div className="w-full min-h-screen px-4 py-8 sm:px-6 lg:px-8 max-w-6xl mx-auto overflow-x-hidden">
      <h2 className="text-3xl font-bold mb-6 text-center break-words">
        🔍 Results for: <span className="italic">{query}</span>
      </h2>

      {filtered.length === 0 ? (
        <div className="border p-4 rounded-lg text-center">
          No results found. Please try different keywords.
        </div>
      ) : (
        <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {filtered.map((item) => (
            <li key={item.id}>
              <Link
                to={item.path}
                className="block p-4 h-full rounded-xl shadow hover:shadow-md transition border break-words"
              >
                <h3 className="text-lg font-semibold mb-2 truncate">{item.title}</h3>
                <p className="text-sm opacity-80 break-words">
                  {item.tags.join(", ")}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SearchResults;
