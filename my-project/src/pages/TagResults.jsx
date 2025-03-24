import React from "react";
import { useParams, Link } from "react-router-dom";
import { keywords } from "../data/keywords";

const TagResults = () => {
  const { tagName } = useParams();
  const decodedTag = decodeURIComponent(tagName);

  const filteredItems = keywords.filter((item) =>
    item.tags.some(
      (tag) => tag.toLowerCase().replace(/\s+/g, "-") === decodedTag.toLowerCase()
    )
  );

  return (
    <div className="w-full mt-9  min-h-screen px-4 py-8 sm:px-6 lg:px-8 max-w-6xl mx-auto overflow-x-hidden">
      <h2 className="text-3xl font-bold mb-6 text-center break-words">
         Results for Tag: <span className="italic">{decodedTag}</span>
      </h2>

      {filteredItems.length === 0 ? (
        <div className="border p-4 rounded-lg text-center">
          No content found for this tag.
        </div>
      ) : (
        <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredItems.map((item) => (
            <li key={item.id}>
              <Link
                to={item.path}
                className="block p-4 h-full rounded-xl shadow hover:shadow-md transition border break-words"
              >
                <h3 className="text-lg font-semibold mb-2 truncate">{item.title}</h3>
                <p className="text-sm opacity-80 break-words">{item.tags.join(", ")}</p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default TagResults;
