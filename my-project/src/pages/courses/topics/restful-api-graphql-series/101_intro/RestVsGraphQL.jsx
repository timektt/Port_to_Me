import React from "react";

const RestVsGraphQL = () => {
  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4">REST vs GraphQL</h1>
      <p className="text-lg">
        REST and GraphQL are two popular methods for building APIs. REST uses a resource-based approach,
        while GraphQL allows clients to query exactly the data they need.
      </p>
      <p className="mt-4">
        - REST follows a structured approach using endpoints like <code>/users</code> or <code>/posts</code>.<br />
        - GraphQL allows fetching multiple resources in a single request, improving efficiency.
      </p>
    </div>
  );
};

export default RestVsGraphQL;
