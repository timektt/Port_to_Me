// src/pages/Home.jsx
import Navbar from "../components/Navbar";

const Home = () => {
  return (
    <div>
      <Navbar />
      <div className="text-white p-10">
        <h1 className="text-3xl font-bold">🎉 Welcome to your Dashboard</h1>
        <p className="mt-2">This is your private area after logging in.</p>
      </div>
    </div>
  );
};

export default Home;
