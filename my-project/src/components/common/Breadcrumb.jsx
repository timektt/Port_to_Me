import { FaHome, FaAngleRight } from "react-icons/fa";
import { Link, useLocation, useNavigate } from "react-router-dom";

// ✅ Mapping หัวข้อหลักของแต่ละคอร์ส
const topicToMainCategory = {
  python: {
    intro: "101",
    variables: "101",
    "control-structure": "101",
    "input-function": "101",
    leetcode: "101",
    data: "201",
    dictionaries: "201",
    set: "201",
    pandas: "201",
    matplotlib: "202",
    seaborn: "202",
    plotly: "202",
    "data-cleaning": "203",
    "data-transformation": "203",
    "data-formatting": "203",
    "basic-statistics": "204",
    probability: "204",
    "hypothesis-testing": "204",
    regression: "205",
    clustering: "205",
    "deep-learning": "205",
  },

  nodejs: {
    "node-intro": "301",
    "node-async": "301",
    "node-rest-api": "301",
    "node-streams": "301",
    "node-security": "301",
  },

  reactjs: {
    "react-intro": "401",
    "react-components": "401",
    "react-state": "401",
    "react-hooks": "401",
    "react-context": "401",
  },

  webdev: {
    "web-html": "501",
    "web-css": "501",
    "web-javascript": "501",
    "web-responsive-design": "501",
  },

  restfulapi: {
    "rest-api": "601",
    "graphql": "601",
    "api-authentication": "601",
    "rate-limiting": "601",
    "api-documentation": "601",
    "advanced-api-security": "601",
  },

  basicprogramming: {
    "programming-intro": "701",
    "algorithms": "701",
    "data-structures": "701",
    "oop": "701",
    "functional-programming": "701",
    "competitive-programming": "701",
  },
};

// ✅ Mapping คอร์สหลักที่จะแสดงใน Breadcrumb
const mainTopics = {
  python: {
    "101": "Basic Python 101",
    "201": "Data Processing 201",
    "202": "Data Visualization 202",
    "203": "Data Wrangling & Transform 203",
    "204": "Statistical Analysis 204",
    "205": "Machine Learning Basics 205",
  },

  nodejs: {
    "301": "Node.js Fundamentals",
  },

  reactjs: {
    "401": "React.js Development",
  },

  webdev: {
    "501": "Web Development Basics",
  },

  restfulapi: {
    "601": "RESTful API & GraphQL",
  },

  basicprogramming: {
    "701": "Basic Programming",
  },
};

const Breadcrumb = ({ courseName, theme }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const pathnames = location.pathname.split("/").filter((x) => x); // ✅ แยก path ออกจาก URL

  // ✅ หาชื่อคอร์สหลักจาก URL (เช่น python, nodejs, reactjs)
  const courseKey = pathnames.length >= 2 ? pathnames[1].replace("-series", "") : null;
  const subTopic = pathnames.length >= 3 ? pathnames[2] : null;

  // ✅ หาหัวข้อหลักของหัวข้อย่อย ถ้ามี
  const mainTopicKey =
    courseKey && subTopic && topicToMainCategory[courseKey] && topicToMainCategory[courseKey][subTopic]
      ? topicToMainCategory[courseKey][subTopic]
      : null;

  const mainTopicName =
    courseKey && mainTopicKey && mainTopics[courseKey] && mainTopics[courseKey][mainTopicKey]
      ? mainTopics[courseKey][mainTopicKey]
      : null;

  return (
    <div className="w-full flex flex-wrap md:flex-nowrap items-center gap-4 py-3 px-5 rounded-lg shadow-md bg-opacity-90">
      {/* ปุ่มกลับหน้า Home */}
      <button
        className={`flex items-center gap-2 px-4 py-2 rounded-md font-semibold transition-all ${
          theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-300 text-black hover:bg-gray-400"
        }`}
        onClick={() => navigate("/")}
      >
        <FaHome size={18} />
        <span>Home</span>
      </button>

      <FaAngleRight className="text-gray-400" />

      {/* ✅ ชื่อคอร์สหลัก (เช่น Python Series) */}
      <span
        className={`px-4 py-2 rounded-md text-lg md:text-xl font-semibold border-2 border-green-500 ${
          theme === "dark" ? "bg-green-700 text-white" : "bg-green-300 text-black"
        }`}
      >
        {courseName}
      </span>

      {/* ✅ แสดงหัวข้อหลัก ถ้ามี (ไม่มีกรอบ) */}
      {mainTopicName && (
        <>
          <FaAngleRight className="text-gray-400" />
          <span className="text-lg md:text-xl font-semibold">{mainTopicName}</span>
        </>
      )}

      {/* ✅ แสดงหัวข้อย่อย (ไม่มีกรอบ) */}
      {subTopic && (
        <>
          <FaAngleRight className="text-gray-400" />
          <span className="text-lg md:text-xl font-semibold">{decodeURIComponent(subTopic.replace("-", " "))}</span>
        </>
      )}
    </div>
  );
};

export default Breadcrumb;
