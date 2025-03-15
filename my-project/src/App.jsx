import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/common/Navbar";
import CourseGrid from "./home/CourseGrid";
import PythonSeries from "./pages/courses/PythonSeries";
import NodeSeries from "./pages/courses/NodejsSeries";
import RestfulApiGraphQLSeries from "./pages/courses/RestfulApiGraphQLSeries";
import ReactJsSeries from "./pages/courses/ReactJsSeries";
import WebDevSeries from "./pages/courses/WebDevSeries";
import BasicProgrammingSeries from "./pages/courses/BasicProgrammingSeries";
import AllCourses from "./pages/courses/AllCourses";
import SupportMeButton from "./support/SupportMeButton";
import Footer from "./components/common/Footer";
import BasicProgrammingMobileMenu from "./components/common/sidebar/MobileMenus/BasicProgrammingMobileMenu";

// ✅ Import Python Subtopics (อัปเดตเส้นทางหลังแยกโฟลเดอร์)
// Python 101
import PythonIntro from "./pages/courses/topics/python/101_basic_python/PythonIntro";
import PythonVariables from "./pages/courses/topics/python/101_basic_python/PythonVariables";
import PythonControlStructure from "./pages/courses/topics/python/101_basic_python/PythonControlStructure";
import PythonInputFunction from "./pages/courses/topics/python/101_basic_python/PythonInputFunction";
import PythonLeetcode from "./pages/courses/topics/python/101_basic_python/PythonLeetcode";
// Python 201
import ListsTuples from "./pages/courses/topics/python/201_data/ListsTuples";
import Dictionaries from "./pages/courses/topics/python/201_data/Dictionaries";
import SetsFrozenset from "./pages/courses/topics/python/201_data/SetFrozenset";
import PandasData from "./pages/courses/topics/python/201_data/PandasData";

// Python 202
import MatplotlibBasics from "./pages/courses/topics/python/202_visualization/MatplotlibBasics";
import SeabornDataVisualization from "./pages/courses/topics/python/202_visualization/SeabornDataVisualization";
import PlotlyInteractiveGraphs from "./pages/courses/topics/python/202_visualization/PlotlyInteractiveGraphs";

// Python 203
import DataCleaning from "./pages/courses/topics/python/203_data_wrangling_transform/DataCleaning";
import DataTransformation from "./pages/courses/topics/python/203_data_wrangling_transform/DataTransformation";
import DataFormatting from "./pages/courses/topics/python/203_data_wrangling_transform/DataFormatting";


import { Outlet } from "react-router-dom"; // ✅ ใช้ Outlet เพื่อให้ PythonSeries เป็น Layout หลัก

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false); 

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <Router>
      <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-gray-900"}`}>
        <Navbar theme={theme} setTheme={setTheme} onMenuToggle={() => setMobileMenuOpen(true)} />

        {mobileMenuOpen && (
          <BasicProgrammingMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
        )}

        <div className="flex-1">
          <Routes>
            {/* ✅ หน้าแรก */}
            <Route path="/" element={<CourseGrid theme={theme} />} />
            <Route path="/courses" element={<AllCourses theme={theme} />} />

            {/* ✅ หน้าคอร์สทั้งหมด */}
            <Route path="/courses/python-series/*" element={<PythonSeries theme={theme} setTheme={setTheme} />}>
                        <Route index element={<PythonIntro />} /> {/* ✅ หน้า default เมื่อเข้า /courses/python-series */}
                        <Route path="intro" element={<PythonIntro />} />
                        <Route path="variables" element={<PythonVariables />} />
                        <Route path="control-structure" element={<PythonControlStructure />} />
                        <Route path="input-function" element={<PythonInputFunction />} />
                        <Route path="leetcode" element={<PythonLeetcode />} />


                        {/*data201*/}
                        <Route path="data" element={<ListsTuples />} />
                        <Route path="dictionaries" element={<Dictionaries />} />
                        <Route path="set" element={<SetsFrozenset />} />
                        <Route path="pandas" element={<PandasData />} />


                        {/*visualization202*/}
                        <Route path="matplotlib" element={<MatplotlibBasics />} />
                        <Route path="seaborn" element={<SeabornDataVisualization />} />
                        <Route path="plotly" element={<PlotlyInteractiveGraphs />} />


                        <Route path="data-cleaning" element={<DataCleaning />} />
                        <Route path="data-transformation" element={<DataTransformation />} />
                        <Route path="data-formatting" element={<DataFormatting />} />

                </Route>
 

            {/* ✅ คอร์สอื่น ๆ */}
            <Route path="/courses/nodejs-series" element={<NodeSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/reactjs-series" element={<ReactJsSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/web-development" element={<WebDevSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/basic-programming" element={<BasicProgrammingSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/restful-api-graphql-series" element={<RestfulApiGraphQLSeries theme={theme} setTheme={setTheme} />} />
          </Routes>
        </div>

        <Footer />
        <div className="fixed bottom-16 right-4 z-50">
          <SupportMeButton />
        </div>
      </div>
    </Router>
  );
}

export default App;
