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
import CourseTags from "./components/common/CourseTags"; // ✅ Import Component ใหม่
import TagsPage from "./pages/TagsPage";
import PopularTags from "./pages/courses/PopularTags";

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


// Python 204
import BasicStatistics from "./pages/courses/topics/python/204_statistic_analysis/BasicStatistics";
import ProbabilityDistribution from "./pages/courses/topics/python/204_statistic_analysis/ProbabilityDistribution";
import HypothesisTesting from "./pages/courses/topics/python/204_statistic_analysis/HypothesisTesting";


// Python 205
import RegressionAnalysis from "./pages/courses/topics/python/205_statistic_learning/RegressionAnalysis";
import ClusteringMethods from "./pages/courses/topics/python/205_statistic_learning/ClusteringMethods";
import DeepLearningBasics from "./pages/courses/topics/python/205_statistic_learning/DeepLearningBasics";


import NodeIntro from "./pages/courses/topics/nodejs/101_node_basics/NodeIntro";
import NodeJsRunCode from "./pages/courses/topics/nodejs/101_node_basics/NodeJsRunCode";
import NodeModules from "./pages/courses/topics/nodejs/101_node_basics/NodeModules";
import NodePackageManager from "./pages/courses/topics/nodejs/101_node_basics/NodePackageManager";
import NodeSetup from "./pages/courses/topics/nodejs/101_node_basics/NodeSetup";


// Node.js 201
import AsyncCallbacks from "./pages/courses/topics/nodejs/201_async_js/AsyncCallbacks";
import PromisesAsyncAwait from "./pages/courses/topics/nodejs/201_async_js/PromisesAsyncAwait";
import EventEmitter from "./pages/courses/topics/nodejs/201_async_js/EventEmitter";
import StreamsBuffer from "./pages/courses/topics/nodejs/201_async_js/StreamsBuffer";

// Node.js 202
import EventLoop from "./pages/courses/topics/nodejs/202_event_loop/EventLoop";
import TimersIO from "./pages/courses/topics/nodejs/202_event_loop/TimersIO";
import ProcessNextTick from "./pages/courses/topics/nodejs/202_event_loop/ProcessNextTick";


// Node.js 203
import RestApiBasics from "./pages/courses/topics/nodejs/203_api_development/RestApiBasics";
import HandlingHttpRequests from "./pages/courses/topics/nodejs/203_api_development/HandlingHttpRequests";
import MiddlewareConcepts from "./pages/courses/topics/nodejs/203_api_development/MiddlewareConcepts";


// Node.js 204
import ExpressIntro from "./pages/courses/topics/nodejs/204_express/ExpressIntro";
import ExpressRouting from "./pages/courses/topics/nodejs/204_express/ExpressRouting";
import ExpressMiddleware from "./pages/courses/topics/nodejs/204_express/ExpressMiddleware";
// Node.js 205
import MongoDBIntegration from "./pages/courses/topics/nodejs/205_database/MongoDBIntegration";
import PostgreSQLIntegration from "./pages/courses/topics/nodejs/205_database/PostgreSQLIntegration";
import MongooseORM from "./pages/courses/topics/nodejs/205_database/MongooseORM";
import KnexJSPostgreSQL from "./pages/courses/topics/nodejs/205_database/KnexJSPostgreSQL";




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
            {/* ✅ ใช้ CourseTags ในหน้า All Courses */}
            <Route path="/courses/all-courses" element={<AllCourses theme={theme} />} />
            <Route path="/courses/tags" element={<CourseTags />} /> {/* ✅ Route ใหม่สำหรับ Tag Page */}
            <Route path="/tags" element={<TagsPage />} /> {/* ✅ เส้นทางไปที่ TagsPage */}
            <Route path="/popular-tags" element={<PopularTags />} /> {/* ✅ เส้นทางไปที่ PopularTags */}

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

                        {/*data203*/}
                        <Route path="data-cleaning" element={<DataCleaning />} />
                        <Route path="data-transformation" element={<DataTransformation />} />
                        <Route path="data-formatting" element={<DataFormatting />} />

                        {/*statistic204*/}
                        <Route path="basic-statistics" element={<BasicStatistics />} />
                        <Route path="probability" element={<ProbabilityDistribution />} />
                        <Route path="hypothesis-testing" element={<HypothesisTesting />} />

                        {/*statistic205*/}
                        <Route path="regression" element={<RegressionAnalysis />} />
                        <Route path="clustering" element={<ClusteringMethods />} />
                        <Route path="deep-learning" element={<DeepLearningBasics />} />
            </Route>

            {/* ✅ Node.js Series */}
            <Route path="/courses/nodejs-series/*" element={<NodeSeries theme={theme} setTheme={setTheme} />}>
                <Route index element={<NodeIntro />} />
                <Route path="node-intro" element={<NodeIntro />} />
                <Route path="node-run-code" element={<NodeJsRunCode />} />
                <Route path="node-modules" element={<NodeModules />} />
                <Route path="node-package-manager" element={<NodePackageManager />} />
                <Route path="node-setup" element={<NodeSetup />} />

                {/* ✅ Asynchronous JavaScript */}
                <Route path="async-callbacks" element={<AsyncCallbacks />} />
                <Route path="promises-async-await" element={<PromisesAsyncAwait />} />
                <Route path="event-emitter" element={<EventEmitter />} />
                <Route path="streams-buffer" element={<StreamsBuffer />} />

                {/* ✅ Event Loop & Async Operations */}
                <Route path="event-loop" element={<EventLoop />} />
                <Route path="timers-io" element={<TimersIO />} />
                <Route path="process-next-tick" element={<ProcessNextTick />} />

                {/* ✅ API Development */}
                <Route path="rest-api-basics" element={<RestApiBasics />} />
                <Route path="handling-http-requests" element={<HandlingHttpRequests />} />
                <Route path="middleware-concepts" element={<MiddlewareConcepts />} />

                {/* ✅ Express.js Framework */}
                <Route path="express-intro" element={<ExpressIntro />} />
                <Route path="express-routing" element={<ExpressRouting />} />
                <Route path="express-middleware" element={<ExpressMiddleware />} />

                {/* ✅ Database Integration */}
                <Route path="mongodb-integration" element={<MongoDBIntegration />} />
                <Route path="postgresql-integration" element={<PostgreSQLIntegration />} />
                <Route path="mongoose-orm" element={<MongooseORM />} />
                <Route path="knexjs-postgresql" element={<KnexJSPostgreSQL />} />
            </Route>

            {/* ✅ คอร์สอื่น ๆ */}
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
