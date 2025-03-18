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
import ErrorHandling from "./pages/courses/topics/nodejs/203_api_development/ErrorHandling"; 


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
import FsPromises from "./pages/courses/topics/nodejs/201_async_js/FsPromises";

// Node.js 202
import EventLoop from "./pages/courses/topics/nodejs/202_event_loop/EventLoop";
import TimersIO from "./pages/courses/topics/nodejs/202_event_loop/TimersIO";
import ProcessNextTick from "./pages/courses/topics/nodejs/202_event_loop/ProcessNextTick";
import AsyncErrors from "./pages/courses/topics/nodejs/202_event_loop/AsyncErrors";
import ChildProcesses from "./pages/courses/topics/nodejs/202_event_loop/ChildProcesses";

// Node.js 203
import RestApiBasics from "./pages/courses/topics/nodejs/203_api_development/RestApiBasics";
import HandlingHttpRequests from "./pages/courses/topics/nodejs/203_api_development/HandlingHttpRequests";
import MiddlewareConcepts from "./pages/courses/topics/nodejs/203_api_development/MiddlewareConcepts";
import ValidationErrorHandling from "./pages/courses/topics/nodejs/203_api_development/AuthenticationJWT";
import AuthenticationJWT from "./pages/courses/topics/nodejs/203_api_development/AuthenticationJWT";


// Node.js 204
import ExpressIntro from "./pages/courses/topics/nodejs/204_express/ExpressIntro";
import ExpressRouting from "./pages/courses/topics/nodejs/204_express/ExpressRouting";
import ExpressMiddleware from "./pages/courses/topics/nodejs/204_express/ExpressMiddleware";
import ExpressCORS from "./pages/courses/topics/nodejs/204_express/ExpressCORS";
import ExpressErrorHandling from "./pages/courses/topics/nodejs/204_express/ExpressErrorHandling";

// Node.js 205
import MongoDBIntegration from "./pages/courses/topics/nodejs/205_database/MongoDBIntegration";
import PostgreSQLIntegration from "./pages/courses/topics/nodejs/205_database/PostgreSQLIntegration";
import MongooseORM from "./pages/courses/topics/nodejs/205_database/MongooseORM";
import KnexJSPostgreSQL from "./pages/courses/topics/nodejs/205_database/KnexJSPostgreSQL";
import RedisIntegration from "./pages/courses/topics/nodejs/205_database/RedisIntegration";

// restful-api-graphql-series 101
import ApiIntro from "./pages/courses/topics/restful-api-graphql-series/101_intro/ApiIntro";
import RestVsGraphQL from "./pages/courses/topics/restful-api-graphql-series/101_intro/RestVsGraphQL";
import HowApisWork from "./pages/courses/topics/restful-api-graphql-series/101_intro/HowApisWork";
import ApiTypes from "./pages/courses/topics/restful-api-graphql-series/101_intro/ApiTypes";
import ApiDocumentation from "./pages/courses/topics/restful-api-graphql-series/101_intro/ApiDocumentation";


// ✅ Import RESTful API 201
import RestBasics from "./pages/courses/topics/restful-api-graphql-series/201_restful_api/RestBasics";
import RestNodejs from "./pages/courses/topics/restful-api-graphql-series/201_restful_api/RestNodejs";
import RestCrud from "./pages/courses/topics/restful-api-graphql-series/201_restful_api/RestCrud";
import RestErrorHandling from "./pages/courses/topics/restful-api-graphql-series/201_restful_api/RestErrorHandling";
import RestVersioning from "./pages/courses/topics/restful-api-graphql-series/201_restful_api/RestVersioning";


// ✅ Import GraphQL 202
import GraphQLBasics from "./pages/courses/topics/restful-api-graphql-series/202_graphql/GraphQLBasics";
import BuildingGraphQLApi from "./pages/courses/topics/restful-api-graphql-series/202_graphql/BuildingGraphQLApi";
import QueriesMutations from "./pages/courses/topics/restful-api-graphql-series/202_graphql/QueriesMutations";
import GraphQLSchemaResolvers from "./pages/courses/topics/restful-api-graphql-series/202_graphql/GraphQLSchemaResolvers";
import GraphQLVsRest from "./pages/courses/topics/restful-api-graphql-series/202_graphql/GraphQLVsRest";

// ✅ Import API Security 203
import ApiAuthentication from "./pages/courses/topics/restful-api-graphql-series/203_api_security/ApiAuthentication";
import RateLimitingCORS from "./pages/courses/topics/restful-api-graphql-series/203_api_security/RateLimitingCORS";
import OAuthApiKeys from "./pages/courses/topics/restful-api-graphql-series/203_api_security/OAuthApiKeys";
import JwtSessionManagement from "./pages/courses/topics/restful-api-graphql-series/203_api_security/JwtSessionManagement";
import ApiSecurityBestPractices from "./pages/courses/topics/restful-api-graphql-series/203_api_security/ApiSecurityBestPractices";

// ✅ Import Advanced API Concepts 204
import ApiGatewaysMicroservices from "./pages/courses/topics/restful-api-graphql-series/204_advanced_api/ApiGatewaysMicroservices";
import GraphQLSubscriptions from "./pages/courses/topics/restful-api-graphql-series/204_advanced_api/GraphQLSubscriptions";
import ApiPerformanceOptimization from "./pages/courses/topics/restful-api-graphql-series/204_advanced_api/ApiPerformanceOptimization";
import ApiTestingMonitoring from "./pages/courses/topics/restful-api-graphql-series/204_advanced_api/ApiTestingMonitoring";
import ApiDeploymentScaling from "./pages/courses/topics/restful-api-graphql-series/204_advanced_api/ApiDeploymentScaling";

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
                <Route path="node-npm-yarn" element={<NodePackageManager />} />
                <Route path="node-setup" element={<NodeSetup />} />

                {/* ✅ Asynchronous JavaScript */}
                <Route path="async-callbacks" element={<AsyncCallbacks />} />
                <Route path="promises-async-await" element={<PromisesAsyncAwait />} />
                <Route path="event-emitter" element={<EventEmitter />} />
                <Route path="streams-buffer" element={<StreamsBuffer />} />
                <Route path="fs-promises" element={<FsPromises />} />

                {/* ✅ Event Loop & Async Operations */}
                <Route path="event-loop" element={<EventLoop />} />
                <Route path="timers-io" element={<TimersIO />} />
                <Route path="process-next-tick" element={<ProcessNextTick />} />
                <Route path="async-error-handling" element={<AsyncErrors />} />
                <Route path="child-processes" element={<ChildProcesses />} />

                {/* ✅ API Development */}
                <Route path="rest-api-basics" element={<RestApiBasics />} />
                <Route path="handling-http-requests" element={<HandlingHttpRequests />} />
                <Route path="middleware-concepts" element={<MiddlewareConcepts />} />
                <Route path="error-handling" element={<ValidationErrorHandling />} />
                <Route path="api-authentication" element={<AuthenticationJWT />} />

        

                {/* ✅ Express.js Framework */}
                <Route path="express-intro" element={<ExpressIntro />} />
                <Route path="express-routing" element={<ExpressRouting />} />
                <Route path="express-middleware" element={<ExpressMiddleware />} />
                <Route path="express-cors" element={<ExpressCORS />} />
                <Route path="express-error-handling" element={<ExpressErrorHandling />} />

                {/* ✅ Database Integration */}
                <Route path="mongodb-integration" element={<MongoDBIntegration />} />
                <Route path="postgresql-integration" element={<PostgreSQLIntegration />} />
                <Route path="mongoose-orm" element={<MongooseORM />} />
                <Route path="knexjs-postgresql" element={<KnexJSPostgreSQL />} />
                <Route path="redis-integration" element={<RedisIntegration />} />


                </Route>

                {/* ✅ GraphQL API */}
                {/* ✅ GraphQL API */}
                  <Route path="/courses/restful-api-graphql-series/*" element={<RestfulApiGraphQLSeries theme={theme} setTheme={setTheme} />}>
                    <Route index element={<ApiIntro />} />  {/* ✅ หน้าแรกเมื่อเข้า /courses/restful-api-graphql-series */}
                    <Route path="intro" element={<ApiIntro />} />
                    <Route path="rest-vs-graphql" element={<RestVsGraphQL />} />
                    <Route path="how-apis-work" element={<HowApisWork />} />
                    <Route path="api-types" element={<ApiTypes />} />
                    <Route path="api-documentation" element={<ApiDocumentation />} />


                    <Route path="rest-basics" element={<RestBasics />} />
                    <Route path="rest-nodejs" element={<RestNodejs />} />
                    <Route path="rest-crud" element={<RestCrud />} />
                    <Route path="rest-error-handling" element={<RestErrorHandling />} />
                    <Route path="rest-versioning" element={<RestVersioning />} />

                    <Route path="graphql-basics" element={<GraphQLBasics />} />
                    <Route path="graphql-api" element={<BuildingGraphQLApi />} />
                    <Route path="graphql-queries-mutations" element={<QueriesMutations />} />
                    <Route path="graphql-schema-resolvers" element={<GraphQLSchemaResolvers />} />
                    <Route path="graphql-vs-rest" element={<GraphQLVsRest />} />

                    {/* ✅ API Security 203 */}
                    <Route path="api-security" element={<ApiAuthentication />} />
                    <Route path="rate-limiting" element={<RateLimitingCORS />} />
                    <Route path="oauth-api-keys" element={<OAuthApiKeys />} />
                    <Route path="jwt-session" element={<JwtSessionManagement />} />
                    <Route path="api-security-best-practices" element={<ApiSecurityBestPractices />} />

                    {/* ✅ Advanced API Concepts 204 */}
                    <Route path="api-gateways-microservices" element={<ApiGatewaysMicroservices />} />
                    <Route path="graphql-subscriptions" element={<GraphQLSubscriptions />} />
                    <Route path="api-performance" element={<ApiPerformanceOptimization />} />
                    <Route path="api-testing-monitoring" element={<ApiTestingMonitoring />} />
                    <Route path="api-deployment-scaling" element={<ApiDeploymentScaling />} />
                  </Route>

            {/* ✅ คอร์สอื่น ๆ */}
            <Route path="/courses/reactjs-series" element={<ReactJsSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/web-development" element={<WebDevSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/basic-programming" element={<BasicProgrammingSeries theme={theme} setTheme={setTheme} />} />
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
