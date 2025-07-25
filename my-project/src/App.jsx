import React, { useState, useEffect, lazy, Suspense } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/common/Navbar";
import SupportMeButton from "./support/SupportMeButton";
import Footer from "./components/common/Footer";
import { AuthProvider } from "./components/context/AuthContext";
import AdminRoute from "./routes/AdminRoute";
import MainMobileMenu from "./menu/MainMobileMenu";
import Profile from "./pages/Profile";
import { Analytics } from '@vercel/analytics/react';
import LatestUpdatesAll from "./home/LatestUpdatesAll";



const CourseGrid = lazy(() => import("./home/CourseGrid"));
const AllCourses = lazy(() => import("./pages/courses/AllCourses"));
const PythonSeries = lazy(() => import("./pages/courses/PythonSeries"));
const NodeSeries = lazy(() => import("./pages/courses/NodejsSeries"));
const WebDevSeries = lazy(() => import("./pages/courses/WebDevSeries"));
const BasicProgrammingSeries = lazy(() => import("./pages/courses/BasicProgrammingSeries"));
const ReactJsSeries = lazy(() => import("./pages/courses/ReactJsSeries"));
const RestfulApiGraphQLSeries = lazy(() => import("./pages/courses/RestfulApiGraphQLSeries"));
const SearchResults = lazy(() => import("./pages/SearchResults"));
const Login = lazy(() => import("./pages/Login"));
const LoginFirebase = lazy(() => import("./pages/LoginFirebase"));
const Register = lazy(() => import("./pages/Register"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const AdminDashboard = lazy(() => import("./pages/admin/AdminDashboard"));
const TagsPage = lazy(() => import("./pages/TagsPage"));
const TagResults = lazy(() => import("./pages/TagResults"));
const PopularTags = lazy(() => import("./pages/courses/PopularTags"));
const CourseTags = lazy(() => import("./components/common/CourseTags"));
const ProtectedRoute = lazy(() => import("./routes/ProtectedRoute"));
const AiSeries = lazy(() => import("./pages/courses/AiSeries"));

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



// ✅ Import React.js 101 (Introduction to React.js)
import ReactIntro from "./pages/courses/topics/reactjs/101_intro/ReactIntro";
import ReactSetup from "./pages/courses/topics/reactjs/101_intro/ReactSetup";
import ReactJSXRendering from "./pages/courses/topics/reactjs/101_intro/ReactJSXRendering";
import ReactVirtualDOM from "./pages/courses/topics/reactjs/101_intro/ReactVirtualDOM";
import ReactVsFrameworks from "./pages/courses/topics/reactjs/101_intro/ReactVsFrameworks";

// React.js 201
import FunctionalClassComponents from "./pages/courses/topics/reactjs/201_components_props/FunctionalClassComponents";
import PropsDrilling from "./pages/courses/topics/reactjs/201_components_props/PropsDrilling";
import ComponentLifecycle from "./pages/courses/topics/reactjs/201_components_props/ComponentLifecycle";
import ReusableComponents from "./pages/courses/topics/reactjs/201_components_props/ReusableComponents";
import CompositionVsInheritance from "./pages/courses/topics/reactjs/201_components_props/CompositionVsInheritance";

// React.js 202
import UseStateHook from "./pages/courses/topics/reactjs/202_state_management/UseStateHook";
import ContextAPI from "./pages/courses/topics/reactjs/202_state_management/ContextAPI";
import ReduxBasics from "./pages/courses/topics/reactjs/202_state_management/ReduxBasics";
import RecoilZustand from "./pages/courses/topics/reactjs/202_state_management/RecoilZustand";
import GlobalStateManagement from "./pages/courses/topics/reactjs/202_state_management/GlobalStateManagement";

// React.js 203
import HooksIntro from "./pages/courses/topics/reactjs/203_react_hooks/HooksIntro";
import CustomHooks from "./pages/courses/topics/reactjs/203_react_hooks/CustomHooks";
import UseEffectHook from "./pages/courses/topics/reactjs/203_react_hooks/UseEffectHook";
import UseReducerHook from "./pages/courses/topics/reactjs/203_react_hooks/UseReducerHook";
import UseRefHook from "./pages/courses/topics/reactjs/203_react_hooks/UseRefHook";

// React.js 204
import ReactRouterIntro from "./pages/courses/topics/reactjs/204_react_router/ReactRouterIntro";
import NestedRoutes from "./pages/courses/topics/reactjs/204_react_router/NestedRoutes";
import ProgrammaticNavigation from "./pages/courses/topics/reactjs/204_react_router/ProgrammaticNavigation";
import ProtectedRoutesPage from "./pages/courses/topics/reactjs/204_react_router/ProtectedRoutes";
import LazyLoading from "./pages/courses/topics/reactjs/204_react_router/LazyLoading";


// React.js 205
import FetchingDataWithFetchAPI from "./pages/courses/topics/reactjs/205_fetching_data/FetchingDataWithFetchAPI";
import UsingAxiosForHttpRequests from "./pages/courses/topics/reactjs/205_fetching_data/UsingAxiosForHttpRequests";
import HandlingLoadingAndErrors from "./pages/courses/topics/reactjs/205_fetching_data/HandlingLoadingAndErrors";
import GraphQLIntegration from "./pages/courses/topics/reactjs/205_fetching_data/GraphQLIntegration";
import CachingAndOptimizingAPICalls from "./pages/courses/topics/reactjs/205_fetching_data/CachingAndOptimizingAPICalls";


// WebDev101
import IntroductionToWebDevelopment from "./pages/courses/topics/webdev/101_web_dev_basics/IntroductionToWebDevelopment";
import FrontendVsBackend from "./pages/courses/topics/webdev/101_web_dev_basics/FrontendVsBackend";
import HowTheWebWorks from "./pages/courses/topics/webdev/101_web_dev_basics/HowTheWebWorks";
import ClientVsServer from "./pages/courses/topics/webdev/101_web_dev_basics/ClientVsServer";
import EssentialWebDevTools from "./pages/courses/topics/webdev/101_web_dev_basics/EssentialWebDevTools";

// ✅ Import Web Dev 201 Topics
import HTMLBasics from "./pages/courses/topics/webdev/201_html_css_basics/HTMLBasics";
import CSSBasics from "./pages/courses/topics/webdev/201_html_css_basics/CSSBasics";
import ResponsiveDesign from "./pages/courses/topics/webdev/201_html_css_basics/ResponsiveDesign";
import CSSGridFlexbox from "./pages/courses/topics/webdev/201_html_css_basics/CSSGridFlexbox";
import CSSPreprocessors from "./pages/courses/topics/webdev/201_html_css_basics/CSSPreprocessors";

// ✅ Import Web Dev 202 Topics
import JavaScriptBasics from "./pages/courses/topics/webdev/202_javascript_for_web/JavaScriptBasics";
import DOMManipulation from "./pages/courses/topics/webdev/202_javascript_for_web/DOMManipulation";
import ES6ModernJS from "./pages/courses/topics/webdev/202_javascript_for_web/ES6ModernJS";
import EventHandling from "./pages/courses/topics/webdev/202_javascript_for_web/EventHandling";
import AsyncJS from "./pages/courses/topics/webdev/202_javascript_for_web/AsyncJS";


// ✅ Import Web Dev 203 Topics
import ReactIntroWebdev from "./pages/courses/topics/webdev/203_frontend_frameworks/ReactIntroWebdev";
import VueIntro from "./pages/courses/topics/webdev/203_frontend_frameworks/VueIntro";
import AngularIntro from "./pages/courses/topics/webdev/203_frontend_frameworks/AngularIntro";
import StateManagement from "./pages/courses/topics/webdev/203_frontend_frameworks/StateManagement";
import SSRvsCSR from "./pages/courses/topics/webdev/203_frontend_frameworks/SSRvsCSR";

// ✅ Import Web Dev 204 Topics
import NodeExpress from "./pages/courses/topics/webdev/204_backend_development/NodeExpress";
import APIDevelopment from "./pages/courses/topics/webdev/204_backend_development/APIDevelopment";
import Authentication from "./pages/courses/topics/webdev/204_backend_development/Authentication";
import FileUpload from "./pages/courses/topics/webdev/204_backend_development/FileUpload";
import WebSockets from "./pages/courses/topics/webdev/204_backend_development/WebSockets";

// ✅ Import Web Dev 205 Topics
import MongoDBBasics from "./pages/courses/topics/webdev/205_databases_apis/MongoDBBasics";
import SQLFundamentals from "./pages/courses/topics/webdev/205_databases_apis/SQLFundamentals";
import RestGraphQL from "./pages/courses/topics/webdev/205_databases_apis/RestGraphQL";
import CachingStrategies from "./pages/courses/topics/webdev/205_databases_apis/CachingStrategies";
import DatabaseOptimization from "./pages/courses/topics/webdev/205_databases_apis/DatabaseOptimization";

// ✅ Import Basic Programming 101 Topics
import WhatIsProgramming from "./pages/courses/topics/basic-programming/101_introduction_to_programming/WhatIsProgramming";
import ComputerExecution from "./pages/courses/topics/basic-programming/101_introduction_to_programming/ComputerExecution";
import ProgrammingLanguages from "./pages/courses/topics/basic-programming/101_introduction_to_programming/ProgrammingLanguages";
import CompilersInterpreters from "./pages/courses/topics/basic-programming/101_introduction_to_programming/CompilersInterpreters";
import SetupEnvironment from "./pages/courses/topics/basic-programming/101_introduction_to_programming/SetupEnvironment";


// ✅ Variables & Data Types 201 Topics
import VariableIntro from "./pages/courses/topics/basic-programming/201_variables_datatypes/UnderstandingVariables";
import PrimitiveDataTypes from "./pages/courses/topics/basic-programming/201_variables_datatypes/PrimitiveDataTypes";
import TypeConversion from "./pages/courses/topics/basic-programming/201_variables_datatypes/TypeConversion";
import Constants from "./pages/courses/topics/basic-programming/201_variables_datatypes/Constants";
import VariableScope from "./pages/courses/topics/basic-programming/201_variables_datatypes/VariableScope";

// 🔁 Control Flow & Loops 202
import ConditionalStatements from "./pages/courses/topics/basic-programming/202_control_flow&loops/ConditionalStatements";
import LoopsForWhile from "./pages/courses/topics/basic-programming/202_control_flow&loops/LoopsForWhile";
import BreakContinueStatements from "./pages/courses/topics/basic-programming/202_control_flow&loops/BreakContinueStatements";
import NestedLoops from "./pages/courses/topics/basic-programming/202_control_flow&loops/NestedLoops";
import RecursionBasics from "./pages/courses/topics/basic-programming/202_control_flow&loops/RecursionBasics";


// ✅ Functions & Modules 203
import DefiningFunctions from "./pages/courses/topics/basic-programming/203_functions&modules/DefiningFunctions";
import WorkingWithModules from "./pages/courses/topics/basic-programming/203_functions&modules/WorkingWithModules";
import FunctionParameters from "./pages/courses/topics/basic-programming/203_functions&modules/FunctionParameters";
import ReturnValuesScope from "./pages/courses/topics/basic-programming/203_functions&modules/ReturnValuesScope";
import LambdaFunctions from "./pages/courses/topics/basic-programming/203_functions&modules/LambdaFunctions";


//  204
import ClassesObjects from "./pages/courses/topics/basic-programming/204_object_oriented_programming/ClassesObjects";
import EncapsulationInheritance from "./pages/courses/topics/basic-programming/204_object_oriented_programming/EncapsulationInheritance";
import PolymorphismMethodOverriding from "./pages/courses/topics/basic-programming/204_object_oriented_programming/PolymorphismMethodOverriding";
import AbstractionInterfaces from "./pages/courses/topics/basic-programming/204_object_oriented_programming/AbstractionInterfaces";
import OOPDesignPatterns from "./pages/courses/topics/basic-programming/204_object_oriented_programming/OOPDesignPatterns";

// 205
import CommonProgrammingErrors from "./pages/courses/topics/basic-programming/205_debugging_error_handling/CommonProgrammingErrors";
import DebuggingTools from "./pages/courses/topics/basic-programming/205_debugging_error_handling/DebuggingTools";
import ErrorTypes from "./pages/courses/topics/basic-programming/205_debugging_error_handling/ErrorTypes";
import ExceptionHandling from "./pages/courses/topics/basic-programming/205_debugging_error_handling/ExceptionHandling";
import LoggingMonitoring from "./pages/courses/topics/basic-programming/205_debugging_error_handling/LoggingMonitoring";

//Ai
import Day1_VectorMatrix from "./pages/standalone/ai100day/Day1_VectorMatrix";
import Day2_VectorOperations from "./pages/standalone/ai100day/Day2_VectorOperations";
import Day3_DotProduct from "./pages/standalone/ai100day/Day3_DotProduct";
import Day4_MatrixMultiplication from "./pages/standalone/ai100day/Day4_MatrixMultiplication";
import Day5_LinearTransform from "./pages/standalone/ai100day/Day5_LinearTransform";
import Day6_ActivationFunctions from "./pages/standalone/ai100day/Day6_ActivationFunctions";
import Day7_LossOptimization from "./pages/standalone/ai100day/Day7_LossAndOptimization";
import Day8_Backpropagation from "./pages/standalone/ai100day/Day8_Backpropagation";
import Day9_Regularization from "./pages/standalone/ai100day/Day9_Regularization";
import Day10_BiasVariance from "./pages/standalone/ai100day/Day10_BiasVariance";
import Day11_CrossValidation from "./pages/standalone/ai100day/Day11_CrossValidation";
import Day12_Overfitting from "./pages/standalone/ai100day/Day12_Overfitting";
import Day13_ModelExplainability from "./pages/standalone/ai100day/Day13_ModelExplainability";
import Day14_ModelFairness from "./pages/standalone/ai100day/Day14_ModelFairness";
import Day15_AIGovernance from "./pages/standalone/ai100day/Day15_AIGovernance";



//Neural Networks
import Day16_NeuralNetworkIntro from "./pages/standalone/ai100day/Day16-40/Day16_NeuralNetworkIntro";
import Day17_PerceptronMLP from "./pages/standalone/ai100day/Day16-40/Day17_PerceptronMLP";
import Day18_WeightInitialization from "./pages/standalone/ai100day/Day16-40/Day18_WeightInitialization";
import Day19_GradientDescentVariants from "./pages/standalone/ai100day/Day16-40/Day19_GradientDescentVariants";
import Day20_BatchLayerNormalization from "./pages/standalone/ai100day/Day16-40/Day20_BatchLayerNormalization";
import Day21_CNNIntro from "./pages/standalone/ai100day/Day16-40/Day21_CNNIntro";
import Day22_CNNArchitecture from "./pages/standalone/ai100day/Day16-40/Day22_CNNArchitecture";
import Day23_PoolingStride from "./pages/standalone/ai100day/Day16-40/Day23_PoolingStride";
import Day24_CNNVision from "./pages/standalone/ai100day/Day16-40/Day24_CNNVision";
import Day25_CNNRegularization from "./pages/standalone/ai100day/Day16-40/Day25_CNNRegularization";
import Day26_SequenceTimeSeries from "./pages/standalone/ai100day/Day16-40/Day26_SequenceTimeSeries";
import Day27_LSTM from "./pages/standalone/ai100day/Day16-40/Day27_LSTM";
import Day28_GRU from "./pages/standalone/ai100day/Day16-40/Day28_GRU";
import Day29_BiDeepRNNs from "./pages/standalone/ai100day/Day16-40/Day29_BiDeepRNNs";
import Day30_AttentionClassic from "./pages/standalone/ai100day/Day16-40/Day30_AttentionClassic";
import Day31_TransformerOverview from "./pages/standalone/ai100day/Day16-40/Day31_TransformerOverview";
import Day32_PositionalEncoding from "./pages/standalone/ai100day/Day16-40/Day32_PositionalEncoding";
import Day33_SelfAttention from "./pages/standalone/ai100day/Day16-40/Day33_SelfAttention";
import Day34_EncoderDecoder from "./pages/standalone/ai100day/Day16-40/Day34_EncoderDecoder";
import Day35_TransferLearning from "./pages/standalone/ai100day/Day16-40/Day35_TransferLearning";
import Day36_FineTuning from "./pages/standalone/ai100day/Day16-40/Day36_FineTuning";
import Day37_DataAugmentation from "./pages/standalone/ai100day/Day16-40/Day37_DataAugmentation";
import Day38_HyperparameterTuning from "./pages/standalone/ai100day/Day16-40/Day38_HyperparameterTuning";
import Day39_BestPractices from "./pages/standalone/ai100day/Day16-40/Day39_BestPractices";
import Day40_ModelDeployment from "./pages/standalone/ai100day/Day16-40/Day40_ModelDeployment";

//Deep Learning
import Day41_IntroCNN from "./pages/standalone/ai100day/Day41-60/Day41_IntroCNN";
import Day42_CNNFiltersFeatureMaps from "./pages/standalone/ai100day/Day41-60/Day42_CNNFiltersFeatureMaps";
import Day43_PoolingLayers from "./pages/standalone/ai100day/Day41-60/Day43_PoolingLayers";
import Day44_CNNImageClassification from "./pages/standalone/ai100day/Day41-60/Day44_CNNImageClassification";
import Day45_TransferLearning from "./pages/standalone/ai100day/Day41-60/Day45_TransferLearning";
import Day46_RNNIntro from "./pages/standalone/ai100day/Day41-60/Day46_RNNIntro";
import Day47_RNNVariants from "./pages/standalone/ai100day/Day41-60/Day47_RNN_LSTM_GRU";
import Day48_RNNApplications from "./pages/standalone/ai100day/Day41-60/Day48_RNNApplications";
import Day49_EncoderDecoder from "./pages/standalone/ai100day/Day41-60/Day49_EncoderDecoder";
import Day50_AttentionMechanism from "./pages/standalone/ai100day/Day41-60/Day50_AttentionMechanism";
import Day51_IntroductionTransformers from "./pages/standalone/ai100day/Day41-60/Day51_IntroductionTransformers";
import Day52_SelfAttention_PositionalEncoding from "./pages/standalone/ai100day/Day41-60/Day52_SelfAttention_PositionalEncoding";
import Day53_TransformerArchitectures from "./pages/standalone/ai100day/Day41-60/Day53_TransformerArchitectures";
import Day54_FineTuningLanguageModels from "./pages/standalone/ai100day/Day41-60/Day54_FineTuningLanguageModels";
import Day55_Seq2SeqTransformers from "./pages/standalone/ai100day/Day41-60/Day55_Seq2SeqTransformers";
import Day56_ImageCaptioning from "./pages/standalone/ai100day/Day41-60/Day56_ImageCaptioning";
import Day57_MultiModalModels from "./pages/standalone/ai100day/Day41-60/Day57_MultiModalModels";
import Day58_GenerativeModels from "./pages/standalone/ai100day/Day41-60/Day58_GenerativeModels";
import Day59_GANs from "./pages/standalone/ai100day/Day41-60/Day59_GANs";
import Day60_StableDiffusion from "./pages/standalone/ai100day/Day41-60/Day60_StableDiffusion";

//Reinforcement Learning
import Day61_IntroRL from "./pages/standalone/ai100day/Day61-80/Day61_IntroRL";


function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false); 

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <AuthProvider>
    <Router>
      <div className={`min-h-screen flex flex-col ${theme === "dark" ? "bg-gray-900 text-white" : "bg-gradient-to-r from-gray-100 to-gray-300 text-gray-900"}`}>
      <Navbar theme={theme} setTheme={setTheme} onMenuToggle={() => setMobileMenuOpen(true)} />

        
      {mobileMenuOpen && (
   <MainMobileMenu onClose={() => setMobileMenuOpen(false)} theme={theme} setTheme={setTheme} />
      )}
        <div className="flex-1">

       

        <Suspense fallback={<div>Loading...</div>}>
        <Routes >
            {/* ✅ หน้าแรก */}
            <Route path="/" element={<CourseGrid theme={theme} />} />
            <Route path="/courses" element={<AllCourses theme={theme} />} />
            {/* ✅ ใช้ CourseTags ในหน้า All Courses */}
            <Route path="/courses/all-courses" element={<AllCourses theme={theme} />} />
            <Route path="/courses/tags" element={<CourseTags />} /> {/* ✅ Route ใหม่สำหรับ Tag Page */}
            <Route path="/tags" element={<TagsPage />} /> {/* ✅ เส้นทางไปที่ TagsPage */}
            <Route path="/popular-tags" element={<PopularTags />} /> {/* ✅ เส้นทางไปที่ PopularTags */}
            <Route path="/latest-updates/all" element={<LatestUpdatesAll theme={theme} />} />

                          {/* ✅ หน้าคอร์สทั้งหมด */}
                          <Route
                path="/courses/python-series/*"
                element={
                  <ProtectedRoute>
                    <PythonSeries theme={theme} setTheme={setTheme} />
                  </ProtectedRoute>
                }
              >
                <Route index element={<PythonIntro />} />
                <Route path="intro" element={<PythonIntro />} />
                <Route path="variables" element={<PythonVariables />} />
                <Route path="control-structure" element={<PythonControlStructure />} />
                <Route path="input-function" element={<PythonInputFunction />} />
                <Route path="leetcode" element={<PythonLeetcode />} />

                {/* data201 */}
                <Route path="data" element={<ListsTuples />} />
                <Route path="dictionaries" element={<Dictionaries />} />
                <Route path="set" element={<SetsFrozenset />} />
                <Route path="pandas" element={<PandasData />} />

                {/* visualization202 */}
                <Route path="matplotlib" element={<MatplotlibBasics />} />
                <Route path="seaborn" element={<SeabornDataVisualization />} />
                <Route path="plotly" element={<PlotlyInteractiveGraphs />} />

                {/* data203 */}
                <Route path="data-cleaning" element={<DataCleaning />} />
                <Route path="data-transformation" element={<DataTransformation />} />
                <Route path="data-formatting" element={<DataFormatting />} />

                {/* statistic204 */}
                <Route path="basic-statistics" element={<BasicStatistics />} />
                <Route path="probability" element={<ProbabilityDistribution />} />
                <Route path="hypothesis-testing" element={<HypothesisTesting />} />

                {/* statistic205 */}
                <Route path="regression" element={<RegressionAnalysis />} />
                <Route path="clustering" element={<ClusteringMethods />} />
                <Route path="deep-learning" element={<DeepLearningBasics />} />
              </Route>

          {/* ✅ Node.js Series */}
          <Route
            path="/courses/nodejs-series/*"
            element={
              <ProtectedRoute>
                <NodeSeries theme={theme} setTheme={setTheme} />
              </ProtectedRoute>
            }
          >
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

        {/* ✅ GraphQL API Series (Protected) */}
        <Route
          path="/courses/restful-api-graphql-series/*"
          element={
            <ProtectedRoute>
              <RestfulApiGraphQLSeries theme={theme} setTheme={setTheme} />
            </ProtectedRoute>
          }
        >
          <Route index element={<ApiIntro />} />
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

        <Route
  path="/courses/reactjs-series/*"
  element={
    <ProtectedRoute>
      <ReactJsSeries theme={theme} setTheme={setTheme} />
    </ProtectedRoute>
  }
>
  <Route index element={<ReactIntro />} />
  <Route path="intro" element={<ReactIntro />} />
  <Route path="setup" element={<ReactSetup />} />
  <Route path="jsx-rendering" element={<ReactJSXRendering />} />
  <Route path="virtual-dom" element={<ReactVirtualDOM />} />
  <Route path="react-vs-frameworks" element={<ReactVsFrameworks />} />

  {/* ✅ React 201 */}
  <Route path="components" element={<FunctionalClassComponents />} />
  <Route path="props" element={<PropsDrilling />} />
  <Route path="lifecycle" element={<ComponentLifecycle />} />
  <Route path="reusable-components" element={<ReusableComponents />} />
  <Route path="composition-vs-inheritance" element={<CompositionVsInheritance />} />

  {/* ✅ React 202 */}
  <Route path="state" element={<UseStateHook />} />
  <Route path="context-api" element={<ContextAPI />} />
  <Route path="redux" element={<ReduxBasics />} />
  <Route path="recoil-zustand" element={<RecoilZustand />} />
  <Route path="global-state" element={<GlobalStateManagement />} />

  {/* ✅ React Hooks */}
  <Route path="hooks-intro" element={<HooksIntro />} />
  <Route path="useeffect" element={<UseEffectHook />} />
  <Route path="useref" element={<UseRefHook />} />
  <Route path="usereducer" element={<UseReducerHook />} />
  <Route path="custom-hooks" element={<CustomHooks />} />

  <Route path="react-router" element={<ReactRouterIntro />} />
  <Route path="nested-routes" element={<NestedRoutes />} />
  <Route path="navigation" element={<ProgrammaticNavigation />} />
  <Route path="protected-routes" element={<ProtectedRoutesPage />} />
  <Route path="lazy-loading" element={<LazyLoading />} />

  {/* ✅ Fetching Data & API Integration */}
  <Route path="fetch-api" element={<FetchingDataWithFetchAPI />} />
  <Route path="axios" element={<UsingAxiosForHttpRequests />} />
  <Route path="loading-errors" element={<HandlingLoadingAndErrors />} />
  <Route path="graphql" element={<GraphQLIntegration />} />
  <Route path="caching-api" element={<CachingAndOptimizingAPICalls />} />
</Route>

<Route
  path="/courses/web-development/*"
  element={
    <ProtectedRoute>
      <WebDevSeries theme={theme} setTheme={setTheme} />
    </ProtectedRoute>
  }
>
  <Route index element={<IntroductionToWebDevelopment />} />
  <Route path="intro" element={<IntroductionToWebDevelopment />} />
  <Route path="frontend-backend" element={<FrontendVsBackend />} />
  <Route path="how-web-works" element={<HowTheWebWorks />} />
  <Route path="client-server" element={<ClientVsServer />} />
  <Route path="web-dev-tools" element={<EssentialWebDevTools />} />

  {/* ✅ Web Dev 201 */}
  <Route path="html-basics" element={<HTMLBasics />} />
  <Route path="css-basics" element={<CSSBasics />} />
  <Route path="responsive-design" element={<ResponsiveDesign />} />
  <Route path="css-grid-flexbox" element={<CSSGridFlexbox />} />
  <Route path="css-preprocessors" element={<CSSPreprocessors />} />

  {/* ✅ Web Dev 202 */}
  <Route path="javascript-basics" element={<JavaScriptBasics />} />
  <Route path="dom-manipulation" element={<DOMManipulation />} />
  <Route path="es6-modern-js" element={<ES6ModernJS />} />
  <Route path="event-handling" element={<EventHandling />} />
  <Route path="async-js" element={<AsyncJS />} />

  {/* ✅ Web Dev 203 */}
  <Route path="react-intro" element={<ReactIntroWebdev />} />
  <Route path="vue-intro" element={<VueIntro />} />
  <Route path="angular-intro" element={<AngularIntro />} />
  <Route path="state-management" element={<StateManagement />} />
  <Route path="ssr-vs-csr" element={<SSRvsCSR />} />

  {/* ✅ Web Dev 204 */}
  <Route path="node-express" element={<NodeExpress />} />
  <Route path="api-development" element={<APIDevelopment />} />
  <Route path="authentication" element={<Authentication />} />
  <Route path="file-upload" element={<FileUpload />} />
  <Route path="websockets" element={<WebSockets />} />

  {/* ✅ Web Dev 205 */}
  <Route path="mongodb" element={<MongoDBBasics />} />
  <Route path="sql-basics" element={<SQLFundamentals />} />
  <Route path="rest-graphql" element={<RestGraphQL />} />
  <Route path="caching-strategies" element={<CachingStrategies />} />
  <Route path="db-optimization" element={<DatabaseOptimization />} />
</Route>

<Route
  path="/courses/basic-programming/*"
  element={
    <ProtectedRoute>
      <BasicProgrammingSeries theme={theme} setTheme={setTheme} />
    </ProtectedRoute>
  }
>
  <Route path="intro" element={<WhatIsProgramming />} />
  <Route path="computer-execution" element={<ComputerExecution />} />
  <Route path="programming-languages" element={<ProgrammingLanguages />} />
  <Route path="compilers-interpreters" element={<CompilersInterpreters />} />
  <Route path="setup" element={<SetupEnvironment />} />

  {/* ✅ Basic Programming: Variables & Data Types */}
  <Route path="variables" element={<VariableIntro />} />
  <Route path="data-types" element={<PrimitiveDataTypes />} />
  <Route path="type-conversion" element={<TypeConversion />} />
  <Route path="constants" element={<Constants />} />
  <Route path="scope" element={<VariableScope />} />

  {/* ✅ 202: Control Flow & Loops */}
  <Route path="conditions" element={<ConditionalStatements />} />
  <Route path="loops" element={<LoopsForWhile />} />
  <Route path="break-continue" element={<BreakContinueStatements />} />
  <Route path="nested-loops" element={<NestedLoops />} />
  <Route path="recursion" element={<RecursionBasics />} />

  {/* ✅ 203: Arrays & Objects */}
  <Route path="functions" element={<DefiningFunctions />} />
  <Route path="modules" element={<WorkingWithModules />} />
  <Route path="parameters" element={<FunctionParameters />} />
  <Route path="return-values" element={<ReturnValuesScope />} />
  <Route path="lambda-functions" element={<LambdaFunctions />} />

  {/* ✅ OOP Series */}
  <Route path="oop" element={<ClassesObjects />} />
  <Route path="oop-inheritance" element={<EncapsulationInheritance />} />
  <Route path="polymorphism" element={<PolymorphismMethodOverriding />} />
  <Route path="abstraction" element={<AbstractionInterfaces />} />
  <Route path="oop-design-patterns" element={<OOPDesignPatterns />} />

  {/* ✅ Debugging & Error Handling */}
  <Route path="debugging" element={<CommonProgrammingErrors />} />
  <Route path="debugging-tools" element={<DebuggingTools />} />
  <Route path="error-types" element={<ErrorTypes />} />
  <Route path="exception-handling" element={<ExceptionHandling />} />
  <Route path="logging-monitoring" element={<LoggingMonitoring />} />
</Route>

                    

            {/* ✅ คอร์สอื่น ๆ */}
            <Route path="/courses/reactjs-series" element={<ReactJsSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/courses/web-development" element={<WebDevSeries theme={theme} setTheme={setTheme} />} />
            <Route path="/search" element={<SearchResults />} />
            <Route path="/tags/:tagName" element={<TagResults />} />
            <Route path="/tags" element={<TagsPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/login-firebase" element={<LoginFirebase />} />
            <Route path="/register" element={<Register />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
              <Route
          path="/admin"
          element={
            <AdminRoute>
              <AdminDashboard />
            </AdminRoute>
          }
        />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />

               
                <Route path="/courses/ai-series" element={<AiSeries theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/intro-to-vectors-matrices" element={<Day1_VectorMatrix theme={theme} setTheme={setTheme} />} />
                <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
                <Route path="/courses/ai/vector-addition&scalarmultiplication"element={<Day2_VectorOperations theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/dot-product&cosinesimilarity" element={<Day3_DotProduct theme={theme} setTheme={setTheme} />}/><Route  path="/courses/ai/matrix-multiplication"element={<Day4_MatrixMultiplication theme={theme} setTheme={setTheme} />}/>          <Route path="/courses/ai/linear-transformation&feature-extraction" element={<Day5_LinearTransform theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/activation-functions" element={<Day6_ActivationFunctions theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/lossfunctions&optimization" element={<Day7_LossOptimization theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/backpropagation&trainingLoop" element={<Day8_Backpropagation theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/regularization&generalization" element={<Day9_Regularization theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/bias-variancetradeoff&modelcapacity" element={<Day10_BiasVariance theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/cross-validation&modelevaluation" element={<Day11_CrossValidation theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/overfitting-underfitting&model-diagnostics" element={<Day12_Overfitting theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/modele-interpretability&explainability" element={<Day13_ModelExplainability theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/fairness,bias&ethics" element=
                {<Day14_ModelFairness theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/ai-governance&risk-management" element=
                {<Day15_AIGovernance theme={theme} setTheme={setTheme} />} />

                // Neural Networks
                <Route path="/courses/ai/neural-network-intro" element=
                {<Day16_NeuralNetworkIntro theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/perceptron-mlp" element=
                {<Day17_PerceptronMLP theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/weight-initialization" element=
                {<Day18_WeightInitialization theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/gradient-descent-variants" element=
                {<Day19_GradientDescentVariants theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/batch-layer-normalization" element=
                {<Day20_BatchLayerNormalization theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/intro-to-cnn" element=
                {<Day21_CNNIntro theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/cnn-architecture" element=
                {<Day22_CNNArchitecture theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/pooling-stride" element=
                {<Day23_PoolingStride theme={theme} setTheme={setTheme} />} />
                 <Route path="/courses/ai/cnn-computer-vision" element=
                {<Day24_CNNVision theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/intro-to-rnn" element=
                {<Day25_CNNRegularization theme={theme} setTheme={setTheme} />} />
                 <Route path="/courses/ai/sequence-modeling" element=
                {<Day26_SequenceTimeSeries theme={theme} setTheme={setTheme} />} />
                 <Route path="/courses/ai/lstm-explained" element=
                {<Day27_LSTM theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/gru-explained" element=
                {<Day28_GRU theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/bidirectional-rnn" element=
                {<Day29_BiDeepRNNs theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/attention-mechanism" element=
                {<Day30_AttentionClassic theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/transformer-overview" element=
                {<Day31_TransformerOverview theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/positional-encoding" element=
                {<Day32_PositionalEncoding theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/self-attention" element=
                {<Day33_SelfAttention theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/transformer-encoder-decoder" element=
                {<Day34_EncoderDecoder theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/transfer-learning" element=
                {<Day35_TransferLearning theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/fine-tuning" element=
                {<Day36_FineTuning theme={theme} setTheme={setTheme} />} />
                <Route path="courses/ai/data-augmentation" element=
                {<Day37_DataAugmentation theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/hyperparameter-tuning" element=
                {<Day38_HyperparameterTuning theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/supervised-learning-best-practices" element=
                {<Day39_BestPractices theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/model-deployment-basics" element=
                {<Day40_ModelDeployment theme={theme} setTheme={setTheme} />} />
                 <Route path="/courses/ai/deep-cnn-intro" element=
                {<Day41_IntroCNN theme={theme} setTheme={setTheme} />} />
                 <Route path="/courses/ai/cnn-featuremaps" element=
                {<Day42_CNNFiltersFeatureMaps theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/cnn-pooling" element=
                {<Day43_PoolingLayers theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/cnn-image-classification" element=
                {<Day44_CNNImageClassification theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/cnn-transfer-learning" element=
                {<Day45_TransferLearning theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/rnn-intro-sequence" element=
                {<Day46_RNNIntro theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/rnn-lstm-gru" element=
                {<Day47_RNNVariants theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/rnn-usecases" element=
                {<Day48_RNNApplications theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/encoder-decoder-architecture" element=
                {<Day49_EncoderDecoder theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/attention-in-deep-learning" element=
                {<Day50_AttentionMechanism theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/transformers-intro" element=
                {<Day51_IntroductionTransformers theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/self-attention-positional" element=
                {<Day52_SelfAttention_PositionalEncoding theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/transformer-bert-gpt" element=
                {<Day53_TransformerArchitectures theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/fine-tune-language-models" element=
                {<Day54_FineTuningLanguageModels theme={theme} setTheme={setTheme} />} />
                 <Route path="/courses/ai/seq2seq-transformers" element=
                {<Day55_Seq2SeqTransformers theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/vision-language-models" element=
                {<Day56_ImageCaptioning theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/multimodal-clip-flamingo" element=
                {<Day57_MultiModalModels theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/generative-overview" element=
                {<Day58_GenerativeModels theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/gans" element=
                {<Day59_GANs theme={theme} setTheme={setTheme} />} />
                <Route path="/courses/ai/stable-diffusion-text2img" element=
                {<Day60_StableDiffusion  theme={theme} setTheme={setTheme} />} />
                // Reinforcement Learning
                <Route path="/courses/ai/rl-intro" element=
                {<Day61_IntroRL theme={theme} setTheme={setTheme} />} />
                
      
        </Routes>

        
        
        </Suspense>
        
        </div>
        
        <Footer />
       
        <div className="fixed bottom-16 right-4 z-50">
          <SupportMeButton />
        </div>
       
      </div>
      <Analytics /> 
    </Router>
    </AuthProvider>
  );
}

export default App;
