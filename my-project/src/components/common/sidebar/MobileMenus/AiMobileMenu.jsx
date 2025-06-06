import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  FaTimes,
  FaChevronDown,
  FaChevronRight,
  FaArrowLeft,
  FaMoon,
} from "react-icons/fa";
import { FiSun } from "react-icons/fi";

const sidebarItems = [
  {
    id: "101",
    title: "101: AI Core Concepts",
    subItems: [
      { id: "day1", title: "Day 1: Vectors & Matrices", path: "/courses/ai/intro-to-vectors-matrices" },
      { id: "day2", title: "Day 2: Vector Addition & Scalar Multiplication", path: "/courses/ai/vector-addition&scalarmultiplication" },
      { id: "day3", title: "Day 3: Dot Product & Cosine Similarity", path: "/courses/ai/dot-product&cosinesimilarity" },
      { id: "day4", title: "Day 4: Matrix Multiplication", path: "/courses/ai/matrix-multiplication" },
      { id: "day5", title: "Day 5: Linear Transformation & Feature Extraction", path: "/courses/ai/linear-transformation&feature-extraction" },
      { id: "day6", title: "Day 6: Activation Functions", path: "/courses/ai/activation-functions" },
      { id: "day7", title: "Day 7: Loss Functions & Optimization", path: "/courses/ai/lossfunctions&optimization" },
      { id: "day8", title: "Day 8: Backpropagation & Training Loop", path: "/courses/ai/backpropagation&trainingLoop" },
      { id: "day9", title: "Day 9: Regularization & Generalization", path: "/courses/ai/regularization&generalization" },
      { id: "day10", title: "Day 10: Bias-Variance Tradeoff & Model Capacity", path: "/courses/ai/bias-variancetradeoff&modelcapacity" },
      { id: "day11", title: "Day 11: Cross Validation & Model Evaluation", path: "/courses/ai/cross-validation&modelevaluation" },
      { id: "day12", title: "Day 12: Overfitting, Underfitting & Model Diagnostics", path: "/courses/ai/overfitting-underfitting&model-diagnostics" },
      { id: "day13", title: "Day 13: Model Interpretability & Explainability", path: "/courses/ai/modele-interpretability&explainability" },
      { id: "day14", title: "Day 14: Fairness, Bias & Ethics in AI", path: "/courses/ai/fairness,bias&ethics" },
      { id: "day15", title: "Day 15: AI Governance & Risk Management", path: "/courses/ai/ai-governance&risk-management" },
    ],
  },
  {
    id: "201",
    title: "201: Neural Networks & Supervised Learning",
    subItems: [
      { id: "day16", title: "Day 16: Introduction to Neural Networks", path: "/courses/ai/neural-network-intro" },
      { id: "day17", title: "Day 17: Perceptron & Multi-Layer Perceptron", path: "/courses/ai/perceptron-mlp" },
      { id: "day18", title: "Day 18: Weight Initialization Strategies", path: "/courses/ai/weight-initialization" },
      { id: "day19", title: "Day 19: Gradient Descent Variants", path: "/courses/ai/gradient-descent-variants" },
      { id: "day20", title: "Day 20: Batch Normalization & Layer Norm", path: "/courses/ai/batch-layer-normalization" },
      { id: "day21", title: "Day 21: Introduction to Convolutional Neural Networks (CNN)", path: "/courses/ai/intro-to-cnn" },
      { id: "day22", title: "Day 22: CNN Architecture & Filters", path: "/courses/ai/cnn-architecture" },
      { id: "day23", title: "Day 23: Pooling & Stride Techniques", path: "/courses/ai/pooling-stride" },
      { id: "day24", title: "Day 24: CNN in Computer Vision", path: "/courses/ai/cnn-computer-vision" },
      { id: "day25", title: "Day 25: Introduction to Recurrent Neural Networks (RNN)", path: "/courses/ai/intro-to-rnn" },
      { id: "day26", title: "Day 26: Sequence Modeling & Time Series", path: "/courses/ai/sequence-modeling" },
      { id: "day27", title: "Day 27: Long Short-Term Memory (LSTM)", path: "/courses/ai/lstm-explained" },
      { id: "day28", title: "Day 28: Gated Recurrent Units (GRU)", path: "/courses/ai/gru-explained" },
      { id: "day29", title: "Day 29: Bidirectional & Deep RNNs", path: "/courses/ai/bidirectional-rnn" },
      { id: "day30", title: "Day 30: Attention Mechanisms (Classic)", path: "/courses/ai/attention-mechanism" },
      { id: "day31", title: "Day 31: Transformer Architecture Overview", path: "/courses/ai/transformer-overview" },
      { id: "day32", title: "Day 32: Positional Encoding in Transformers", path: "/courses/ai/positional-encoding" },
      { id: "day33", title: "Day 33: Self-Attention & Multi-Head Attention", path: "/courses/ai/self-attention" },
      { id: "day34", title: "Day 34: Encoder-Decoder Structure", path: "/courses/ai/transformer-encoder-decoder" },
      { id: "day35", title: "Day 35: Transfer Learning & Pretraining", path: "/courses/ai/transfer-learning" },
      { id: "day36", title: "Day 36: Fine-tuning Pretrained Models", path: "/courses/ai/fine-tuning" },
      { id: "day37", title: "Day 37: Data Augmentation Techniques", path: "/courses/ai/data-augmentation" },
      { id: "day38", title: "Day 38: Hyperparameter Tuning Strategies", path: "/courses/ai/hyperparameter-tuning" },
      { id: "day39", title: "Day 39: Supervised Learning Best Practices", path: "/courses/ai/supervised-learning-best-practices" },
      { id: "day40", title: "Day 40: Model Deployment Basics", path: "/courses/ai/model-deployment-basics" },
    ],
  },
  {
    id: "301",
    title: "301: Deep Learning Architectures & Use Cases",
    subItems: [
      { id: "day41", title: "Day 41: Introduction to CNNs", path: "/courses/ai/deep-cnn-intro" },
      { id: "day42", title: "Day 42: CNN Filters & Feature Maps", path: "/courses/ai/cnn-featuremaps" },
      { id: "day43", title: "Day 43: Pooling Layers & Spatial Reduction", path: "/courses/ai/cnn-pooling" },
      { id: "day44", title: "Day 44: CNN for Image Classification", path: "/courses/ai/cnn-image-classification" },
      { id: "day45", title: "Day 45: Transfer Learning with Pretrained CNNs", path: "/courses/ai/cnn-transfer-learning" },
      { id: "day46", title: "Day 46: Introduction to RNNs & Sequence Models", path: "/courses/ai/rnn-intro-sequence" },
      { id: "day47", title: "Day 47: RNN Variants: LSTM & GRU", path: "/courses/ai/rnn-lstm-gru" },
      { id: "day48", title: "Day 48: Applications of RNNs: Text & Time Series", path: "/courses/ai/rnn-usecases" },
      { id: "day49", title: "Day 49: Encoder-Decoder Architecture", path: "/courses/ai/encoder-decoder-architecture" },
      { id: "day50", title: "Day 50: Attention Mechanism in Deep Learning", path: "/courses/ai/attention-in-deep-learning" },
      { id: "day51", title: "Day 51: Introduction to Transformers", path: "/courses/ai/transformers-intro" },
      { id: "day52", title: "Day 52: Self-Attention & Positional Encoding", path: "/courses/ai/self-attention-positional" },
      { id: "day53", title: "Day 53: Transformer-based Architectures (BERT, GPT)", path: "/courses/ai/transformer-bert-gpt" },
      { id: "day54", title: "Day 54: Fine-tuning Language Models", path: "/courses/ai/fine-tune-language-models" },
      { id: "day55", title: "Day 55: Sequence-to-Sequence with Transformers", path: "/courses/ai/seq2seq-transformers" },
      { id: "day56", title: "Day 56: Image Captioning & Vision-Language Models", path: "/courses/ai/vision-language-models" },
      { id: "day57", title: "Day 57: Multi-Modal Models: CLIP, Flamingo", path: "/courses/ai/multimodal-clip-flamingo" },
      { id: "day58", title: "Day 58: Generative Models Overview (GANs, VAEs)", path: "/courses/ai/generative-overview" },
      { id: "day59", title: "Day 59: Generative Adversarial Networks (GANs)", path: "/courses/ai/gans" },
      { id: "day60", title: "Day 60: Stable Diffusion & Text-to-Image Generation", path: "/courses/ai/stable-diffusion-text2img" }
    ],
  },
  {
    id: "401",
    title: "401: Reinforcement Learning & Autonomous Systems",
    subItems: [
      { id: "day61", title: "Day 61: Introduction to Reinforcement Learning", path: "/courses/ai/rl-intro" },
      { id: "day62", title: "Day 62: Markov Decision Processes (MDP)", path: "/courses/ai/markov-decision-process" },
      { id: "day63", title: "Day 63: Bellman Equations & Value Iteration", path: "/courses/ai/bellman-value-iteration" },
      { id: "day64", title: "Day 64: Policy vs Value-Based Methods", path: "/courses/ai/policy-vs-value-methods" },
      { id: "day65", title: "Day 65: Q-Learning & ε-Greedy Exploration", path: "/courses/ai/q-learning-epsilon" },
      { id: "day66", title: "Day 66: Deep Q Networks (DQN)", path: "/courses/ai/deep-q-networks" },
      { id: "day67", title: "Day 67: Policy Gradient & REINFORCE Algorithm", path: "/courses/ai/policy-gradient-reinforce" },
      { id: "day68", title: "Day 68: Actor-Critic Methods", path: "/courses/ai/actor-critic-methods" },
      { id: "day69", title: "Day 69: Advantage Actor-Critic (A2C, A3C)", path: "/courses/ai/advantage-actor-critic" },
      { id: "day70", title: "Day 70: Proximal Policy Optimization (PPO)", path: "/courses/ai/proximal-policy-optimization" },
      { id: "day71", title: "Day 71: Reward Engineering & Shaping", path: "/courses/ai/reward-engineering" },
      { id: "day72", title: "Day 72: Exploration vs Exploitation Tradeoff", path: "/courses/ai/exploration-vs-exploitation" },
      { id: "day73", title: "Day 73: Multi-Agent Reinforcement Learning", path: "/courses/ai/multi-agent-rl" },
      { id: "day74", title: "Day 74: Hierarchical RL & Temporal Abstraction", path: "/courses/ai/hierarchical-rl" },
      { id: "day75", title: "Day 75: Simulation Environments (Gym, Isaac, Habitat)", path: "/courses/ai/rl-simulation-envs" },
      { id: "day76", title: "Day 76: RL in Robotics & Control Systems", path: "/courses/ai/rl-in-robotics" },
      { id: "day77", title: "Day 77: Safe Reinforcement Learning", path: "/courses/ai/safe-rl" },
      { id: "day78", title: "Day 78: Human-in-the-Loop RL", path: "/courses/ai/human-in-loop-rl" },
      { id: "day79", title: "Day 79: Offline RL & Batch-Constrained Learning", path: "/courses/ai/offline-rl" },
      { id: "day80", title: "Day 80: Real-World RL Challenges & Scaling", path: "/courses/ai/realworld-rl-challenges" }
    ],
  },

  {
    id: "501",
    title: "501: AI in Production & Future Systems",
    subItems: [
      { id: "day81", title: "Day 81: From Research to Production", path: "/courses/ai/research-to-production" },
      { id: "day82", title: "Day 82: ML Engineering & Deployment Pipelines", path: "/courses/ai/ml-engineering-pipelines" },
      { id: "day83", title: "Day 83: Continuous Training (CT) & Continuous Evaluation (CE)", path: "/courses/ai/continuous-training-eval" },
      { id: "day84", title: "Day 84: Model Monitoring & Drift Detection", path: "/courses/ai/model-monitoring-drift" },
      { id: "day85", title: "Day 85: Scalable Inference & Serving Architectures", path: "/courses/ai/scalable-inference-serving" },
      { id: "day86", title: "Day 86: MLOps Foundations & Tooling (MLflow, TFX, Kubeflow)", path: "/courses/ai/mlops-foundations-tools" },
      { id: "day87", title: "Day 87: Model Versioning, Registry & Rollback", path: "/courses/ai/model-versioning-registry" },
      { id: "day88", title: "Day 88: A/B Testing & Online Experimentation", path: "/courses/ai/ab-testing-experimentation" },
      { id: "day89", title: "Day 89: Responsible AI & Model Accountability", path: "/courses/ai/responsible-ai-accountability" },
      { id: "day90", title: "Day 90: Federated Learning & Privacy-Preserving AI", path: "/courses/ai/federated-learning-privacy" },
      { id: "day91", title: "Day 91: Edge AI & TinyML Applications", path: "/courses/ai/edge-ai-tinyml" },
      { id: "day92", title: "Day 92: Generative AI in Production", path: "/courses/ai/generative-ai-production" },
      { id: "day93", title: "Day 93: Prompt Engineering & Fine-Tuning LLMs", path: "/courses/ai/prompting-fine-tuning-llms" },
      { id: "day94", title: "Day 94: Retrieval-Augmented Generation (RAG) Systems", path: "/courses/ai/retrieval-augmented-generation" },
      { id: "day95", title: "Day 95: AI Governance at Scale & Global Regulations", path: "/courses/ai/ai-governance-global" },
      { id: "day96", title: "Day 96: Responsible Scaling & Sustainability", path: "/courses/ai/responsible-scaling-sustainability" },
      { id: "day97", title: "Day 97: AI in Society: Disruption, Labor & Policy", path: "/courses/ai/ai-society-disruption-policy" },
      { id: "day98", title: "Day 98: AGI, Safety & Existential Risks", path: "/courses/ai/agi-safety-existential-risk" },
      { id: "day99", title: "Day 99: The Future of AI: Trends, Frontiers & Open Questions", path: "/courses/ai/future-ai-trends" },
      { id: "day100", title: "Day 100: Final Review, Summary & Personal AI Roadmap", path: "/courses/ai/day100-review-roadmap" }
    ],
  },
  
  
  
];


const AiMobileMenu = ({ onClose, theme, setTheme }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState({});

  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  const toggleSection = (id) => {
    setExpandedSections((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <div
      className={`fixed top-0 left-0 w-64 h-full p-4 z-50 shadow-lg transition-all duration-300 ${
        theme === "dark" ? "bg-gray-900 text-white" : "bg-white text-black"
      } overflow-y-auto pb-20`}
    >
      {/* ❌ ปุ่มปิดเมนู */}
      <button
        className={`absolute right-4 top-4 text-2xl transition-colors duration-200 ${
          theme === "dark"
            ? "text-white hover:text-gray-400"
            : "text-black hover:text-gray-600"
        }`}
        onClick={onClose}
      >
        <FaTimes />
      </button>

      {/* ✅ Logo + ชื่อ + ปุ่มสลับธีม */}
      <div className="mt-6 flex items-center mb-3">
        <img
          src="/spm2.jpg"
          alt="Logo"
          className="w-8 h-8 mr-2 object-cover rounded-full"
        />
        <div className="flex items-center space-x-2">
          <span className="text-lg font-bold cursor-pointer hover:text-gray-400 transition">
            Superbear
          </span>
          <button
            className="cursor-pointer transition-transform transform hover:scale-110"
            onClick={toggleTheme}
          >
            {theme === "dark" ? (
              <FiSun className="text-yellow-400 text-2xl" />
            ) : (
              <FaMoon className="text-blue-400 text-2xl" />
            )}
          </button>
        </div>
      </div>

      {/* ✅ ปุ่มย้อนกลับ */}
      <button
        className={`w-full text-left text-sm font-medium px-5 py-3 rounded-lg mb-4 transition 
          ${theme === "dark" ? "bg-gray-800 text-white hover:bg-gray-700" : "bg-gray-200 text-black hover:bg-gray-300"}`}
        onClick={() => {
          navigate("/courses/ai-series");
          onClose();
        }}
      >
        <FaArrowLeft className="inline-block mr-2" /> AI Series
      </button>

      {/* ✅ เมนูรายวัน (dropdown) */}
      <ul className="space-y-2 mt-4">
        {sidebarItems.map((item) => (
          <li key={item.id} className="border-b border-gray-700">
            <button
              className="flex items-center justify-between w-full p-4 rounded-lg transition duration-300 ease-in-out
                hover:bg-gray-700 hover:shadow-lg text-left"
              onClick={() => toggleSection(item.id)}
            >
              {item.title}
              {expandedSections[item.id] ? <FaChevronDown /> : <FaChevronRight />}
            </button>

            {expandedSections[item.id] && (
              <ul className="pl-5 space-y-2 mt-2">
                {item.subItems.map((subItem) => (
                  <li
                    key={subItem.id}
                    className={`p-2 rounded-lg cursor-pointer transition duration-200 ${
                      location.pathname === subItem.path
                        ? "bg-green-500 text-white font-bold"
                        : "hover:bg-gray-600"
                    }`}
                    onClick={() => {
                      navigate(subItem.path);
                      onClose();
                    }}
                  >
                    {subItem.title}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AiMobileMenu;
