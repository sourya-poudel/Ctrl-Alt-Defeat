import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Camera, CameraOff, FileSearch, Database, Info, HelpCircle } from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.3,
      duration: 0.8,
      ease: "easeOut",
    },
  },
};

const itemVariants = {
  hidden: { y: 40, opacity: 0, scale: 0.95 },
  visible: {
    y: 0,
    opacity: 1,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 120,
      damping: 14,
    },
  },
};

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const buttons = [
    {
      title: "CCTV Footage Analyzer",
      description: "Analyze recorded CCTV footage for criminal activity using AI.",
      icon: <Camera className="w-8 h-8" />,
      path: "/analyze",
      gradient: "from-purple-600 to-blue-600",
    },
    {
      title: "Live CCTV Detection",
      description: "Real-time detection of suspicious activity from live CCTV feeds.",
      icon: <CameraOff className="w-8 h-8" />,
      path: "/live",
      gradient: "from-blue-600 to-cyan-600",
    },
    {
      title: "Sketch to Image",
      description: "Convert criminal sketches to realistic images for identification.",
      icon: <FileSearch className="w-8 h-8" />,
      path: "/sketch",
      gradient: "from-cyan-600 to-teal-600",
    },
    {
      title: "Records",
      description: "Browse and manage criminal records securely.",
      icon: <Database className="w-8 h-8" />,
      path: "/records",
      gradient: "from-teal-600 to-green-600",
    },
    {
      title: "About",
      description: "Learn more about our AI-powered criminal detection platform.",
      icon: <Info className="w-8 h-8" />,
      path: "/about",
      gradient: "from-indigo-600 to-purple-600",
    },
    {
      title: "Help & FAQ",
      description: "Get help and find answers to common questions.",
      icon: <HelpCircle className="w-8 h-8" />,
      path: "/help",
      gradient: "from-pink-600 to-red-600",
    },
  ];

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white px-4 py-8 flex flex-col"
    >
      {/* Animated Heading */}
      <motion.div variants={itemVariants} className="mb-10">
        <motion.h1
          variants={itemVariants}
          className="text-5xl md:text-7xl font-extrabold text-center mb-4 relative"
        >
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500 filter blur-sm absolute inset-0 animate-pulse pointer-events-none">
            AI Criminal Detection
          </span>
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500 relative">
            AI Criminal Detection
          </span>
        </motion.h1>
        <motion.p
          variants={itemVariants}
          className="text-lg md:text-2xl text-center text-gray-300 mt-4 max-w-2xl mx-auto"
        >
          Empowering law enforcement with advanced AI tools for criminal detection, analysis, and record management. Fast, secure, and easy to use on any device.
        </motion.p>
      </motion.div>

      {/* Info Banner */}
      <motion.div
        variants={itemVariants}
        className="bg-gradient-to-r from-blue-700 to-purple-700 rounded-xl shadow-lg p-4 mb-8 mx-auto max-w-3xl flex flex-col md:flex-row items-center justify-between gap-4"
      >
        <span className="text-white font-medium text-center md:text-left">
          <span className="font-bold">New:</span> Try our real-time CCTV detection for instant alerts!
        </span>
        <button
          onClick={() => navigate("/live")}
          className="bg-white text-blue-700 font-semibold px-4 py-2 rounded-lg shadow hover:bg-blue-100 transition"
        >
          Try Now
        </button>
      </motion.div>

      {/* Navigation Grid */}
      <motion.div
        variants={containerVariants}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 w-full max-w-7xl mx-auto"
      >
        {buttons.map((button) => (
          <motion.div
            key={button.title}
            variants={itemVariants}
            whileHover={{ scale: 1.04, boxShadow: "0 0 32px 0 rgba(99,102,241,0.25)" }}
            whileTap={{ scale: 0.97 }}
            className={`bg-gradient-to-r ${button.gradient} p-0.5 rounded-2xl cursor-pointer transition-all duration-300 hover:shadow-2xl`}
          >
            <button
              onClick={() => navigate(button.path)}
              className="w-full h-full bg-gray-900 rounded-2xl p-6 md:p-8 flex flex-col items-center justify-center gap-4 hover:bg-opacity-90 transition-all duration-300"
            >
              <div className="p-4 bg-gray-800 rounded-full shadow-lg flex items-center justify-center">
                {button.icon}
              </div>
              <h2 className="text-xl md:text-2xl font-semibold text-center">
                {button.title}
              </h2>
              <p className="text-gray-300 text-center text-sm md:text-base">
                {button.description}
              </p>
            </button>
          </motion.div>
        ))}
      </motion.div>

      {/* Footer */}
      <motion.footer
        variants={itemVariants}
        className="mt-16 text-center text-gray-400 text-sm"
      >
        &copy; {new Date().getFullYear()} AI Criminal Detection Platform. All rights reserved.
      </motion.footer>
    </motion.div>
  );
};

export default HomePage;
