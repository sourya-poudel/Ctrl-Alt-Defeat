import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Upload,
  Image as ImageIcon,
  Loader2,
  Download,
} from "lucide-react";

const SketchToImagePage: React.FC = () => {
  const navigate = useNavigate();
  const [sketch, setSketch] = useState<File | null>(null);
  const [sketchPreview, setSketchPreview] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSketchUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSketch(file);
      setError("");
      const reader = new FileReader();
      reader.onloadend = () => setSketchPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSketch(file);
      setError("");
      const reader = new FileReader();
      reader.onloadend = () => setSketchPreview(reader.result as string);
      reader.readAsDataURL(file);
    } else {
      setError("Please upload a valid image file.");
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => setDragActive(false);

  const handleGenerate = async () => {
    if (!sketch) return;
    setIsProcessing(true);
    setError("");
    setGeneratedImage("");
    try {
      const formData = new FormData();
      formData.append("sketch", sketch);
      const response = await fetch("http://localhost:8000/sketch-to-image", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.status === "error") {
        setError(data.message || "An error occurred during image generation");
      } else {
        setGeneratedImage(data.generated_image);
      }
    } catch {
      setError("Failed to connect to the server. Please try again later.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (generatedImage) {
      const link = document.createElement("a");
      link.href = generatedImage;
      link.download = "generated-image.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/home")}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl shadow-md transition"
          >
            <ArrowLeft className="w-5 h-5" />
            <span className="font-semibold">Back</span>
          </motion.button>

          <h1 className="text-3xl font-bold">Sketch to Image</h1>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Upload Card */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-800 rounded-2xl shadow-md p-6 space-y-4"
          >
            <h2 className="text-xl font-semibold mb-2">Upload Sketch</h2>
            <div
              onClick={() => inputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`aspect-square rounded-xl border-2 border-dashed flex items-center justify-center cursor-pointer transition
                ${
                  dragActive
                    ? "border-blue-500 bg-blue-900/30"
                    : sketch
                    ? "border-blue-500"
                    : "border-gray-600"
                }
                bg-gray-700
              `}
            >
              {sketchPreview ? (
                <img
                  src={sketchPreview}
                  alt="Uploaded sketch"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-gray-400">
                  <Upload className="w-10 h-10 mb-2" />
                  <p>Click or drag & drop to upload sketch</p>
                </div>
              )}
              <input
                ref={inputRef}
                id="sketchUpload"
                type="file"
                accept="image/*"
                onChange={handleSketchUpload}
                className="hidden"
              />
            </div>
            <p className="text-sm text-gray-400">
              Supported formats: PNG, JPG, JPEG
            </p>
          </motion.div>

          {/* Generated Image Card */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-800 rounded-2xl shadow-md p-6 space-y-4"
          >
            <h2 className="text-xl font-semibold mb-2">Generated Image</h2>
            <div className="aspect-square rounded-xl bg-gray-700 flex items-center justify-center">
              {isProcessing ? (
                <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
              ) : generatedImage ? (
                <img
                  src={generatedImage}
                  alt="Generated"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <ImageIcon className="w-12 h-12 text-gray-500" />
              )}
            </div>
            <button
              onClick={handleDownload}
              disabled={!generatedImage || isProcessing}
              className={`w-full px-4 py-2 rounded-lg font-semibold flex items-center justify-center gap-2 transition
                ${
                  !generatedImage || isProcessing
                    ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                    : "bg-green-600 hover:bg-green-500 text-white"
                }
              `}
            >
              <Download className="w-5 h-5" />
              Download Image
            </button>
          </motion.div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6 p-4 bg-red-600 text-white rounded-xl text-center font-medium shadow">
            {error}
          </div>
        )}

        {/* Generate Button */}
        <div className="mt-8 flex justify-center">
          <button
            onClick={handleGenerate}
            disabled={!sketch || isProcessing}
            className={`px-8 py-3 rounded-lg font-bold text-lg transition
              ${
                !sketch || isProcessing
                  ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-500 text-white shadow"
              }
            `}
          >
            {isProcessing ? "Processing..." : "Generate Image"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SketchToImagePage;
