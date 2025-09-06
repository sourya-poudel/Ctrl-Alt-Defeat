import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Image as ImageIcon, Loader2, Download } from 'lucide-react';

const SketchToImagePage: React.FC = () => {
  const navigate = useNavigate();
  const [sketch, setSketch] = useState<File | null>(null);
  const [sketchPreview, setSketchPreview] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSketchUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSketch(file);
      setError('');
      const reader = new FileReader();
      reader.onloadend = () => setSketchPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSketch(file);
      setError('');
      const reader = new FileReader();
      reader.onloadend = () => setSketchPreview(reader.result as string);
      reader.readAsDataURL(file);
    } else {
      setError('Please upload a valid image file.');
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
    setError('');
    setGeneratedImage('');
    try {
      const formData = new FormData();
      formData.append('sketch', sketch);
      const response = await fetch('http://localhost:8000/sketch-to-image', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.status === 'error') {
        setError(data.message || 'An error occurred during image generation');
      } else {
        setGeneratedImage(data.generated_image);
      }
    } catch {
      setError('Failed to connect to the server. Please try again later.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (generatedImage) {
      const link = document.createElement('a');
      link.href = generatedImage;
      link.download = 'generated-image.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-gray-900 text-white p-6">
      <div className="max-w-6xl mx-auto rounded-3xl bg-gradient-to-br from-purple-800/40 to-gray-900/80 shadow-2xl p-8">
        <div className="flex items-center gap-4 mb-10">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/home')}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-700 to-pink-600 rounded-xl hover:from-purple-600 hover:to-pink-500 transition-colors shadow-lg"
          >
            <ArrowLeft className="w-5 h-5" />
            <span className="font-semibold">Back to Home</span>
          </motion.button>
        </div>
        <motion.h1
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-5xl font-extrabold mb-12 text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-indigo-400"
        >
          Sketch to Image Converter
        </motion.h1>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-10"
        >
          {/* Sketch Upload */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-bold mb-2">Upload Sketch</h2>
            <div
              onClick={() => inputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`aspect-square rounded-2xl border-4 border-dashed flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900 cursor-pointer transition-all
                ${dragActive ? 'border-pink-500 bg-purple-900/30' : sketch ? 'border-purple-500' : 'border-gray-600'}
              `}
            >
              {sketchPreview ? (
                <img
                  src={sketchPreview}
                  alt="Uploaded sketch"
                  className="w-full h-full object-contain rounded-xl"
                />
              ) : (
                <div className="flex flex-col items-center justify-center">
                  <Upload className="w-14 h-14 text-purple-400 mb-4 animate-bounce" />
                  <p className="text-gray-300 font-medium">Drag & drop or click to upload sketch</p>
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
            <p className="text-sm text-gray-400">Supported formats: PNG, JPG, JPEG</p>
          </motion.div>
          {/* Generated Image */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-bold mb-2">Generated Image</h2>
            <div className="aspect-square rounded-2xl bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center shadow-lg">
              {isProcessing ? (
                <Loader2 className="w-14 h-14 text-purple-400 animate-spin" />
              ) : generatedImage ? (
                <img
                  src={generatedImage}
                  alt="Generated"
                  className="w-full h-full object-contain rounded-xl"
                />
              ) : (
                <ImageIcon className="w-14 h-14 text-gray-500" />
              )}
            </div>
            <div className="flex justify-center">
              <button
                onClick={handleDownload}
                disabled={!generatedImage || isProcessing}
                className={`px-7 py-3 rounded-xl font-semibold text-lg flex items-center gap-2
                  transition-all duration-300 hover:scale-105
                  ${!generatedImage || isProcessing
                    ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-green-600 to-blue-600 text-white hover:shadow-lg hover:shadow-green-500/20'
                  }`}
              >
                <Download className="w-5 h-5" />
                Download Image
              </button>
            </div>
          </motion.div>
        </motion.div>
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 p-4 bg-red-900/80 text-white rounded-xl text-center font-semibold shadow-lg"
          >
            {error}
          </motion.div>
        )}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mt-10 flex justify-center"
        >
          <button
            onClick={handleGenerate}
            disabled={!sketch || isProcessing}
            className={`px-10 py-4 rounded-xl font-bold text-lg transition-all duration-300 hover:scale-105
              ${!sketch || isProcessing
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg hover:shadow-pink-500/20'
              }`}
          >
            {isProcessing ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5 inline" />
                Processing...
              </>
            ) : (
              'Generate Image'
            )}
          </button>
        </motion.div>
      </div>
    </div>
  );
};

export default SketchToImagePage;