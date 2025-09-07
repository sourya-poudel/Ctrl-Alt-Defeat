import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft, Camera, AlertCircle, Upload,
  UserPlus, Play, Loader2, Video, VideoOff
} from 'lucide-react';

// API endpoint constants
const API_BASE_URL = 'http://localhost:8000';
const UPLOAD_SUSPECT_ENDPOINT = 'http://localhost:8000/upload-suspect';
const PROCESS_FRAME_ENDPOINT = 'http://localhost:8000/process-live-frame';

const FRAME_PROCESS_INTERVAL = 500;
const IMAGE_QUALITY = 0.6;

interface Suspect {
  id: string;
  name: string;
  crime: string;
  photo: File;
  photoPreview: string;
}

interface Detection {
  bbox: number[];
  name: string;
  location: string;
  similarity: number;
  recognized: boolean;
  image_path?: string;
}

const LiveDetectionPage: React.FC = () => {
  const navigate = useNavigate();
  const [suspects, setSuspects] = useState<Suspect[]>([]);
  const [currentName, setCurrentName] = useState('');
  const [currentCrime, setCurrentCrime] = useState('');
  const [currentPhoto, setCurrentPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [detectedSuspect, setDetectedSuspect] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [processingEnabled, setProcessingEnabled] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const processingAnimationRef = useRef<NodeJS.Timeout | null>(null);

  const lastProcessedTime = useRef<number>(0);
  const processingInProgress = useRef<boolean>(false);
  const [frameRate, setFrameRate] = useState<number>(0);
  const frameCounter = useRef<number>(0);
  const lastFpsUpdateTime = useRef<number>(performance.now());
  const currentDetectionsRef = useRef<Detection[]>([]);

  const userPoliceStation = localStorage.getItem('police_station');
  const policeStation =
    userPoliceStation === 'Kathmandu Central Police'
      ? 'Kathmandu Central Police'
      : 'Chitwan Police Station';

  // Drag-and-drop for photo upload
  const [dragActive, setDragActive] = useState(false);
  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handlePhotoChange({ target: { files: e.dataTransfer.files } } as any);
    }
  };

  const drawVideoFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (currentDetectionsRef.current.length > 0) {
      currentDetectionsRef.current.forEach((detection: Detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        context.strokeStyle = detection.recognized ? '#22c55e' : '#ef4444';
        context.lineWidth = 2;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const labelText = `${detection.name} (${detection.similarity.toFixed(2)})`;
        const labelWidth = context.measureText(labelText).width + 10;
        context.fillStyle = detection.recognized ? 'rgba(34,197,94,0.8)' : 'rgba(239,68,68,0.8)';
        context.fillRect(x1, y1 - 25, labelWidth, 25);
        context.fillStyle = '#fff';
        context.font = '16px Arial';
        context.fillText(labelText, x1 + 5, y1 - 5);
      });
    }
    const now = performance.now();
    frameCounter.current++;
    if (now - lastFpsUpdateTime.current >= 1000) {
      setFrameRate(Math.round((frameCounter.current * 1000) / (now - lastFpsUpdateTime.current)));
      frameCounter.current = 0;
      lastFpsUpdateTime.current = now;
    }
    animationRef.current = requestAnimationFrame(drawVideoFrame);
  }, [isStreaming]);

  const processFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !processingEnabled) return;
    const now = performance.now();
    const shouldProcess = now - lastProcessedTime.current >= FRAME_PROCESS_INTERVAL && !processingInProgress.current;
    if (shouldProcess && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
      processingInProgress.current = true;
      lastProcessedTime.current = now;
      try {
        const canvas = canvasRef.current;
        const imageData = canvas.toDataURL('image/jpeg', IMAGE_QUALITY);
        const response = await fetch(PROCESS_FRAME_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify({ image: imageData }),
          mode: 'cors',
        });
        if (!response.ok) {
          processingInProgress.current = false;
          return;
        }
        const result = await response.json();
        if (result.status === 'success') {
          setDetections(result.detections || []);
          currentDetectionsRef.current = result.detections || [];
          const recognizedSuspects = result.detections?.filter((d: Detection) => d.recognized);
          if (recognizedSuspects?.length > 0) {
            setDetectedSuspect(recognizedSuspects[0].name);
            setTimeout(() => setDetectedSuspect(null), 3000);
          }
        }
      } catch (err) {
        // silent error
      } finally {
        processingInProgress.current = false;
      }
    }
    processingAnimationRef.current = setTimeout(() => {
      if (processingEnabled) processFrame();
    }, 100);
  }, [processingEnabled]);

  const startWebcam = async () => {
    try {
      setCameraError(null);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(() => setCameraError('Failed to play video stream.'));
        };
      }
      setIsStreaming(true);
    } catch {
      setCameraError('Failed to access webcam. Please check your camera and permissions.');
      setIsStreaming(false);
    }
  };

  const stopWebcam = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    if (processingAnimationRef.current) clearTimeout(processingAnimationRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsStreaming(false);
    setProcessingEnabled(false);
    setDetections([]);
    currentDetectionsRef.current = [];
  };

  const handlePhotoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCurrentPhoto(file);
      const reader = new FileReader();
      reader.onloadend = () => setPhotoPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleAddSuspect = async () => {
    if (!currentName || !currentCrime || !currentPhoto) {
      setError('Please fill in all fields and upload a photo');
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('suspect_image', currentPhoto);
      formData.append('suspect_name', currentName);
      formData.append('police_station', policeStation);
      const response = await fetch(UPLOAD_SUSPECT_ENDPOINT, {
        method: 'POST',
        body: formData,
        mode: 'cors',
        headers: { 'Accept': 'application/json' },
      });
      if (!response.ok) {
        try {
          const errorData = await response.json();
          throw new Error(errorData.message || `Server error: ${response.status}`);
        } catch {
          throw new Error(`Server error: ${response.status}`);
        }
      }
      const newSuspect: Suspect = {
        id: Date.now().toString(),
        name: currentName,
        crime: currentCrime,
        photo: currentPhoto,
        photoPreview: photoPreview
      };
      setSuspects([...suspects, newSuspect]);
      setCurrentName('');
      setCurrentCrime('');
      setCurrentPhoto(null);
      setPhotoPreview('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload error');
    } finally {
      setIsUploading(false);
    }
  };

  const startDetection = () => {
    if (suspects.length === 0) {
      setError('Please add at least one suspect before starting detection');
      return;
    }
    setIsProcessing(true);
    setError(null);
    if (!isStreaming) {
      startWebcam();
    } else {
      setProcessingEnabled(true);
      processFrame();
    }
    setTimeout(() => setIsProcessing(false), 3000);
  };

  useEffect(() => {
    if (isStreaming) {
      animationRef.current = requestAnimationFrame(drawVideoFrame);
      if (processingEnabled) processFrame();
    }
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (processingAnimationRef.current) clearTimeout(processingAnimationRef.current);
    };
  }, [isStreaming, processingEnabled, drawVideoFrame, processFrame]);

  useEffect(() => {
    if (isStreaming) {
      if (videoRef.current && !videoRef.current.srcObject && streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
        videoRef.current.play().catch(() => setCameraError('Failed to play video stream.'));
      }
    }
  }, [isStreaming]);

  useEffect(() => () => stopWebcam(), []);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-blue-900 via-BLUE-900 to-gray-900 text-white p-0">
      <div className="max-w-7xl mx-auto py-10 px-4">
        <div className="flex items-center gap-4 mb-12">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/home')}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-700 to-BLUE-700 rounded-xl shadow hover:shadow-lg transition"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={isStreaming ? stopWebcam : startWebcam}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl shadow transition ${
              isStreaming
                ? 'bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700'
                : 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700'
            }`}
          >
            {isStreaming ? (
              <>
                <VideoOff className="w-5 h-5" />
                <span>Stop Camera</span>
              </>
            ) : (
              <>
                <Video className="w-5 h-5" />
                <span>Start Camera</span>
              </>
            )}
          </motion.button>
        </div>

        <motion.h1
  initial={{ opacity: 0, y: -30 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.7 }}
  className="text-5xl font-bold mb-10 text-center text-white drop-shadow-lg
    cursor-default px-4 py-2 leading-normal"
>
  Live CCTV Analyzer
</motion.h1>


        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-10"
        >
          {/* Suspect Management */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div className="bg-gradient-to-br from-blue-800 via-BLUE-800 to-gray-800 rounded-2xl shadow-xl p-8 space-y-6">
              <h2 className="text-2xl font-bold mb-2">Add Suspects for Detection</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Suspect Name
                  </label>
                  <input
                    type="text"
                    value={currentName}
                    onChange={e => setCurrentName(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400"
                    placeholder="Enter suspect's name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Crime Details
                  </label>
                  <input
                    type="text"
                    value={currentCrime}
                    onChange={e => setCurrentCrime(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400"
                    placeholder="Enter crime details"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Police Station
                  </label>
                  <input
                    type="text"
                    value={policeStation}
                    disabled
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-400 cursor-not-allowed"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Suspect Photo
                  </label>
                  <div
                    onClick={() => document.getElementById('photoUpload')?.click()}
                    onDragEnter={handleDrag}
                    onDragOver={handleDrag}
                    onDragLeave={handleDrag}
                    onDrop={handleDrop}
                    className={`relative aspect-video rounded-xl border-2 border-dashed transition-colors cursor-pointer bg-gray-700 flex items-center justify-center overflow-hidden ${
                      dragActive ? 'border-blue-400 bg-blue-900/40' : 'border-gray-600 hover:border-blue-500'
                    }`}
                  >
                    {photoPreview ? (
                      <img src={photoPreview} alt="Preview" className="w-full h-full object-cover rounded-xl" />
                    ) : (
                      <div className="text-center">
                        <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                        <p className="text-sm text-gray-400">Click or drag photo here</p>
                      </div>
                    )}
                    <input
                      id="photoUpload"
                      type="file"
                      accept="image/*"
                      onChange={handlePhotoChange}
                      className="hidden"
                    />
                  </div>
                </div>
              </div>
              {error && (
                <div className="flex items-center gap-2 text-red-500 bg-red-500/10 p-3 rounded-lg">
                  <AlertCircle className="w-5 h-5" />
                  <span className="text-sm">{error}</span>
                </div>
              )}
              <button
                onClick={handleAddSuspect}
                disabled={isUploading}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-BLUE-600 hover:from-blue-700 hover:to-BLUE-700 rounded-xl shadow transition disabled:bg-blue-800 disabled:cursor-not-allowed"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <UserPlus className="w-5 h-5" />
                    Add Suspect
                  </>
                )}
              </button>
            </div>
            <AnimatePresence>
              {suspects.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-gradient-to-br from-blue-800 via-BLUE-800 to-gray-800 rounded-2xl shadow-xl p-8"
                >
                  <h3 className="text-xl font-semibold mb-4">Added Suspects</h3>
                  <div className="space-y-4">
                    {suspects.map(suspect => (
                      <motion.div
                        key={suspect.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className="flex items-center gap-4 bg-gradient-to-r from-gray-700 via-BLUE-700 to-blue-700 rounded-xl p-4 shadow"
                      >
                        <img src={suspect.photoPreview} alt={suspect.name} className="w-16 h-16 rounded-xl object-cover shadow" />
                        <div className="flex-1">
                          <h4 className="font-medium">{suspect.name}</h4>
                          <p className="text-sm text-gray-300">{suspect.crime}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Live Detection Section */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div className="bg-gradient-to-br from-blue-800 via-BLUE-800 to-gray-800 rounded-2xl shadow-xl p-8">
              <div className="aspect-video rounded-2xl bg-black relative overflow-hidden shadow-lg">
                {isStreaming ? (
                  <>
                    <video ref={videoRef} autoPlay playsInline muted className="hidden" />
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 w-full h-full object-cover z-10 rounded-2xl"
                      style={{ transform: 'scaleX(-1)' }}
                    />
                    <div className="absolute top-24 left-4 text-xs text-white z-30 bg-black/50 p-1 rounded">
                      {videoRef.current?.videoWidth || 0}x{videoRef.current?.videoHeight || 0} | {frameRate} FPS
                    </div>
                    <div className="absolute bottom-4 left-4 flex items-center gap-2 z-20">
                      <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-sm">Live</span>
                    </div>
                    {processingEnabled && (
                      <div className="absolute top-4 right-4 bg-blue-500/80 px-3 py-1 rounded-full text-xs font-medium z-20 shadow">
                        Detection Active
                      </div>
                    )}
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <Camera className="w-16 h-16 text-gray-600" />
                    <p className="mt-4 text-gray-400">Camera not active</p>
                  </div>
                )}
                {cameraError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                    <div className="text-center p-4">
                      <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
                      <p className="text-red-400">{cameraError}</p>
                    </div>
                  </div>
                )}
                <AnimatePresence>
                  {detectedSuspect && (
                    <motion.div
                      initial={{ opacity: 0, y: 50 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 50 }}
                      className="absolute top-4 left-4 right-4 bg-red-500/90 text-white p-4 rounded-xl backdrop-blur-sm flex items-center gap-3 z-20 shadow-lg"
                    >
                      <AlertCircle className="w-6 h-6" />
                      <span className="font-medium">
                        Suspect Detected: {detectedSuspect}
                      </span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              <div className="mt-6">
                <button
                  onClick={startDetection}
                  disabled={isProcessing || suspects.length === 0}
                  className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-medium shadow transition-all ${
                    isProcessing || suspects.length === 0
                      ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-600 to-BLUE-600 hover:shadow-lg hover:shadow-blue-500/20'
                  }`}
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Start Detection
                    </>
                  )}
                </button>
              </div>
            </div>
            <div className="bg-gradient-to-br from-blue-800 via-BLUE-800 to-gray-800 rounded-2xl shadow-xl p-8">
              <h3 className="text-xl font-semibold mb-4">Detection Stats</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gradient-to-r from-blue-700 to-BLUE-700 rounded-xl p-4 flex items-center gap-3 shadow">
                  <UserPlus className="w-6 h-6 text-blue-300" />
                  <div>
                    <p className="text-sm text-gray-300">Suspects Added</p>
                    <p className="text-2xl font-bold">{suspects.length}</p>
                  </div>
                </div>
                <div className="bg-gradient-to-r from-BLUE-700 to-blue-700 rounded-xl p-4 flex items-center gap-3 shadow">
                  <Play className="w-6 h-6 text-BLUE-300" />
                  <div>
                    <p className="text-sm text-gray-300">Detection Status</p>
                    <p className="text-2xl font-bold">
                      {processingEnabled ? 'Active' : 'Inactive'}
                    </p>
                  </div>
                </div>
              </div>
              {detections.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm text-gray-400 mb-2">Recent Detections</p>
                  <div className="space-y-2">
                    {detections.slice(0, 3).map((detection, index) => (
                      <div
                        key={index}
                        className={`px-3 py-2 rounded-xl text-sm shadow ${
                          detection.recognized
                            ? 'bg-green-500/20 text-green-300'
                            : 'bg-red-500/20 text-red-300'
                        }`}
                      >
                        {detection.name} - {detection.similarity.toFixed(2)} similarity
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default LiveDetectionPage;
