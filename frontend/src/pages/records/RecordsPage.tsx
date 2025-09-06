import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Trash2, AlertCircle, Loader2, RefreshCw, ShieldCheck } from 'lucide-react';

// API endpoint constants
const API_BASE_URL = 'http://localhost:8000';
const GET_FACES_ENDPOINT = `${API_BASE_URL}/get-faces`;
const DELETE_FACE_ENDPOINT = `${API_BASE_URL}/delete-face`;

interface FaceData {
  id: string;
  name: string;
  location: string;
  image_path: string;
  created_at: string;
}

const RecordsPage: React.FC = () => {
  const navigate = useNavigate();
  const [faces, setFaces] = useState<FaceData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState<{[key: string]: boolean}>({});
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  // Fetch faces from database
  const fetchFaces = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch(GET_FACES_ENDPOINT, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        mode: 'cors',
      });
      if (!response.ok) throw new Error(`Server error: ${response.status} ${response.statusText}`);
      const data = await response.json();
      if (data.status === 'success') setFaces(data.faces || []);
      else throw new Error(data.message || 'Failed to fetch faces');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch faces from database');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Delete face from database
  const deleteFace = async (faceId: string) => {
    try {
      setIsDeleting(prev => ({ ...prev, [faceId]: true }));
      const response = await fetch(`${DELETE_FACE_ENDPOINT}/${faceId}`, {
        method: 'DELETE',
        headers: { 'Accept': 'application/json' },
        mode: 'cors',
      });
      if (!response.ok) throw new Error(`Server error: ${response.status} ${response.statusText}`);
      const result = await response.json();
      if (result.status === 'success') setFaces(faces.filter(face => face.id !== faceId));
      else throw new Error(result.message || 'Failed to delete face');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete face');
    } finally {
      setIsDeleting(prev => {
        const newState = { ...prev };
        delete newState[faceId];
        return newState;
      });
      setConfirmDelete(null);
    }
  };

  useEffect(() => { fetchFaces(); }, [fetchFaces]);

  // Framer Motion variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { type: "spring", stiffness: 100 } }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-950 via-blue-900 to-indigo-900 text-white">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="max-w-6xl mx-auto w-full px-4 pt-8 pb-4 flex-1"
      >
        {/* Header */}
        <motion.div
          variants={itemVariants}
          className="flex items-center justify-between gap-4 mb-10"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/home')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-900/80 rounded-xl hover:bg-blue-800 transition-colors shadow"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Home</span>
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={fetchFaces}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 rounded-xl hover:bg-indigo-700 transition-colors shadow disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <RefreshCw className="w-5 h-5" />}
            <span>Refresh</span>
          </motion.button>
        </motion.div>

        {/* Animated Heading */}
        <motion.div
          variants={itemVariants}
          className="mb-10 text-center"
        >
          <motion.h1
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 120, delay: 0.2 }}
            className="text-5xl font-extrabold bg-gradient-to-r from-indigo-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2 drop-shadow"
          >
            Face Records
          </motion.h1>
          <p className="text-blue-200 text-lg">Manage suspects and view database entries</p>
        </motion.div>

        {/* Error State */}
        {error && (
          <motion.div
            variants={itemVariants}
            className="mb-8 flex items-center gap-2 text-red-500 bg-red-500/10 p-4 rounded-xl shadow"
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </motion.div>
        )}

        {/* Loading State */}
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-12 h-12 text-indigo-400 animate-spin mb-4" />
            <p className="text-blue-200">Loading face records...</p>
          </div>
        ) : faces.length === 0 ? (
          <motion.div
            variants={itemVariants}
            className="bg-blue-900/80 rounded-2xl p-10 text-center shadow"
          >
            <ShieldCheck className="w-10 h-10 mx-auto text-blue-400 mb-4" />
            <p className="text-2xl text-blue-200 mb-2">No face records found</p>
            <p className="text-blue-400">Add suspects from the Live Detection page</p>
          </motion.div>
        ) : (
          <AnimatePresence>
            <motion.div
              variants={itemVariants}
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8"
            >
              {faces.map((face) => (
                <motion.div
                  key={face.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="bg-gradient-to-br from-blue-900 via-indigo-900 to-blue-950 rounded-2xl overflow-hidden shadow-lg flex flex-col"
                >
                  <div className="relative w-full h-56 bg-blue-950 flex-shrink-0">
                    <img
                      src={face.image_path.startsWith('http') ? face.image_path : `${API_BASE_URL}/${face.image_path}`}
                      alt={face.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'https://via.placeholder.com/300x220?text=No+Image';
                      }}
                    />
                  </div>
                  <div className="p-6 flex-1 flex flex-col justify-between">
                    <div>
                      <h2 className="text-2xl font-bold mb-1 text-blue-200">{face.name}</h2>
                      <p className="text-indigo-400 font-medium">{face.location}</p>
                      {face.created_at && (
                        <p className="text-blue-300 text-sm mt-2">
                          Added on {new Date(face.created_at).toLocaleDateString()} at {new Date(face.created_at).toLocaleTimeString()}
                        </p>
                      )}
                    </div>
                    <div className="mt-4 flex items-center justify-between">
                      <div>
                        <h3 className="text-xs font-semibold text-blue-400 mb-1">Record ID</h3>
                        <p className="font-mono text-xs text-blue-300 break-all">{face.id}</p>
                      </div>
                      <div>
                        {confirmDelete === face.id ? (
                          <motion.div
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.8, opacity: 0 }}
                            className="flex gap-2"
                          >
                            <button
                              onClick={() => deleteFace(face.id)}
                              disabled={isDeleting[face.id]}
                              className="px-3 py-1 bg-red-600 text-white rounded-lg shadow hover:bg-red-700 transition-colors flex items-center gap-1"
                            >
                              {isDeleting[face.id] ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                              Confirm
                            </button>
                            <button
                              onClick={() => setConfirmDelete(null)}
                              className="px-3 py-1 bg-blue-800 text-blue-200 rounded-lg shadow hover:bg-blue-700 transition-colors"
                            >
                              Cancel
                            </button>
                          </motion.div>
                        ) : (
                          <motion.button
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => setConfirmDelete(face.id)}
                            disabled={isDeleting[face.id]}
                            className="p-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-5 h-5" />
                          </motion.button>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </AnimatePresence>
        )}
      </motion.div>
      {/* Footer */}
      <footer className="w-full py-6 mt-8 bg-gradient-to-r from-blue-950 via-indigo-900 to-blue-950 text-center text-blue-300 rounded-t-2xl shadow-lg">
        <span className="font-semibold">© {new Date().getFullYear()} Face Detection System</span>
        <span className="mx-2">|</span>
        <span>Powered by React, Tailwind CSS & Framer Motion</span>
      </footer>
    </div>
  );
};

export default RecordsPage;
