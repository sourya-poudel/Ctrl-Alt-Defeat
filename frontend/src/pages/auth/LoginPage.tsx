// Move content from src/components/LoginPage.tsx
import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import { Lock, AlertCircle, Eye, EyeOff, Loader2 } from 'lucide-react';
import { collection, query, where, getDocs } from 'firebase/firestore';
import { db } from '../../firebase/config';
import { useNavigate } from 'react-router-dom';

interface PoliceStation {
  value: string;
  label: string;
}
const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const [selectedStation, setSelectedStation] = useState<PoliceStation | null>(null);
  const [chaukiID, setchaukiID] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [policeStations, setPoliceStations] = useState<PoliceStation[]>([]);
  const [isLoadingStations, setIsLoadingStations] = useState(true);
  useEffect(() => {
    const fetchPoliceStations = async () => {
      setIsLoadingStations(true);
      try {
        const policeStationsRef = collection(db, 'police_stations');
        console.log(policeStationsRef)
        const q = query(policeStationsRef, where('active', '==', true));
        const querySnapshot = await getDocs(q);
        console.log('Number of stations fetched:', querySnapshot.size);     
        const stations: PoliceStation[] = [];
        querySnapshot.forEach((doc) => {
          console.log('Doc:', doc.id, doc.data());
          const data = doc.data();
          stations.push({
            value: doc.id,
            label: data.chaukiName || 'Unknown Station'
          });
        });
        setPoliceStations(stations);
      } catch (err) {
        console.error('Error fetching police stations:', err);
        setError('Failed to load police stations. Please try again later.');
      } finally {
        setIsLoadingStations(false);
      }
    };
    fetchPoliceStations();
  }, []);
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      if (!selectedStation || !chaukiID || !password) {
        throw new Error('Please fill in all fields');
      }
      const policeStationsRef = collection(db, 'police_stations');
      const q = query(
        policeStationsRef,
        where('chaukiID', '==', chaukiID),
        where('active', '==', true)
      );
      const querySnapshot = await getDocs(q);
      if (querySnapshot.empty) {
        throw new Error('Invalid credentials');
      }
      const policeStation = querySnapshot.docs[0].data();
      if (policeStation.password !== password) {
        throw new Error('Invalid credentials');
      }
      localStorage.setItem('policeStation', JSON.stringify({
        id: querySnapshot.docs[0].id,
        chaukiID: policeStation.chaukiID,
        chaukiName: policeStation.chaukiName
      }));
      navigate('/home');
    } catch (err) {
      console.error('Login error:', err);
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <Lock className="w-12 h-12 text-blue-600 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-800">Police Station Login</h1>
          <p className="text-gray-600 mt-2">Enter your credentials to continue</p>
          
        </div>

        <form onSubmit={handleLogin} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Police Station
            </label>
            {isLoadingStations ? (
              <div className="flex items-center justify-center py-2">
                <Loader2 className="w-5 h-5 text-blue-600 animate-spin mr-2" />
                <span className="text-sm text-gray-500">Loading stations...</span>
              </div>
            ) : (
              <Select
                options={policeStations}
                value={selectedStation}
                onChange={setSelectedStation}
                className="w-full"
                classNamePrefix="select"
                placeholder={policeStations.length ? "Search police station..." : "No stations available"}
                isSearchable
                isDisabled={policeStations.length === 0}
                noOptionsMessage={() => "No matching police stations"}
              />
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chauki ID
            </label>
            <input
              type="text"
              value={chaukiID}
              onChange={(e) => setchaukiID(e.target.value)}
              className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all ${error && !chaukiID ? 'border-red-500' : 'border-gray-300'
                }`}
              placeholder="Enter your Chauki ID"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition-all ${error && !password ? 'border-red-500' : 'border-gray-300'
                  }`}
                placeholder="Enter your password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 transition-colors"
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg">
              <AlertCircle className="w-5 h-5" />
              <span className="text-sm">{error}</span>
              
            </div>
          )}
<p className="text-gray-600 mt-2"><a href="/register">Haven't registered your chauki yet?</a></p>
          <button
            type="submit"
            disabled={isLoading || isLoadingStations}
            className={`w-full bg-blue-600 text-white py-2 rounded-lg font-medium
              transition-all duration-300 hover:bg-blue-700 focus:ring-4 focus:ring-blue-200
              ${(isLoading || isLoadingStations) ? 'opacity-75 cursor-not-allowed' : ''}`}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;