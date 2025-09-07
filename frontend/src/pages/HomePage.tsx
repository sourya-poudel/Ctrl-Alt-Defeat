import React, { useEffect, useState } from "react";
import { Helmet } from "react-helmet";
import {
  UserCircle,
  LayoutDashboard,
  Activity,
  Video,
  Brush,
  Gavel,
  Search,
  LogOut,
  Menu,
  Bell,
  Sun,
  Moon,
  MoreVertical,
  Book,
  UserPlus,
} from "lucide-react";

const HomePage: React.FC = () => {
  const [stationName, setStationName] = useState<string>("");
  const [darkMode, setDarkMode] = useState<boolean>(true); // default: dark

  useEffect(() => {
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme) {
      setDarkMode(storedTheme === "dark");
    }
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  useEffect(() => {
    const storedStation = localStorage.getItem("policeStation");
    if (storedStation) {
      const station = JSON.parse(storedStation);
      setStationName(station.chaukiName || "Unknown Station");
    }
  }, []);

  return (
    <>
      <Helmet>
        <title>Dashboard | Eagle Eye</title>
      </Helmet>

      <div className="flex min-h-screen bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100 transition-colors">
        {/* Sidebar */}
        <aside className="w-64 bg-white dark:bg-gray-800 border-r dark:border-gray-700 p-4 flex flex-col">
          <div className="flex items-center gap-2 mb-8">
            <img src="img/icon-white.png" alt="Logo" className="w-8 h-8" />
            <h2 className="text-xl font-semibold">
              Eagle<span className="text-blue-600">Eye</span>
            </h2>
          </div>

          <nav className="flex flex-col gap-2">
            <a
              href="/home"
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-100 text-blue-700 font-medium dark:bg-blue-900/30 dark:text-blue-400"
            >
              <LayoutDashboard className="w-5 h-5" /> Home
            </a>
            <a
              href="/analyze"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Activity className="w-5 h-5" /> Analyze
            </a>
            <a
              href="/live"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Video className="w-5 h-5" /> Live
            </a>
            <a
              href="/sketch"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Brush className="w-5 h-5" /> Sketch
            </a>
            <a
              href="/records"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Gavel className="w-5 h-5" /> Criminal
            </a>
            <a
              href="/missing"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Search className="w-5 h-5" /> Missing
            </a>
          </nav>

          <div className="mt-auto pt-6 border-t dark:border-gray-700">
            <a
              href="/"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-red-600"
            >
              <LogOut className="w-5 h-5" /> Logout
            </a>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-y-auto">
          <h1 className="text-2xl font-semibold mb-6">Services</h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <a
              href="/live"
              className="p-6 bg-white dark:bg-gray-800 rounded-lg border dark:border-gray-700 hover:shadow-md transition"
            >
              <Video className="w-8 h-8 text-blue-600 mb-4" />
              <h2 className="text-lg font-medium">Live CCTV Detection</h2>
            </a>
            <a
              href="/analyze"
              className="p-6 bg-white dark:bg-gray-800 rounded-lg border dark:border-gray-700 hover:shadow-md transition"
            >
              <Search className="w-8 h-8 text-blue-600 mb-4" />
              <h2 className="text-lg font-medium">CCTV Footage Analyzer</h2>
            </a>
            <a
              href="/sketch"
              className="p-6 bg-white dark:bg-gray-800 rounded-lg border dark:border-gray-700 hover:shadow-md transition"
            >
              <Brush className="w-8 h-8 text-blue-600 mb-4" />
              <h2 className="text-lg font-medium">Sketch to Image</h2>
            </a>
            <a
              href="/records"
              className="p-6 bg-white dark:bg-gray-800 rounded-lg border dark:border-gray-700 hover:shadow-md transition"
            >
              <Book className="w-8 h-8 text-blue-600 mb-4" />
              <h2 className="text-lg font-medium">Criminal Records</h2>
            </a>
            <a
              href="/missing"
              className="p-6 bg-white dark:bg-gray-800 rounded-lg border dark:border-gray-700 hover:shadow-md transition"
            >
              <UserPlus className="w-8 h-8 text-blue-600 mb-4" />
              <h2 className="text-lg font-medium">Missing Records</h2>
            </a>
          </div>
        </main>

        {/* Right Section */}
        <div className="w-80 border-l dark:border-gray-700 bg-white dark:bg-gray-800 flex flex-col">
          <div className="flex items-center justify-between p-4 border-b dark:border-gray-700">
            <div className="flex items-center gap-2">
              <UserCircle className="w-7 h-7 text-blue-600" />
              <span className="font-medium">{stationName}</span>
            </div>
            <button className="md:hidden">
              <Menu className="w-6 h-6" />
            </button>
          </div>

          {/* Dark mode toggle */}
          <div className="flex items-center gap-4 p-4 border-b dark:border-gray-700">
            <button
              onClick={() => setDarkMode(false)}
              className={`p-2 rounded ${!darkMode ? "bg-gray-200 dark:bg-gray-700" : ""}`}
            >
              <Sun className="w-5 h-5 text-yellow-500" />
            </button>
            <button
              onClick={() => setDarkMode(true)}
              className={`p-2 rounded ${darkMode ? "bg-gray-200 dark:bg-gray-700" : ""}`}
            >
              <Moon className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Notifications */}
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">History</h2>
              <Bell className="w-5 h-5 text-gray-500" />
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg border dark:border-gray-600">
                <UserCircle className="w-6 h-6 text-gray-500" />
                <div className="flex-1">
                  <h3 className="font-medium">Suspect "Rajan Poudel"</h3>
                  <small className="text-sm text-gray-500 dark:text-gray-400">
                    Detected at CCTV 01 at Baneshwor
                  </small>
                </div>
                <a href="/live">
                  <MoreVertical className="w-4 h-4 text-gray-400" />
                </a>
              </div>

              <div className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg border dark:border-gray-600">
                <UserCircle className="w-6 h-6 text-gray-500" />
                <div className="flex-1">
                  <h3 className="font-medium">Suspect "Sourya Poudel"</h3>
                  <small className="text-sm text-gray-500 dark:text-gray-400">
                    Detected at CCTV 05 at Putalisadak
                  </small>
                </div>
                <a href="/live">
                  <MoreVertical className="w-4 h-4 text-gray-400" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default HomePage;
