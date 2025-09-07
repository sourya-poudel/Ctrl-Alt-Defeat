import "aos/dist/aos.css";
import "./home.css";
import { Helmet } from "react-helmet";
import { UserCircle } from "lucide-react";
import React, { useEffect, useState } from 'react';

const HomePage: React.FC = () => {
    const [stationName, setStationName] = useState<string>("");
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
        <script src="home.js"></script>
        <meta charSet="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    
    <link rel="icon" type="image/x-icon" href="img/heartbeat-solid.png"/>
    <link rel="stylesheet" href="home.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>
    <script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons+Sharp" rel="stylesheet"/>

    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.3.0/css/all.min.css"
    />
    <title>Dashboard | Eagle Eye</title>
      </Helmet>

<div className="container">
        
        <aside>
            <div className="toggle">
                <div className="logo">
                    <img src="img/heartbeat-solid.png"/>
                    <h2>Eagle<span className="danger">Eye</span></h2>
                </div>
                <div className="close" id="close-btn">
                    <span className="material-icons-sharp">
                        close
                    </span>
                </div>
            </div>

            <div className="sidebar">
                <a href="\home" className="active">
                    <span className="material-icons-sharp notranslate">
                        dashboard
                    </span>
                    <h3>Home</h3>
                </a>
              <a href="\analyze">
    <span className="material-icons-sharp notranslate">
        analytics
    </span>
    <h3>Analyze</h3>
</a>
<a href="\live">
    <span className="material-icons-sharp notranslate">
        videocam
    </span>
    <h3>Live</h3>
</a>
<a href="/sketch">
    <span className="material-icons-sharp notranslate">
        brush
    </span>
    <h3>Sketch</h3>
</a>

<a href="/records">
    <span className="material-icons-sharp notranslate">
        gavel
    </span>
    <h3>Criminal</h3>
</a>
<a href="/records">
    <span className="material-icons-sharp notranslate">
        search
    </span>
    <h3>Missing</h3>
</a>

                
                
            
                <a href="/">
                    <span className="material-icons-sharp notranslate">
                        logout
                    </span>
                    <h3>Logout</h3>
                </a>
            </div>
        </aside>
        <main>
            <br/>
            <br/>
            <br/>
            <h1>Services</h1>
            <br/>
            <br/>
            
            <div className="analyse">
                <div className="visits">
                    <div className="status">
                        <div className="info">
                            <a href="/analyze">
                            <i className="fa-solid fa-video fa-2xl"></i>
                            <h1>Live CCTV Detection</h1>
                                                            
                        </a>
                        </div>
                    </div>
                </div>
                <div className="sales">
                    <div className="status">
                        <div className="info" >
                            <a href="/live">
                                 <i className="fa-solid fa-magnifying-glass fa-2xl"></i>
                                <h1>CCTV Footage Analyzer</h1>
                            </a>
                            
                        </div>
                    </div>
                </div>
                <div className="visits">
                    <div className="status">
                        <div className="info">
                            <a href="/sketch">
                            <i className="fa-solid fa-image fa-2xl"></i>
                            <h1>Sketch to Image</h1>
                            </a>
                        </div>
                    </div>
                </div>
                
                <div className="searches">
                    <div className="status">
                        <div className="info">
                            <a href="/records">
                            <i className="fa-solid fa-book fa-2xl"></i>
                            <h1>Criminal Records</h1>
                        </a>
                        </div>
                    </div>
                </div>
                <div className="visits">
                    <div className="status">
                        <div className="info">
                            <a href="/records">
                            <i className="fa-solid fa-user-plus fa-2xl"></i>
                            <h1>Missing Records</h1>
                        </a>
                        </div>
                    </div>
                </div>
              
            </div>
            

        </main>
        <div className="right-section">
            
            <div className="nav">

                <div className="profile flex items-center gap-1 p-0.01 rounded-lg shadow-sm ">
        <UserCircle className="w-8 h-8 text-blue-600" />
        <span className="font-medium  notranslate">{stationName}</span>
      </div>
                <button id="menu-btn">
                    <span className="material-icons-sharp">
                        menu
                    </span>
                </button>
                
                <div className="dark-mode">
                    <span className="material-icons-sharp active notranslate">
                        light_mode
                    </span>
                    <span className="material-icons-sharp notranslate" >
                        dark_mode
                    </span>
                </div>

           
            </div>


           

            <div className="reminders">
                <div className="reminder">
                    <h2>History</h2>
                    <span className="material-icons-sharp notranslate">
                        notifications_none
                    </span>
                </div>

                <div className="notification">
                    <div className="icon">
  <span className="material-icons-sharp">
    person
  </span>
</div>

                    <div className="content">
                        <div className="info">
                            <h3>Suspect "Rajan Poudel"</h3>
                            <small className="text_muted">
                                Detected at CCTV 01 at Baneshwor 
                            </small>
                        </div>
                        <a href="/live">
                        <span className="material-icons-sharp">
                            more_vert
                        </span>
                        </a>
                    </div>
                </div>

                <div className="notification">
                    <div className="icon">
  <span className="material-icons-sharp">
    person
  </span>
</div>

                    <div className="content">
                        <div className="info">
                            <h3>Suspect "Sourya Poudel"</h3>
                            <small className="text_muted">
                                Detected at CCTV 05 at Putalisadak 
                            </small>
                        </div>
                        <a href="/live">
                        <span className="material-icons-sharp">
                            more_vert
                        </span>
                        </a>
                    </div>
                </div>

               


    

            </div>

        </div>


    </div>
    
    </>
  );
}

export default HomePage;
