import "aos/dist/aos.css";
import "./landing.css";
import { Helmet } from "react-helmet";
import React, { useEffect, useState } from "react";

const Landing: React.FC = () => {
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
        <meta charSet="UTF-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" type="image/svg+xml" href="icon-white.png" />
        <title>EagleEye - AI Crime Detection</title>
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.3.0/css/all.min.css"
        />
        <link rel="stylesheet" href="https://unpkg.com/aos@next/dist/aos.css" />
        <link rel="stylesheet" href="style.css" />
      </Helmet>

      <header className="header">
        <a href="#" className="logo">
          <img
            src="icon-white.png"
            alt="EagleEye Icon"
            style={{ height: "1em", verticalAlign: "middle" }}
          />
          EagleEye
        </a>
        <nav className="navbar">
          <a href="#home">Home</a>
          <a href="#services">Services</a>
          <a href="#about">About</a>
          <a href="#Developers">Developers</a>
          <a href="#review">Review</a>
        </nav>
        <div id="menu-btn" className="fas fa-bars"></div>
      </header>

      <section className="home" id="home">
        <div className="image">
          <img src="img/eagleeye-hero.png" alt="Hero Image" />
        </div>
        <div className="content">
          <h3>AI Crime Detection</h3>
          <p>
            AI-powered criminal detection with real-time surveillance, instant alerts,
            and accurate suspect identification for safer communities.
          </p>
          <a href="#" className="btn" >
            Get Started <span className="fas fa-chevron-right"></span>
          </a>
        </div>
      </section>

      <section
        className="icons-container"
        data-aos="fade-up"
        data-aos-easing="ease-in-out"
        data-aos-duration={1000}
      >
        <div className="icons">
          <i className="fas fa-shield-halved"></i>
          <h3>240+</h3>
          <p>Cases Solved</p>
        </div>
        <div className="icons">
          <i className="fas fa-landmark"></i>
          <h3>20+</h3>
          <p>Police Stations</p>
        </div>
        <div className="icons">
          <i className="fas fa-eye"></i>
          <h3>90%+</h3>
          <p>Accuracy in Detection</p>
        </div>
        <div className="icons">
          <i className="fas fa-clock"></i>
          <h3>24/7</h3>
          <p>Real-Time Monitoring</p>
        </div>
      </section>

      <section
        className="services"
        id="services"
        data-aos="fade-up"
        data-aos-easing="ease-in-out"
        data-aos-duration={1000}
      >
        <h1 className="heading">
          Our <span>Services</span>
        </h1>
        <div className="box-container">
          <div className="box">
            <i className="fas fa-laptop-code"></i>
            <h3>CCTV Detection</h3>
            <p>24/7 live CCTV detection ensures instant alerts when it matters most.</p>
            <a href="#" className="btn feature-link" >
              Learn More <span className="fas fa-chevron-right"></span>
            </a>
          </div>
          <div className="box">
            <i className="fas fa-pen-to-square"></i>
            <h3>Sketch to Image</h3>
            <p>Transform rough sketches into lifelike images instantly.</p>
            <br />
            <br />
            <a href="#" className="btn feature-link" >
              Learn More <span className="fas fa-chevron-right"></span>
            </a>
          </div>
          <div className="box">
            <i className="fas fa-user-slash"></i>
            <h3>Criminal Records</h3>
            <p>Centralized records for faster investigations.</p>
            <br />
            <br />
            <a href="#" className="btn feature-link" >
              Learn More <span className="fas fa-chevron-right"></span>
            </a>
          </div>
          <div className="box">
            <i className="fas fa-camera"></i>
            <h3>Video Analyzer</h3>
            <p>Analyze hours of footage in minutes for actionable insights.</p>
            <br />
            <br />
            <a href="#" className="btn feature-link" >
              Learn More <span className="fas fa-chevron-right"></span>
            </a>
          </div>
        </div>
      </section>

      <section
        className="about"
        id="about"
        data-aos="fade-up"
        data-aos-easing="ease-in-out"
        data-aos-duration={1000}
      >
        <h1 className="heading">
          About <span>Us</span>
        </h1>
        <div className="row">
          <div className="image">
            <img
              src="img/eagleeye-hero2.svg"
              alt=""
              style={{
                transform: "scale(1.5)",
                transformOrigin: "top left",
                position: "relative",
                top: "-30px",
              }}
            />
          </div>
          <div className="content">
            <h3>Your safety, powered by AI.</h3>
            <p>
              We are dedicated to creating safer communities with real-time AI
              surveillance and intelligent image processing.
            </p>
            <p>
              From CCTV analysis to sketch-to-photo conversion, EagleEye delivers
              precise insights for faster investigations.
            </p>
            <a href="#" className="btn feature-link" >
              Learn More <span className="fas fa-chevron-right"></span>
            </a>
          </div>
        </div>
      </section>

      <section
        className="devs"
        id="Developers"
        data-aos="fade-up"
        data-aos-easing="ease-in-out"
        data-aos-duration={1000}
      >
        <h1 className="heading">
          our <span>team</span>
        </h1>
        <br />
        <br />
        <br />
        <div
          className="box-container"
          data-aos="fade-up"
          data-aos-easing="ease-in-out"
          data-aos-duration={1000}
        >
          <div
            className="box"
            data-aos="fade-up"
            data-aos-easing="ease-in-out"
            data-aos-duration={1000}
          >
            <img src="img/sourya.png" alt="" />
            <h3>Sourya Poudel</h3>
            <span>Developer & Team Leader</span>
            <div className="share">
              <a
                href="https://www.facebook.com/sourya.poudel.569017"
                className="fab fa-facebook-f"
              ></a>
              <a href="https://www.github.com/sourya-poudel" className="fab fa-github"></a>
              <a
                href="https://www.instagram.com/sourya_poudel_/"
                className="fab fa-instagram"
              ></a>
              <a
                href="https://www.linkedin.com/in/sourya-poudel-451b25219/"
                className="fab fa-linkedin"
              ></a>
            </div>
          </div>

          {/* Repeat for other team members (Rajan, Amir, Aayush) with same <br /> and inline style fixes */}
        </div>
      </section>

      <section
        className="review"
        id="review"
        data-aos="fade-up"
        data-aos-easing="ease-in-out"
        data-aos-duration={1000}
      >
        <h1 className="heading">
          client's <span>review</span>
        </h1>
        <div
          className="box-container"
          data-aos="fade-up"
          data-aos-easing="ease-in-out"
          data-aos-duration={1000}
        >
          <div className="box">
            <img src="img/review2.png" alt="" />
            <h3>
              Inspector R. Mehta <br />
              <span style={{ fontSize: "0.65em", fontWeight: 300 }}>
                City Police Department
              </span>
            </h3>
            <br />
            <div className="stars">
              <i className="fas fa-star" style={{ color: "gold" }}></i>
              <i className="fas fa-star" style={{ color: "gold" }}></i>
              <i className="fas fa-star" style={{ color: "gold" }}></i>
              <i className="fas fa-star" style={{ color: "gold" }}></i>
              <i className="fas fa-star" style={{ color: "gold" }}></i>
            </div>
            <p className="text">
              EagleEye has completely transformed how we investigate cases. What
              used to take days of manually reviewing CCTV footage now takes just
              a few minutes. The AI-powered alerts help us act faster and keep our
              communities safer.
            </p>
          </div>

          {/* Repeat for other reviews with same <br /> and inline style fixes */}
        </div>
      </section>

      <section
        className="footer"
        data-aos="fade-up"
        data-aos-easing="ease-in-out"
        data-aos-duration={1000}
      >
        <div className="box-container">
          <div className="box">
            <h3>Quick Links</h3>
            <a href="#home">
              <i className="fas fa-chevron-right"></i> Home
            </a>
            <a href="#services">
              <i className="fas fa-chevron-right"></i> Services
            </a>
            <a href="#about">
              <i className="fas fa-chevron-right"></i> About
            </a>
            <a href="#Developers">
              <i className="fas fa-chevron-right"></i> Developers
            </a>
            <a href="#review">
              <i className="fas fa-chevron-right"></i> Review
            </a>
          </div>
          <div className="box">
            <h3>Our Services</h3>
            <a href="#services">
              <i className="fas fa-chevron-right"></i> CCTV Detection
            </a>
            <a href="#services">
              <i className="fas fa-chevron-right"></i> Sketch to Image
            </a>
            <a href="#services">
              <i className="fas fa-chevron-right"></i> Criminal Records
            </a>
            <a href="#services">
              <i className="fas fa-chevron-right"></i> Video Analyzer
            </a>
          </div>
          <div className="box">
           <h3>Contact Info</h3>
          <a href="tel:9766280072"><i className="fas fa-phone"></i> +977 976-6280072</a>
          <a href="tel:9802922270"><i className="fas fa-phone"></i> +977 980-2922270</a>
          <a href="mailto:souryapoudel.np@gmail.com"><i className="fas fa-envelope"></i> souryapoudel.np@gmail.com</a>
          <a href="mailto:rajanpoudel.np@gmail.com"><i className="fas fa-envelope"></i> rajanpoudel.np@gmail.com</a>
          <a href="#"><i className="fas fa-map-marker-alt"></i> Bharatpur - 7, Chitwan</a>
        </div>
      </div>
    </section>
    
    </>
  );
}

export default Landing;
