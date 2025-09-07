import "./landing.css";
import { Helmet } from "react-helmet";
import React, { useEffect } from "react";

const Landing: React.FC = () => {

  useEffect(() => {
  }, []);

  return (
    <>
      <Helmet>
         <meta charSet="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" type="image/svg+xml" href="img/icon-white.png" />
    <title>EagleEye - AI Crime Detection</title>
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.3.0/css/all.min.css"
    />
    <link rel="stylesheet" href="https://unpkg.com/aos@next/dist/aos.css" />
    
      </Helmet>


    <header className="header">
      <a href="/" className="logo">
        <img src="img/icon-white.png" alt="EagleEye Icon" className="class1" />
        <span className="Padding">EagleEye</span>
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
        <a href="/login" className="btn" >
          Get Started <span className="fas fa-chevron-right"></span>
        </a>
      </div>
    </section>

    <section className="icons-container" >
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

<section className="services" id="services" >
      <h1 className="heading">Our <span>Services</span></h1>
      <div className="box-container">
        <div className="box">
          <i className="fas fa-laptop-code"></i>
          <h3>CCTV Detection</h3>
          <p>24/7 live CCTV detection ensures instant alerts when it matters most.</p>
          <a href="/login" className="btn feature-link" >Learn More <span className="fas fa-chevron-right"></span></a>
        </div>
        <div className="box">
          <i className="fas fa-pen-to-square"></i>
          <h3>Sketch to Image</h3>
          <p>Transform rough sketches into lifelike images instantly.</p><br/><br/>
          <a href="/login" className="btn feature-link" >Learn More <span className="fas fa-chevron-right"></span></a>
        </div>
        <div className="box">
          <i className="fas fa-user-slash"></i>
          <h3>Criminal Records</h3>
          <p>Centralized records for faster investigations.</p><br/><br/>
          <a href="/login" className="btn feature-link" >Learn More <span className="fas fa-chevron-right"></span></a>
        </div>
        <div className="box">
          <i className="fas fa-camera"></i>
          <h3>Video Analyzer</h3>
          <p>Analyze hours of footage in minutes for actionable insights.</p><br/><br/>
          <a href="/login" className="btn feature-link" >Learn More <span className="fas fa-chevron-right"></span></a>
        </div>
      </div>
    </section>
    <section className="about" id="about" >
      <h1 className="heading">About <span>Us</span></h1>
      <div className="row">
        <div className="image">
          <img src="img/eagleeye-hero2.svg" alt="" className="class2" />
        </div>
        <div className="content">
          <h3>Your safety, powered by AI.</h3>
          <p>We are dedicated to creating safer communities with real-time AI surveillance and intelligent image processing.</p>
          <p>From CCTV analysis to sketch-to-photo conversion, EagleEye delivers precise insights for faster investigations.</p>
          <a href="/login" className="btn feature-link" >Learn More <span className="fas fa-chevron-right"></span></a>
        </div>
      </div>
    </section>

    <section className="devs" id="Developers" >

      <h1 className="heading">our <span>team</span></h1>
            
      <div className="box-container" >
        <div className="box" >
          <img src="img/sourya.png" alt="" />
          <h3>Sourya Poudel</h3>
          <span>Developer & Team Leader</span>
          <div className="share">
            <a href="https://www.facebook.com/sourya.poudel.569017" className="fab fa-facebook-f"></a>
            <a href="https://www.github.com/sourya-poudel" className="fab fa-github"></a>
            <a href="https://www.instagram.com/sourya_poudel_/" className="fab fa-instagram"></a>
            <a href="https://www.linkedin.com/in/sourya-poudel-451b25219/" className="fab fa-linkedin"></a>
          </div>
        </div>

        <div className="box" >
          <img src="img/rajan.png" alt="" />
          <h3>Rajan Poudel</h3>
          <span>Developer</span>
          <div className="share">
            <a href="https://www.facebook.com/rajan.poudel.626944" className="fab fa-facebook-f"></a>
            <a href="https://www.github.com/rajan-poudel" className="fab fa-github"></a>
            <a href="https://www.instagram.com/_rajanpoudel" className="fab fa-instagram"></a>
            <a href="https://www.linkedin.com/in/rajanpoudel-np/" className="fab fa-linkedin"></a>
          </div>
        </div>

        <div className="box" >
          <img src="img/barun.png" alt="" />
          <h3> Amir Jung KC</h3>
          <span>Developer</span>
          <div className="share">
            <a href="https://www.facebook.com/barun.kc.3150" className="fab fa-facebook-f"></a>
            <a href="https://www.github.com/barunkc" className="fab fa-github"></a>
            <a href="https://www.google.com" className="fab fa-instagram"></a>
            <a href="https://www.linkedin.com/in/barun-kc-928aab382/" className="fab fa-linkedin"></a>
          </div>
        </div>

        <div className="box"  >
          <img src="img/aayush.png" alt="" />
          <h3>Aayush Kandel</h3>
          <span>Developer</span>
          <div className="share">
            <a href="https://www.facebook.com/aayush.kandel.21" className="fab fa-facebook-f"></a>
            <a href="" className="fab fa-github"></a>
            <a href="https://www.instagram.com/aayu_shkandel7/" className="fab fa-instagram"></a>
            <a href="https://www.linkedin.com/in/aayush-kandel-758520376/" className="fab fa-linkedin"></a>
          </div>
        </div>
      </div>
    </section>

    <section className="review" id="review" >
      <h1 className="heading">client's <span>review</span></h1>
      <div className="box-container" >
        <div className="box">
          <img src="img/review2.png" alt="" />
          <h3>
            Inspector B. Adhikari <br/>
            <span className="class3">City Police Department</span>
          </h3>
          <br/>
          <div className="stars">
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
          </div>
          <p className="text">
            EagleEye has completely transformed how we investigate cases. What used to take days of manually reviewing CCTV footage now takes just a few minutes. The AI-powered alerts help us act faster and keep our communities safer.
          </p>
        </div>

        <div className="box">
          <img src="img/review1.png" alt="" />
          <h3>Superintendent A. Khan<br/><span className="class5" >State Crime Branch</span></h3><br/>
          <div className="stars">
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star-half-alt class4"></i>
          </div>
          <p className="text">
          The sketch-to-photo conversion tool has been a game-changer for us. Even with rough sketches, we’ve been able to identify suspects accurately and connect them to real cases. This technology gives us an edge we never had before. </p>
        </div>

        <div className="box">
          <img src="img/review3.png" alt="" />
          <h3>Superintendent S. Poudel<br/><span className="class6">Metropolitan Police</span></h3><br/>
          <div className="stars">
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
            <i className="fas fa-star class4"></i>
          </div>
          <p className="text">
            We rely on EagleEye for both live monitoring and historical analysis. The timestamped clips and detailed reports save hours of effort during investigations. It’s like having an extra officer who never gets tired.
          </p>
        </div>
      </div>
    </section>

    <section className="footer" >
      <div className="box-container">
        <div className="box">
          <h3>Quick Links</h3>
          <a href="#home"><i className="fas fa-chevron-right"></i> Home</a>
          <a href="#services"><i className="fas fa-chevron-right"></i> Services</a>
          <a href="#about"><i className="fas fa-chevron-right"></i> About</a>
          <a href="#Developers"><i className="fas fa-chevron-right"></i> Developers</a>
          <a href="#review"><i className="fas fa-chevron-right"></i> Review</a>
        </div>
        <div className="box">
          <h3>Our Services</h3>
          <a href="#services"><i className="fas fa-chevron-right"></i> CCTV Detection</a>
          <a href="#services"><i className="fas fa-chevron-right"></i> Sketch to Image</a>
          <a href="#services"><i className="fas fa-chevron-right"></i> Criminal Records</a>
          <a href="#services"><i className="fas fa-chevron-right"></i> Video Analyzer</a>
        </div>
        <div className="box">
          <h3>Contact Info</h3>
          <a href="tel:9766280072"><i className="fas fa-phone"></i> +977 976-6280072</a>
          <a href="tel:9802922270"><i className="fas fa-phone"></i> +977 980-2922270</a>
          <a href="mailto:souryapoudel.np@gmail.com"><i className="fas fa-envelope"></i> souryapoudel.np@gmail.com</a>
          <a href="mailto:rajanpoudel.np@gmail.com"><i className="fas fa-envelope"></i> rajanpoudel.np@gmail.com</a>
          <a href="https://www.google.com/maps/place/EagleEye/data=!4m2!3m1!1s0x0:0x755956bb2c47c63?sa=X&ved=1t:2428&hl=en&gl=np&ictx=111"><i className="fas fa-map-marker-alt"></i> Bharatpur - 7, Chitwan</a>
        </div>
      </div>
    </section>
    
    
    </>
  );
}

export default Landing;

