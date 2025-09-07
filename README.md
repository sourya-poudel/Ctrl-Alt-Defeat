# 🚨 Eagle Eye

**Project for CyberUtsav 2.0 – The Ultimate 12-Hour Hackathon Experience**  
**Team Members:** Soruya Poudel, Rajan Poudel, Aayush Kandel, Amir Jung KC  

---

## 🔍 Overview  
Eagle Eye is an **AI-powered surveillance and investigation assistant** that helps law enforcement agencies detect and identify suspects from multiple sources in real-time.  

With recent security concerns in Nepal — such as the **Simara Jail conflict where inmates temporarily took control of the jail** — efficient monitoring tools are urgently needed. Our solution is designed to support **Nepal Police and other security agencies** in preventing, investigating, and managing crimes.  

---

## 🎯 Key Features  
- **Live CCTV Monitoring** – Detects suspicious individuals from live surveillance footage.  
- **Video Analysis** – Processes existing video files to extract and identify suspects.  
- **Sketch-to-Image Matching** – Matches hand-drawn or digital sketches against a suspect database.  
- **Suspect Database Integration** – Cross-checks detected individuals with pre-existing police records.  
- **Real-Time Alerts** – Sends notifications when suspects are identified.  
- **Scalable System** – Can be deployed in prisons, airports, border checkpoints, and public spaces.  

---

### Backend / AI
- **Deep Learning & Computer Vision:** PyTorch (`torch`), TorchVision (`torchvision`), OpenCV (`opencv-python`)  
- **Data Processing & Utilities:** NumPy (`numpy`), Pillow (`pillow`), Matplotlib (`matplotlib`), tqdm (`tqdm`)  
- **API / Server:** FastAPI (`fastapi`), Uvicorn (`uvicorn`), Python Multipart (`python-multipart`)  
- **Vector Search / Similarity Matching:** FAISS (`faiss-cpu`)  
- **Cloud / Database Integration:** Firebase Admin SDK (`firebase-admin`)  
- **ONNX Inference:** ONNX Runtime (`onnxruntime`)  

---

## 🚀 Getting Started  

### 🔧 Backend Setup  
1. Clone the project:  
   ```bash
   git clone https://github.com/sourya-poudel/Ctrl-Alt-Defeat
2. Train the pix2pix model:
Train your own model [here](https://github.com/sourya-poudel/Ctrl-Alt-Defeat/blob/main/backend/sketchtoimg-modeltrainer.py)
**Copy and paste the file to backend folder**

4. Inlcude the .json file from google firebase in backend folder
   
5. Backend setup:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn api:app --reload

6. Frontend setup:
   ```bash
   npm install

   npm run dev


