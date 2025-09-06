// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional

const firebaseConfig = {
  apiKey: "AIzaSyBg7JKTpbG19MIQqiNO57PBMLJiEkxnADo",
  authDomain: "eagleeye-e9d6e.firebaseapp.com",
  projectId: "eagleeye-e9d6e",
  storageBucket: "eagleeye-e9d6e.firebasestorage.app",
  messagingSenderId: "464733024753",
  appId: "1:464733024753:web:c0da7050b4979f567dd450",
  measurementId: "G-E142QNY699"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);