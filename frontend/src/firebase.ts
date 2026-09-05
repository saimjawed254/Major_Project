import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDbRMOkP4nENBXoa2okGNB3uxtiWtsZeW8",
  authDomain: "major-project-85da6.firebaseapp.com",
  projectId: "major-project-85da6",
  storageBucket: "major-project-85da6.firebasestorage.app",
  messagingSenderId: "317550509815",
  appId: "1:317550509815:web:a16482e8dea151b2951df2",
  measurementId: "G-1LJ09TZPK4"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
