import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./test.css";

export default function Test() {
  const location = useLocation();
  const navigate = useNavigate();
  const { board, grade, category, topic } = location.state || {};

  // Dummy sample questions (4 basic, 4 moderate, 4 advanced)
  const generateQuestions = () => {
    const basic = Array.from({ length: 4 }, (_, i) => ({
      type: "Basic",
      text: `Basic Question ${i + 1} on ${topic}`
    }));

    const moderate = Array.from({ length: 4 }, (_, i) => ({
      type: "Moderate",
      text: `Moderate Question ${i + 1} on ${topic}`
    }));

    const advanced = Array.from({ length: 4 }, (_, i) => ({
      type: "Advanced",
      text: `Advanced Question ${i + 1} on ${topic}`
    }));

    return [...basic, ...moderate, ...advanced];
  };

  const questions = generateQuestions();

  return (
    <div className="test-container">
      <h1>{topic} - Sample Test</h1>
      <h3>{board} | {grade} | {category}</h3>

      <div className="questions-box">
        {questions.map((q, index) => (
          <div key={index} className={`question-card ${q.type.toLowerCase()}`}>
            <h4>{q.type} Q{index + 1}</h4>
            <p>{q.text}</p>
          </div>
        ))}
      </div>

      <button className="back-btn" onClick={() => navigate(-1)}>⬅ Back</button>
    </div>
  );
}
