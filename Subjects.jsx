import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./Subjects.css";

export default function Subjects() {
  const location = useLocation();
  const navigate = useNavigate();

  const { board = "TN Board", grade = "Select Grade" } = location.state || {};

  // Define subjects by grade
  const subjectsByGrade = {
    "Grade 8": ["Tamil", "English", "Math", "Science", "Social Science"],
    "Grade 9": ["Tamil", "English", "Math", "Science", "Social Science"],
    "Grade 10": ["Tamil", "English", "Math", "Science", "Social Science"],
    "Grade 11": [
      "Tamil", "English", "Math", "Physics", "Chemistry",
      "Botany", "Zoology", "Computer Science", "Commerce",
      "Economy", "Accounts", "Computer Application"
    ],
    "Grade 12": [
      "Tamil", "English", "Math", "Physics", "Chemistry",
      "Botany", "Zoology", "Computer Science", "Commerce",
      "Economy", "Accounts", "Computer Application"
    ],
  };

  const subjects = subjectsByGrade[grade] || [];

  // Social Science topics for Grade 10 (TN Board)
  const socialScienceContents =
    grade === "Grade 10" && board === "TN Board"
      ? {
          History: [
            "Outbreak of World War I and Its Aftermath",
            "The World between two World Wars",
            "World War II",
            "The World after World War II",
            "Social and Religious Reform Movements in the 19th Century",
            "Early Revolts against British Rule in Tamil Nadu",
          ],
          Geography: [
            "India - Location, Relief and Drainage",
            "Climate and Natural Vegetation of India",
            "Indian Agriculture",
            "India - Resources and Industries",
          ],
          Civics: [
            "Indian Constitution",
            "Central Government",
            "State Government",
          ],
          Economics: [
            "Gross Domestic Product and its Growth",
            "Globalization and Trade",
            "Food Security and Nutrition",
          ],
        }
      : {};

  const [selectedSubject, setSelectedSubject] = useState(null);

  // Handle subject selection
  const handleSubjectClick = (subject) => {
    setSelectedSubject((prev) => (prev === subject ? null : subject));
  };

  // Handle topic click → navigate to test page
  const handleContentClick = (category, topic) => {
    navigate(
      `/test?board=${board}&grade=${grade}&category=${category}&topic=${encodeURIComponent(
        topic
      )}`
    );
  };

  // Handle topic click → navigate to AI generator
  const handleGenerateQuestions = (category, topic) => {
    navigate("/generate-questions", {
      state: { board, grade, category, topic },
    });
  };

  return (
    <div className="subjects-container">
      <h1>
        Subjects for {board} - {grade}
      </h1>
      <p>Select a subject to view contents:</p>

      {/* Subject List */}
      <div className="subjects-box">
        {subjects.map((subject, index) => (
          <div
            key={index}
            className={`subject-item ${
              selectedSubject === subject ? "selected" : ""
            }`}
            onClick={() => handleSubjectClick(subject)}
          >
            {subject}
            {subject === "Social Science" &&
              grade === "Grade 10" &&
              board === "TN Board" && (
                <span className="focus-badge">Full Accuracy Focus!</span>
              )}
          </div>
        ))}
      </div>

      {/* Social Science Contents (only for Grade 10 - TN Board) */}
      {selectedSubject === "Social Science" &&
        grade === "Grade 10" &&
        board === "TN Board" && (
          <div className="content-box">
            <h3>Select a Category and Topic</h3>
            {Object.entries(socialScienceContents).map(([category, topics]) => (
              <div key={category} className="content-category">
                <h4>{category}</h4>
                <ul>
                  {topics.map((topic, idx) => (
                    <li key={idx} className="content-topic">
                      <span
                        onClick={() => handleContentClick(category, topic)}
                      >
                        {topic}
                      </span>
                      {/* Add Generate Questions Button */}
                      <button
                        className="generate-btn"
                        onClick={() => handleGenerateQuestions(category, topic)}
                        style={{ marginLeft: "10px" }}
                      >
                        Generate Sample Questions
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
    </div>
  );
}
