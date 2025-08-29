import React, { useState } from "react";
import { useLocation } from "react-router-dom";

function QuestionGenerator() {
  const location = useLocation();
  const { board, grade, category, topic } = location.state || {};

  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  const generateQuestions = async () => {
    if (!topic) {
      setOutput("No topic selected from Subjects page!");
      return;
    }

    setLoading(true);
    setOutput("");

    try {
      const response = await fetch("https://9a58077cf6b3.ngrok-free.app/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: `Generate 12 sample questions: 4 basic, 4 moderate, 4 advanced on ${topic}`
  }),
});



      const data = await response.json();
      if (data.output) setOutput(data.output);
      else if (data.error) setOutput("Error: " + data.error);
    } catch (err) {
      setOutput("Error: " + err.message);
    }

    setLoading(false);
  };

  return (
    <div style={{ maxWidth: "800px", margin: "20px auto", padding: "10px" }}>
      <h2>AI Question Generator</h2>
      <p><strong>Topic:</strong> {topic || "No topic selected"}</p>
      <button onClick={generateQuestions} style={{ marginTop: "10px" }}>
        {loading ? "Generating..." : "Generate Questions"}
      </button>
      <pre
        style={{
          marginTop: "20px",
          background: "#f0f0f0",
          padding: "15px",
          whiteSpace: "pre-wrap",
        }}
      >
        {output}
      </pre>
    </div>
  );
}

export default QuestionGenerator;
