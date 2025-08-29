import React, { useState } from 'react';

// All CSS styles are now embedded directly into the component to resolve the import error.
const AppStyles = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary-color: #4f46e5;
  --primary-hover: #4338ca;
  --background-color: #f8f9fa;
  --card-background: #ffffff;
  --text-color: #212529;
  --subtle-text: #6c757d;
  --border-color: #dee2e6;
  --error-color: #dc3545;
  --success-color: #198754;
}

body {
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: var(--background-color);
  color: var(--text-color);
}

.app-container {
  max-width: 800px;
  margin: 2rem auto;
  padding: 2rem;
}

.app-header {
  text-align: center;
  margin-bottom: 2.5rem;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 1.5rem;
}

.app-header h1 {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 0.5rem;
}

.app-header p {
  font-size: 1.1rem;
  color: var(--subtle-text);
}

.app-main {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.input-section {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.topic-input {
  flex-grow: 1;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.topic-input:focus {
  outline: none;
  border-color: var(--primary-color);
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
}

.generate-btn {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  background-color: var(--primary-color);
  color: white;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
}

.generate-btn:hover:not(:disabled) {
  background-color: var(--primary-hover);
}

.generate-btn:disabled {
  background-color: #a5b4fc;
  cursor: not-allowed;
}

.error-message {
  background-color: #f8d7da;
  color: var(--error-color);
  border: 1px solid #f5c2c7;
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
}

.spinner-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  gap: 1rem;
  color: var(--subtle-text);
}

.spinner {
  width: 48px;
  height: 48px;
  border: 5px solid var(--primary-color);
  border-bottom-color: transparent;
  border-radius: 50%;
  display: inline-block;
  box-sizing: border-box;
  animation: rotation 1s linear infinite;
}

@keyframes rotation {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.results-section h2 {
    text-align: center;
    color: var(--text-color);
    margin-bottom: 2rem;
}

.difficulty-group {
    margin-bottom: 2rem;
}

.difficulty-group h3 {
    font-size: 1.5rem;
    color: var(--primary-color);
    border-bottom: 2px solid var(--primary-hover);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.question-card {
  background-color: var(--card-background);
  padding: 1.5rem;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  margin-bottom: 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  transition: box-shadow 0.2s;
}

.question-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.question-text {
  font-size: 1.1rem;
  margin: 0 0 0.75rem 0;
}

.answer-text {
  font-size: 1rem;
  color: var(--subtle-text);
  background-color: var(--background-color);
  padding: 0.5rem 1rem;
  border-radius: 6px;
  margin: 0;
}
`;

function App() {
  const [topic, setTopic] = useState('');
  const [questions, setQuestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!topic.trim()) {
      setError('Please enter a topic.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setQuestions([]);

    try {
      // Your Flask backend is running on http://localhost:5000
      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topic }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.questions && data.questions.length > 0) {
        setQuestions(data.questions);
      } else {
        setError('The model did not return any questions. Try a different or broader topic.');
      }

    } catch (e) {
      console.error("Fetch error:", e);
      setError(e.message || 'Failed to connect to the backend server.');
    } finally {
      setIsLoading(false);
    }
  };

  const groupQuestionsByDifficulty = () => {
    return questions.reduce((acc, q) => {
      (acc[q.difficulty] = acc[q.difficulty] || []).push(q);
      return acc;
    }, {});
  };

  const groupedQuestions = groupQuestionsByDifficulty();

  return (
    <>
      <style>{AppStyles}</style>
      <div className="app-container">
        <header className="app-header">
          <h1>AI Question Generator</h1>
          <p>Enter a topic from your uploaded PDF to generate exam-style questions.</p>
        </header>
        
        <main className="app-main">
          <div className="input-section">
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., 'The causes of World War I'"
              className="topic-input"
              disabled={isLoading}
            />
            <button onClick={handleGenerate} disabled={isLoading} className="generate-btn">
              {isLoading ? 'Generating...' : 'Generate Questions'}
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}

          {isLoading && (
              <div className="spinner-container">
                  <div className="spinner"></div>
                  <p>Retrieving context and generating questions...</p>
              </div>
          )}

          {questions.length > 0 && (
            <div className="results-section">
              <h2>Generated Questions for "{topic}"</h2>
              {Object.entries(groupedQuestions).map(([difficulty, qs]) => (
                <div key={difficulty} className="difficulty-group">
                  <h3>{difficulty}</h3>
                  {qs.map((q) => (
                    <div key={q.id} className="question-card">
                      <p className="question-text"><strong>{q.id}:</strong> {q.question}</p>
                      <p className="answer-text"><strong>Answer:</strong> {q.answer}</p>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </>
  );
}

export default App;

