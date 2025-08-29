import React, { useState } from 'react';

const AppStyles = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --primary-color: #4f46e5; --primary-hover: #4338ca; --background-color: #f8f9fa;
  --card-background: #ffffff; --text-color: #212529; --subtle-text: #6c757d;
  --border-color: #dee2e6; --error-color: #dc3545; --success-color: #198754;
  --success-bg: #d1e7dd; --error-bg: #f8d7da;
}
body {
  margin: 0; font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale; background-color: var(--background-color); color: var(--text-color);
}
.app-container { max-width: 800px; margin: 2rem auto; padding: 2rem; }
.app-header { text-align: center; margin-bottom: 2.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1.5rem; }
.app-header h1 { font-size: 2.5rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem; }
.app-header p { font-size: 1.1rem; color: var(--subtle-text); }
.app-main { display: flex; flex-direction: column; gap: 2rem; }
.input-section { display: flex; gap: 1rem; align-items: center; }
.topic-input { flex-grow: 1; padding: 0.75rem 1rem; font-size: 1rem; border-radius: 8px; border: 1px solid var(--border-color); transition: border-color 0.2s, box-shadow 0.2s; }
.topic-input:focus { outline: none; border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2); }
.generate-btn { padding: 0.75rem 1.5rem; font-size: 1rem; font-weight: 600; border: none; border-radius: 8px; background-color: var(--primary-color); color: white; cursor: pointer; transition: background-color 0.2s; }
.generate-btn:hover:not(:disabled) { background-color: var(--primary-hover); }
.generate-btn:disabled { background-color: #a5b4fc; cursor: not-allowed; }
.error-message { background-color: var(--error-bg); color: var(--error-color); border: 1px solid #f5c2c7; padding: 1rem; border-radius: 8px; text-align: center; }
.spinner-container { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; gap: 1rem; color: var(--subtle-text); }
.spinner { width: 48px; height: 48px; border: 5px solid var(--primary-color); border-bottom-color: transparent; border-radius: 50%; display: inline-block; animation: rotation 1s linear infinite; }
@keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.results-section h2 { text-align: center; color: var(--text-color); margin-bottom: 2rem; }
.difficulty-group { margin-bottom: 2rem; }
.difficulty-group h3 { font-size: 1.5rem; color: var(--primary-color); border-bottom: 2px solid var(--primary-hover); padding-bottom: 0.5rem; margin-bottom: 1rem; }
.question-card { background-color: var(--card-background); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color); margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.question-text { font-size: 1.1rem; margin: 0 0 1rem 0; }
.options-grid { display: grid; grid-template-columns: 1fr; gap: 0.5rem; }
.mcq-option { padding: 0.75rem 1rem; border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; transition: background-color 0.2s, border-color 0.2s; user-select: none; }
.mcq-option:hover { background-color: #e9ecef; }
.mcq-option.selected { border-color: var(--primary-color); background-color: #e0e7ff; font-weight: 500; }
.mcq-option.correct { background-color: var(--success-bg); border-color: var(--success-color); color: #0f5132; }
.mcq-option.incorrect { background-color: var(--error-bg); border-color: var(--error-color); color: #842029; }
.validation-section { text-align: center; margin-top: 2rem; }
.check-btn { padding: 0.8rem 2rem; font-size: 1.1rem; }
`;

function App() {
  const [topic, setTopic] = useState('');
  const [mcqs, setMcqs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [results, setResults] = useState({});
  const [showResults, setShowResults] = useState(false);

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setIsLoading(true);
    setError(null);
    setMcqs([]);
    setSelectedAnswers({});
    setResults({});
    setShowResults(false);

    try {
      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.mcqs && data.mcqs.length > 0) {
        setMcqs(data.mcqs);
      } else {
        setError(data.error || 'The model did not return any MCQs. Please try a different topic.');
      }
    } catch (e) {
      setError(e.message || 'Failed to connect to the backend.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectAnswer = (questionId, option) => {
    if (showResults) return; // Don't allow changes after checking
    setSelectedAnswers(prev => ({ ...prev, [questionId]: option }));
  };

  const checkAnswers = () => {
    const newResults = {};
    mcqs.forEach(mcq => {
      newResults[mcq.id] = selectedAnswers[mcq.id] === mcq.answer;
    });
    setResults(newResults);
    setShowResults(true);
  };
  
  const groupedMcqs = mcqs.reduce((acc, q) => {
      (acc[q.difficulty] = acc[q.difficulty] || []).push(q);
      return acc;
  }, {});

  return (
    <>
      <style>{AppStyles}</style>
      <div className="app-container">
        <header className="app-header">
          <h1>AI MCQ Generator</h1>
          <p>Enter a topic to generate an interactive multiple-choice quiz.</p>
        </header>
        
        <main className="app-main">
          <div className="input-section">
            <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., 'The causes of World War I'" className="topic-input" disabled={isLoading} />
            <button onClick={handleGenerate} disabled={isLoading} className="generate-btn">
              {isLoading ? 'Generating...' : 'Generate Quiz'}
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}
          {isLoading && <div className="spinner-container"><div className="spinner"></div><p>Generating your quiz...</p></div>}

          {mcqs.length > 0 && (
            <div className="results-section">
              <h2>Quiz on "{topic}"</h2>
              {Object.entries(groupedMcqs).map(([difficulty, qs]) => (
                <div key={difficulty} className="difficulty-group">
                  <h3>{difficulty}</h3>
                  {qs.map((mcq) => (
                    <div key={mcq.id} className="question-card">
                      <p className="question-text"><strong>{mcq.id}:</strong> {mcq.question}</p>
                      <div className="options-grid">
                        {mcq.options.map((option, index) => {
                          const isSelected = selectedAnswers[mcq.id] === option;
                          let optionClass = "mcq-option";
                          if (showResults) {
                            if (option === mcq.answer) optionClass += " correct";
                            else if (isSelected) optionClass += " incorrect";
                          } else if (isSelected) {
                            optionClass += " selected";
                          }
                          return (
                            <div key={index} className={optionClass} onClick={() => handleSelectAnswer(mcq.id, option)}>
                              {String.fromCharCode(65 + index)}) {option}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
              {!showResults && (
                <div className="validation-section">
                  <button onClick={checkAnswers} className="generate-btn check-btn">Check Answers</button>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </>
  );
}

export default App;