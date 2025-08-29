import { useState } from "react";
import { useNavigate } from "react-router-dom"; // Added for navigation
import Particles from "react-tsparticles";
import { loadFull } from "tsparticles";
import "./home.css";

export default function Home() {
  const [loginClicked, setLoginClicked] = useState(false);
  const [selectedBoard, setSelectedBoard] = useState(null);
  const [selectedGrade, setSelectedGrade] = useState("Select Grade");
  const navigate = useNavigate(); // Added navigation hook

  // Particles initialization
  const particlesInit = async (main) => {
    await loadFull(main);
  };

  const handleLogin = () => {
    if (selectedBoard && selectedGrade !== "Select Grade") {
      setLoginClicked(true);
      navigate("/subjects", { state: { board: selectedBoard, grade: selectedGrade } }); // Direct navigation to Subjects
    } else {
      alert("Please select both a board and a grade!");
    }
  };

  const handleLogout = (e) => {
    e.preventDefault();
    window.location.href = "/login"; // Redirect to login page
  };

  return (
    <div className={`home-wrapper ${loginClicked ? "login-active" : ""}`}>
      {/* Background Particles */}
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: { color: "transparent" },
          fpsLimit: 120,
          interactivity: {
            events: {
              onHover: { enable: true, mode: "repulse" },
              resize: true,
            },
            modes: { repulse: { distance: 100, duration: 0.4 } },
          },
          particles: {
            color: { value: ["#00CED1", "#FFA500", "#98FB98"] },
            links: { color: "#ffffff", distance: 150, enable: false, opacity: 0.4, width: 1 },
            collisions: { enable: true },
            move: { direction: "none", enable: true, outModes: "bounce", speed: 1.5, random: true },
            number: { density: { enable: true, area: 800 }, value: 70 },
            opacity: { value: 0.8, random: true },
            shape: { type: "circle" },
            size: { value: { min: 5, max: 15 }, random: true },
          },
          detectRetina: true,
        }}
      />

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">COGNIDIYA CURVE</div>
        <nav className="sidebar-nav">
          <a href="/login" className="logout" onClick={handleLogout}>➜</a>
        </nav>
      </aside>

      {/* Header */}
      <header className="header">
        <div className="header-icons">
          <span className="icon search">🔍</span>
          <span className="icon download">⬇️</span>
          <span className="icon bell">🔔</span>
          <span className="icon settings">⚙️</span>
        </div>
      </header>

      {/* Dashboard Card */}
      <div className="home-container">
        <div className="profile-icon">👤</div>
        <h1>Welcome to Your CogniDiya Curve!</h1>
        <p>You've successfully logged in. Explore your personalized CogniDiya Curve!! and manage your account.</p>
        <div className="dashboard-content">
          <div className="practice-icon">📱</div>
          <h2>Start Your Practice Session</h2>
          <p>Select your board and grade to begin.</p>
          <div className="board-options">
            <div
              className={`board-card ${selectedBoard === "TN Board" ? "selected" : ""}`}
              onClick={() => setSelectedBoard("TN Board")}
            >
              🏛 <span className="board-name">TN Board</span><br /><small>Tamil Nadu State</small>
            </div>
            <div
              className={`board-card ${selectedBoard === "NCERT" ? "selected" : ""}`}
              onClick={() => setSelectedBoard("NCERT")}
            >
              📘 <span className="board-name">NCERT</span><br /><small>National Curriculum</small>
            </div>
            <div
              className={`board-card ${selectedBoard === "ICSE" ? "selected" : ""}`}
              onClick={() => setSelectedBoard("ICSE")}
            >
              📕 <span className="board-name">ICSE</span><br /><small>Indian Certificate</small>
            </div>
          </div>
          <div className="selection-group">
            <select
              value={selectedGrade}
              onChange={(e) => setSelectedGrade(e.target.value)}
              className="custom-dropdown"
            >
              <option value="Select Grade">Select Grade</option>
              <option value="Grade 8">Grade 8</option>
              <option value="Grade 9">Grade 9</option>
              <option value="Grade 10">Grade 10</option>
              <option value="Grade 11">Grade 11</option>
              <option value="Grade 12">Grade 12</option>
            </select>
          </div>
          <button className="btn" onClick={handleLogin}>
            Let's Go
          </button>
        </div>
      </div>
    </div>
  );
}