import React from "react";
import { useNavigate, Link } from "react-router-dom";
import "./Login.css";
import { FaEnvelope, FaLock, FaGoogle } from "react-icons/fa";

function Login({ onLoginSuccess }) {
  const navigate = useNavigate();

  const handleSignIn = () => {
    localStorage.setItem("userToken", "dummy-auth-token");
    onLoginSuccess();
    navigate("/home");
  };

  return (
    <div className="login-page">
      <div className="login-container animate">
        <h2>Welcome Back!</h2>
        <p>Sign in to continue your learning journey.</p>

        <div className="input-field">
          <FaEnvelope className="icon" />
          <input type="email" placeholder="Email Address" required />
        </div>

        <div className="input-field">
          <FaLock className="icon" />
          <input type="password" placeholder="Password" required />
        </div>

        <a href="#" className="forgot">Forgot Password?</a>

        <button className="btn primary" onClick={handleSignIn}>
          Sign In
        </button>

        <div className="divider">or</div>

        <button className="btn google">
          <FaGoogle /> Continue with Google
        </button>

        <p className="signup">
          Don’t have an account? <Link to="/signup">Sign Up</Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
