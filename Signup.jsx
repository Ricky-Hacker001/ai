import React from "react";
import { useNavigate, Link } from "react-router-dom";
import "./Login.css";
import { FaUser, FaEnvelope, FaLock, FaGoogle } from "react-icons/fa";

function Signup({ onSignupSuccess }) {
  const navigate = useNavigate();

  const handleCreateAccount = () => {
    localStorage.setItem("userToken", "dummy-auth-token");
    onSignupSuccess();
    navigate("/home");
  };

  return (
    <div className="login-page">
      <div className="login-container animate">
        <h2>Create Your Account</h2>
        <p>Join us to start your personalized learning adventure.</p>

        <div className="input-field">
          <FaUser className="icon" />
          <input type="text" placeholder="Full Name" required />
        </div>

        <div className="input-field">
          <FaEnvelope className="icon" />
          <input type="email" placeholder="Email Address" required />
        </div>

        <div className="input-field">
          <FaLock className="icon" />
          <input type="password" placeholder="Password" required />
        </div>

        <button className="btn primary" onClick={handleCreateAccount}>
          Create Account
        </button>

        <div className="divider">or</div>

        <button className="btn google">
          <FaGoogle /> Continue with Google
        </button>

        <p className="signup">
          Already have an account? <Link to="/login">Sign In</Link>
        </p>
      </div>
    </div>
  );
}

export default Signup;
