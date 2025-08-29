import React from 'react'
import ReactDOM from 'react-dom/client'
import App from '../App.jsx' // Assuming App.jsx is in the root with package.json

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
