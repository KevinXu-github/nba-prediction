// src/components/Footer.js
import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-6">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">&copy; {new Date().getFullYear()} NBA Parlay Predictor</p>
            <p className="text-xs mt-1">
              This is a prediction tool for entertainment purposes only. Please bet responsibly.
            </p>
          </div>
          <div className="flex space-x-4">
            <a href="#" className="text-sm hover:text-blue-400">Terms of Service</a>
            <a href="#" className="text-sm hover:text-blue-400">Privacy Policy</a>
            <a href="#" className="text-sm hover:text-blue-400">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;