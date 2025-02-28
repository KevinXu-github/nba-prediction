// src/pages/About.js
import React from 'react';

const About = () => {
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">About NBA Parlay Predictor</h1>
      
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-bold mb-4">Our System</h2>
        
        <p className="mb-4">
          The NBA Parlay Predictor uses advanced machine learning algorithms to analyze NBA games and predict the most profitable parlay bets, focusing on over/under predictions.
        </p>
        
        <p className="mb-4">
          Our AI-driven approach combines multiple data sources to provide you with the highest-confidence picks for your parlay bets:
        </p>
        
        <ul className="list-disc ml-6 mb-4 space-y-2">
          <li>Team performance statistics and trends</li>
          <li>Player injuries and lineup changes</li>
          <li>Historical betting patterns and odds movements</li>
          <li>Home/away performance differentials</li>
          <li>Game pace and scoring trends</li>
        </ul>
        
        <p>
          We continuously improve our prediction models based on actual game outcomes and betting results.
        </p>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-bold mb-4">How to Use</h2>
        
        <ol className="list-decimal ml-6 space-y-3">
          <li>
            <p className="font-medium">Generate a Parlay</p>
            <p className="text-sm text-gray-700">
              Navigate to the "Generate Parlay" page and select your preferred parlay size and minimum confidence threshold.
            </p>
          </li>
          <li>
            <p className="font-medium">Review Predictions</p>
            <p className="text-sm text-gray-700">
              Our system will provide you with the optimal parlay selections, including confidence scores and risk levels for each pick.
            </p>
          </li>
          <li>
            <p className="font-medium">View Upcoming Games</p>
            <p className="text-sm text-gray-700">
              Explore all upcoming NBA games with detailed team statistics and betting lines.
            </p>
          </li>
          <li>
            <p className="font-medium">Track Your History</p>
            <p className="text-sm text-gray-700">
              Review your past parlay predictions in the "History" section.
            </p>
          </li>
        </ol>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Responsible Betting</h2>
        
        <p className="mb-4">
          Our predictions are based on statistical analysis and historical data, but sports betting always involves risk. Please keep in mind:
        </p>
        
        <ul className="list-disc ml-6 space-y-2">
          <li>This tool is for entertainment and informational purposes only</li>
          <li>Always bet responsibly and within your means</li>
          <li>No prediction system can guarantee results</li>
          <li>Be aware of gambling laws and regulations in your jurisdiction</li>
        </ul>
      </div>
    </div>
  );
};

export default About;