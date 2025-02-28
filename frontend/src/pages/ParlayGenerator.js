// src/pages/ParlayGenerator.js
import React, { useState } from 'react';

const ParlayGenerator = ({ onGenerateParlay, games, loading }) => {
  const [parlaySize, setParlaySize] = useState(3);
  const [minConfidence, setMinConfidence] = useState(0.6);
  const [parlay, setParlay] = useState(null);
  const [generating, setGenerating] = useState(false);
  
  const handleGenerateParlay = async () => {
    setGenerating(true);
    try {
      const result = await onGenerateParlay(parlaySize, minConfidence);
      setParlay(result);
    } catch (error) {
      console.error('Error generating parlay:', error);
    } finally {
      setGenerating(false);
    }
  };
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Generate Optimal Parlay</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="md:col-span-1 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">Parlay Settings</h2>
          
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Number of Legs:</label>
            <select
              className="w-full p-2 border rounded-md"
              value={parlaySize}
              onChange={(e) => setParlaySize(parseInt(e.target.value))}
            >
              <option value={2}>2-Leg Parlay</option>
              <option value={3}>3-Leg Parlay</option>
              <option value={4}>4-Leg Parlay</option>
              <option value={5}>5-Leg Parlay</option>
            </select>
          </div>
          
          <div className="mb-6">
            <label className="block text-gray-700 mb-2">Minimum Confidence:</label>
            <div className="flex items-center">
              <input
                type="range"
                min="0.5"
                max="0.9"
                step="0.05"
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                className="w-full mr-3"
              />
              <span className="text-sm">{Math.round(minConfidence * 100)}%</span>
            </div>
          </div>
          
          <button
            onClick={handleGenerateParlay}
            disabled={generating || loading || games.length === 0}
            className="w-full py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          >
            {generating ? 'Generating...' : 'Generate Parlay'}
          </button>
          
          {games.length === 0 && !loading && (
            <p className="text-red-500 text-sm mt-2">No games available for prediction</p>
          )}
        </div>
        
        <div className="md:col-span-2 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">Predicted Parlay</h2>
          
          {generating ? (
            <div className="flex justify-center items-center h-64">
              <p>Generating optimal parlay...</p>
            </div>
          ) : !parlay ? (
            <div className="flex justify-center items-center h-64 text-gray-500">
              <p>Generate a parlay to see predictions</p>
            </div>
          ) : (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">{parlay.parlay_size}-Leg Parlay</h3>
                <div className="flex items-center">
                  <span className="mr-3">
                    Overall Confidence: <span className="font-bold">{Math.round(parlay.overall_confidence * 100)}%</span>
                  </span>
                  <span className={`px-3 py-1 rounded-full text-sm text-white ${
                    parlay.risk_level === 'Low' ? 'bg-green-500' : 
                    parlay.risk_level === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}>
                    {parlay.risk_level} Risk
                  </span>
                </div>
              </div>
              
              <div className="space-y-4 mt-6">
                {parlay.games.map((game, index) => (
                  <div key={index} className="border rounded-md p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-bold">{game.away_team} @ {game.home_team}</h4>
                        <p className="text-sm text-gray-600">
                          {new Date(game.game_date).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center">
                          <span className="text-lg font-bold mr-2">{game.prediction}</span>
                          {game.over_under_line && (
                            <span className="text-sm">({game.over_under_line})</span>
                          )}
                        </div>
                        <div className="flex items-center mt-1">
                          <span className="text-sm mr-2">
                            Confidence: {Math.round(game.confidence * 100)}%
                          </span>
                          <span className={`px-2 py-0.5 rounded-full text-xs text-white ${
                            game.risk_level === 'Low' ? 'bg-green-500' : 
                            game.risk_level === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
                          }`}>
                            {game.risk_level}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ParlayGenerator;