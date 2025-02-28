// src/pages/ParlayHistory.js
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const ParlayHistory = () => {
  const [parlays, setParlays] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchParlayHistory();
  }, []);
  
  const fetchParlayHistory = async () => {
    setLoading(true);
    try {
      const history = await api.getParlayHistory();
      setParlays(history);
    } catch (error) {
      console.error('Error fetching parlay history:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Parlay History</h1>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <p>Loading parlay history...</p>
        </div>
      ) : parlays.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <p>No parlay history found</p>
        </div>
      ) : (
        <div className="space-y-6">
          {parlays.map((parlay) => (
            <div key={parlay.parlay_id} className="bg-white rounded-lg shadow-md p-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                <div>
                  <h2 className="text-xl font-bold">{parlay.parlay_size}-Leg Parlay</h2>
                  <p className="text-sm text-gray-600">
                    Created: {new Date(parlay.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="mt-2 md:mt-0 flex items-center">
                  <span className="mr-4">
                    Confidence: <span className="font-bold">{Math.round(parlay.overall_confidence * 100)}%</span>
                  </span>
                  <span className={`px-3 py-1 rounded-full text-sm text-white ${
                    parlay.risk_level === 'Low' ? 'bg-green-500' : 
                    parlay.risk_level === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}>
                    {parlay.risk_level} Risk
                  </span>
                </div>
              </div>
              
              <div className="mt-4 space-y-3">
                {parlay.games.map((game, index) => (
                  <div key={index} className="border-b pb-3">
                    <div className="flex justify-between">
                      <div>
                        <p className="font-medium">{game.away_team} @ {game.home_team}</p>
                        <p className="text-sm text-gray-600">
                          {new Date(game.game_date).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold">{game.prediction} {game.over_under_line && `(${game.over_under_line})`}</p>
                        <div className="flex items-center justify-end mt-1">
                          <span className="text-sm mr-2">
                            {Math.round(game.confidence * 100)}%
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
          ))}
        </div>
      )}
    </div>
  );
};

export default ParlayHistory;