// src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const Dashboard = ({ games, loading }) => {
  const [parlays, setParlays] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    fetchRecentParlays();
  }, []);

  const fetchRecentParlays = async () => {
    setLoadingHistory(true);
    try {
      const history = await api.getParlayHistory();
      setParlays(history.slice(0, 3)); // Show only the 3 most recent parlays
    } catch (error) {
      console.error('Error fetching parlay history:', error);
    } finally {
      setLoadingHistory(false);
    }
  };

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">NBA Parlay Prediction Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">System Overview</h2>
          <div className="mb-4">
            <p className="mb-2">Our AI system analyzes NBA games and predicts the most profitable parlay bets.</p>
            <p>The system considers multiple factors including:</p>
            <ul className="list-disc ml-6 mt-2">
              <li>Team performance stats</li>
              <li>Player injuries</li>
              <li>Historical betting trends</li>
              <li>Home/away performance</li>
              <li>Line movements</li>
            </ul>
          </div>
          <Link 
            to="/generate-parlay" 
            className="inline-block mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
            Generate Parlay
          </Link>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">Upcoming Games ({games.length})</h2>
          {loading ? (
            <p>Loading games...</p>
          ) : games.length === 0 ? (
            <p>No upcoming games found</p>
          ) : (
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {games.slice(0, 5).map((game, index) => (
                <div key={index} className="border-b pb-2">
                  <p className="font-medium">{game.AwayTeam} @ {game.HomeTeam}</p>
                  <p className="text-sm text-gray-600">
                    {new Date(game.Date).toLocaleDateString()} - {game.Stadium}
                  </p>
                  {game.OverUnderLine && (
                    <p className="text-sm">O/U: {game.OverUnderLine}</p>
                  )}
                </div>
              ))}
            </div>
          )}
          <Link 
            to="/games" 
            className="inline-block mt-4 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700">
            View All Games
          </Link>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Recent Parlays</h2>
          <Link 
            to="/history" 
            className="text-blue-600 hover:underline">
            View All
          </Link>
        </div>
        
        {loadingHistory ? (
          <p>Loading recent parlays...</p>
        ) : parlays.length === 0 ? (
          <p>No parlays generated yet</p>
        ) : (
          <div className="space-y-4">
            {parlays.map((parlay) => (
              <div key={parlay.parlay_id} className="border rounded-md p-4">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-bold">{parlay.parlay_size}-Leg Parlay</h3>
                  <span className={`px-2 py-1 text-xs rounded-full text-white ${
                    parlay.risk_level === 'Low' ? 'bg-green-500' : 
                    parlay.risk_level === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}>
                    {parlay.risk_level} Risk
                  </span>
                </div>
                <p className="text-sm mb-2">
                  Confidence: {Math.round(parlay.overall_confidence * 100)}%
                </p>
                <div className="mt-2">
                  {parlay.games.map((game, i) => (
                    <div key={i} className="text-sm mb-1">
                      • {game.away_team} @ {game.home_team}: <span className="font-medium">{game.prediction}</span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Created: {new Date(parlay.created_at).toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;