// src/pages/GamesList.js
import React, { useState } from 'react';

const GamesList = ({ games, loading }) => {
  const [search, setSearch] = useState('');
  
  // Filter games by search term (with null safety)
  const filteredGames = Array.isArray(games) 
    ? games.filter(game => 
        (game?.HomeTeam?.toLowerCase().includes(search.toLowerCase()) ||
         game?.AwayTeam?.toLowerCase().includes(search.toLowerCase()))
      )
    : [];
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Upcoming NBA Games</h1>
      
      <div className="mb-6">
        <input
          type="text"
          placeholder="Search teams..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full p-3 border rounded-md"
        />
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="loading">
            <div></div>
            <div></div>
            <div></div>
          </div>
        </div>
      ) : !games || !Array.isArray(games) || games.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <p className="text-xl font-semibold mb-2">No upcoming games found</p>
          <p className="text-gray-600">Check back later for upcoming NBA games.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredGames.map((game, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md p-4 game-card">
              <div className="flex justify-between items-center mb-3">
                <div>
                  <span className="text-lg font-bold">{game.AwayTeam}</span>
                  <span className="vs-badge">@</span>
                  <span className="text-lg font-bold">{game.HomeTeam}</span>
                </div>
              </div>
              
              <div className="mt-2">
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Date:</span> {new Date(game.Date).toLocaleDateString()}
                </p>
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Time:</span> {new Date(game.Date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </p>
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Venue:</span> {game.Stadium}
                </p>
              </div>
              
              {game.OverUnderLine && (
                <div className="mt-3 pt-3 border-t">
                  <p className="text-sm">
                    <span className="font-semibold">Over/Under:</span> {game.OverUnderLine}
                  </p>
                </div>
              )}
              
              <div className="mt-3 pt-3 border-t grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-gray-600">Home Record</p>
                  <p className="font-medium">{game.HomeTeamWins}-{game.HomeTeamLosses}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Away Record</p>
                  <p className="font-medium">{game.AwayTeamWins}-{game.AwayTeamLosses}</p>
                </div>
              </div>
              
              <div className="mt-3 pt-3 border-t">
                <div className="grid grid-cols-2 gap-2 mb-2">
                  <div>
                    <p className="text-xs text-gray-600">Home PPG</p>
                    <p className="font-medium">{game.HomeTeamPointsPerGame?.toFixed(1) || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Away PPG</p>
                    <p className="font-medium">{game.AwayTeamPointsPerGame?.toFixed(1) || "N/A"}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GamesList;
