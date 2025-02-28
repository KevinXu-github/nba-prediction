// src/pages/GamesList.js
import React, { useState } from 'react';

const GamesList = ({ games, loading }) => {
  const [search, setSearch] = useState('');
  
  // Filter games by search term
  const filteredGames = games.filter(game => 
    game.HomeTeam.toLowerCase().includes(search.toLowerCase()) ||
    game.AwayTeam.toLowerCase().includes(search.toLowerCase())
  );
  
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
          <p>Loading games...</p>
        </div>
      ) : games.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <p>No upcoming games found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredGames.map((game, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md p-4">
              <div className="flex justify-between items-center mb-3">
                <div>
                  <span className="text-lg font-bold">{game.AwayTeam}</span>
                  <span className="mx-2 text-gray-500">@</span>
                  <span className="text-lg font-bold">{game.HomeTeam}</span>
                </div>
              </div>
              
              <p className="text-sm text-gray-600 mb-2">
                {new Date(game.Date).toLocaleDateString()} {new Date(game.Date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
              </p>
              <p className="text-sm mb-2">{game.Stadium}</p>
              
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
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GamesList;