// src/components/Header.js
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Header = ({ onRefreshData, loading, apiConnected }) => {
  const location = useLocation();

  return (
    <header className="bg-blue-900 text-white shadow-lg">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <Link to="/" className="text-2xl font-bold">NBA Parlay Predictor</Link>
            {!apiConnected && (
              <span className="ml-2 px-2 py-1 bg-yellow-500 text-xs font-bold rounded">MOCK DATA</span>
            )}
          </div>
          
          <nav className="flex flex-wrap justify-center">
            <Link 
              to="/" 
              className={"px-3 py-2 mr-2 rounded-md " + (location.pathname === '/' ? 'bg-blue-700' : 'hover:bg-blue-800')}>
              Dashboard
            </Link>
            <Link 
              to="/generate-parlay" 
              className={"px-3 py-2 mr-2 rounded-md " + (location.pathname === '/generate-parlay' ? 'bg-blue-700' : 'hover:bg-blue-800')}>
              Generate Parlay
            </Link>
            <Link 
              to="/games" 
              className={"px-3 py-2 mr-2 rounded-md " + (location.pathname === '/games' ? 'bg-blue-700' : 'hover:bg-blue-800')}>
              Games
            </Link>
            <Link 
              to="/history" 
              className={"px-3 py-2 mr-2 rounded-md " + (location.pathname === '/history' ? 'bg-blue-700' : 'hover:bg-blue-800')}>
              History
            </Link>
            <button 
              onClick={onRefreshData} 
              disabled={loading}
              className="px-3 py-2 rounded-md bg-green-600 hover:bg-green-700 ml-2 disabled:bg-gray-500">
              {loading ? 'Loading...' : 'Refresh Data'}
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
