// src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

// Components
import Header from './components/Header';
import Footer from './components/Footer';
import Dashboard from './pages/Dashboard';
import ParlayGenerator from './pages/ParlayGenerator';
import ParlayHistory from './pages/ParlayHistory';
import GamesList from './pages/GamesList';
import About from './pages/About';

// API service
import api from './services/api';

function App() {
  const [loading, setLoading] = useState(false);
  const [upcomingGames, setUpcomingGames] = useState([]);
  const [apiConnected, setApiConnected] = useState(true);

  // Load upcoming games on startup
  useEffect(() => {
    fetchUpcomingGames();
  }, []);

  // Function to fetch upcoming games
  const fetchUpcomingGames = async () => {
    setLoading(true);
    try {
      // Try to get games from API
      const games = await api.getUpcomingGames();
      
      // If no games are returned, try to use mock data
      if (!games || games.length === 0) {
        console.log('No games returned from API, using mock data');
        setUpcomingGames(api.getMockGames());
        setApiConnected(false);
        toast.warning('Using mock data - API returned no games', {
          position: "bottom-right",
          autoClose: 5000
        });
      } else {
        setUpcomingGames(games);
        setApiConnected(true);
      }
    } catch (error) {
      console.error('Error fetching upcoming games:', error);
      toast.error('Failed to load games from API. Using mock data instead.', {
        position: "bottom-right",
        autoClose: 5000
      });
      
      // Use mock data as fallback
      setUpcomingGames(api.getMockGames());
      setApiConnected(false);
    } finally {
      setLoading(false);
    }
  };

  // Function to generate parlay
  const generateParlay = async (parlaySize, minConfidence) => {
    setLoading(true);
    try {
      // If API is not connected, generate mock data
      if (!apiConnected) {
        // Simple mock parlay response
        const mockParlay = {
          parlay_id: "mock-" + Date.now(),
          games: upcomingGames.slice(0, parlaySize).map(game => ({
            game_id: game.GameID,
            home_team: game.HomeTeam,
            away_team: game.AwayTeam,
            game_date: game.Date,
            prediction: Math.random() > 0.5 ? 'Over' : 'Under',
            confidence: 0.6 + Math.random() * 0.3,
            risk_level: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)],
            over_under_line: game.OverUnderLine
          })),
          overall_confidence: 0.6 + Math.random() * 0.2,
          risk_level: 'Medium',
          parlay_size: parlaySize,
          created_at: new Date().toISOString()
        };
        
        setTimeout(() => {
          setLoading(false);
        }, 1000); // Add artificial delay to simulate API call
        
        return mockParlay;
      }
      
      // Otherwise use the real API
      const parlay = await api.predictParlay(parlaySize, minConfidence);
      return parlay;
    } catch (error) {
      console.error('Error generating parlay:', error);
      toast.error(error.message || 'Failed to generate parlay');
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Function to refresh data
  const refreshData = async () => {
    try {
      if (apiConnected) {
        await api.refreshData();
        toast.info('Data refresh scheduled');
      } else {
        toast.info('Using mock data - API refresh not available');
      }
      
      // Fetch updated games
      fetchUpcomingGames();
    } catch (error) {
      console.error('Error refreshing data:', error);
      toast.error('Failed to refresh data');
    }
  };

  return (
    <Router>
      <div className="app">
        <ToastContainer position="top-right" autoClose={3000} />
        <Header onRefreshData={refreshData} loading={loading} apiConnected={apiConnected} />
        
        <div className="container mx-auto px-4 py-6">
          {!apiConnected && (
            <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6">
              <p className="font-bold">Development Mode</p>
              <p>Using mock data - Backend API connection is unavailable. Some features will be simulated.</p>
            </div>
          )}
          
          <Routes>
            <Route path="/" element={<Dashboard games={upcomingGames} loading={loading} />} />
            <Route 
              path="/generate-parlay" 
              element={
                <ParlayGenerator 
                  onGenerateParlay={generateParlay} 
                  games={upcomingGames}
                  loading={loading} 
                />
              } 
            />
            <Route path="/history" element={<ParlayHistory />} />
            <Route path="/games" element={<GamesList games={upcomingGames} loading={loading} />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
        
        <Footer />
      </div>
    </Router>
  );
}

export default App;
