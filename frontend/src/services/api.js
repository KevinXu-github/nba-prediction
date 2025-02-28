// src/services/api.js
const API_BASE_URL = 'http://localhost:8000/api';

class ApiService {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  async fetchJson(endpoint, options = {}) {
    const url = this.baseUrl + endpoint;
    console.log("API Request: " + (options.method || 'GET') + " " + url);
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      console.log("Response status: " + response.status + " " + response.statusText);
      
      // For non-2xx responses, try to parse error message
      if (!response.ok) {
        let errorMessage = "API Error: " + response.status + " " + response.statusText;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          // If JSON parsing fails, try to get text
          try {
            const errorText = await response.text();
            if (errorText) errorMessage += " - " + errorText;
          } catch (textError) {
            // Ignore text parsing errors
          }
        }
        
        console.error(errorMessage);
        throw new Error(errorMessage);
      }
      
      // For empty responses (like 204 No Content)
      if (response.status === 204) {
        return null;
      }
      
      // Parse JSON response
      const data = await response.json();
      console.log('Response data:', data);
      return data;
    } catch (error) {
      if (error.message.includes('Failed to fetch')) {
        console.error("Network error connecting to " + url + ". Is the server running?");
        throw new Error("Cannot connect to API server. Please check if the backend is running.");
      }
      
      console.error("API Error for " + url + ":", error);
      throw error;
    }
  }

  // Get upcoming games
  async getUpcomingGames() {
    try {
      const games = await this.fetchJson('/games');
      
      // Validate that we got an array
      if (!Array.isArray(games)) {
        console.error('Expected games array, got:', games);
        return []; // Return empty array as fallback
      }
      
      return games;
    } catch (error) {
      console.error('Error getting upcoming games:', error);
      return []; // Return empty array to prevent UI errors
    }
  }

  // Predict optimal parlay
  async predictParlay(parlaySize = 3, minConfidence = 0.6) {
    try {
      return await this.fetchJson('/predict-parlay', {
        method: 'POST',
        body: JSON.stringify({
          parlay_size: parlaySize,
          min_confidence: minConfidence,
        }),
      });
    } catch (error) {
      console.error('Error predicting parlay:', error);
      throw error;
    }
  }

  // Get parlay history
  async getParlayHistory() {
    try {
      const history = await this.fetchJson('/parlay-history');
      
      // Validate that we got an array
      if (!Array.isArray(history)) {
        console.error('Expected history array, got:', history);
        return []; // Return empty array as fallback
      }
      
      return history;
    } catch (error) {
      console.error('Error getting parlay history:', error);
      return []; // Return empty array to prevent UI errors
    }
  }

  // Refresh data
  async refreshData() {
    try {
      return await this.fetchJson('/refresh-data', {
        method: 'POST',
      });
    } catch (error) {
      console.error('Error refreshing data:', error);
      throw error;
    }
  }

  // Generate mock data for development/testing
  getMockGames() {
    console.log('Generating mock games data for development');
    const teams = [
      "LAL", "BOS", "GSW", "MIL", "PHX", "PHI", "BKN", "DEN",
      "LAC", "MIA", "DAL", "ATL", "CHI", "TOR", "CLE", "NYK"
    ];
    
    const mockGames = [];
    const today = new Date();
    
    for (let i = 0; i < 8; i++) {
      // Get two random different teams
      const homeIdx = Math.floor(Math.random() * teams.length);
      let awayIdx;
      do {
        awayIdx = Math.floor(Math.random() * teams.length);
      } while (awayIdx === homeIdx);
      
      const homeTeam = teams[homeIdx];
      const awayTeam = teams[awayIdx];
      
      // Random game date within the next week
      const gameDate = new Date(today);
      gameDate.setDate(today.getDate() + Math.floor(Math.random() * 7) + 1);
      
      mockGames.push({
        GameID: "mock-" + (i + 1),
        Date: gameDate.toISOString(),
        HomeTeam: homeTeam,
        AwayTeam: awayTeam,
        Stadium: homeTeam + " Arena",
        HomeTeamWins: Math.floor(Math.random() * 40) + 10,
        HomeTeamLosses: Math.floor(Math.random() * 40) + 10,
        AwayTeamWins: Math.floor(Math.random() * 40) + 10,
        AwayTeamLosses: Math.floor(Math.random() * 40) + 10,
        HomeTeamPointsPerGame: 100 + Math.random() * 20,
        HomeTeamPointsAllowedPerGame: 100 + Math.random() * 20,
        AwayTeamPointsPerGame: 100 + Math.random() * 20,
        AwayTeamPointsAllowedPerGame: 100 + Math.random() * 20,
        HomeTeamInjuries: Math.floor(Math.random() * 4),
        AwayTeamInjuries: Math.floor(Math.random() * 4),
        OverUnderLine: 210 + Math.floor(Math.random() * 30)
      });
    }
    
    return mockGames;
  }
}

// Create and export the API service
const api = new ApiService(API_BASE_URL);
export default api;
