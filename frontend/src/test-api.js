// Save this as test-api.js
const testApi = async () => {
  try {
    const response = await fetch('http://localhost:8000/api/games');
    
    console.log('Status:', response.status);
    console.log('Status Text:', response.statusText);
    console.log('Headers:', Object.fromEntries([...response.headers]));
    
    if (!response.ok) {
      console.error('Error response:', response.status, response.statusText);
      const text = await response.text();
      console.error('Error body:', text);
      return;
    }
    
    const data = await response.json();
    console.log('API response data:', data);
    console.log('Number of games:', Array.isArray(data) ? data.length : 'Not an array');
  } catch (error) {
    console.error('Fetch error:', error);
  }
};

// Run the test
testApi();
