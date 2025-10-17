// static/js/api.js
// API module for backend communication

const API = {
    baseURL: window.location.origin + '/api',

    // Analyze article from URL
    async analyzeArticle(url) {
        try {
            const response = await fetch(`${this.baseURL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            return await response.json();
        } catch (error) {
            console.error('API Error (analyze):', error);
            throw error;
        }
    },

    // Get database statistics
    async getStats() {
        try {
            const response = await fetch(`${this.baseURL}/stats`);
            return await response.json();
        } catch (error) {
            console.error('API Error (stats):', error);
            throw error;
        }
    },

    // Get all articles
    async getArticles() {
        try {
            const response = await fetch(`${this.baseURL}/articles`);
            return await response.json();
        } catch (error) {
            console.error('API Error (articles):', error);
            throw error;
        }
    },

    // Get similarity map data
    async getSimilarityMap(articleId) {
        try {
            const response = await fetch(`${this.baseURL}/similarity-map/${articleId}`);
            return await response.json();
        } catch (error) {
            console.error('API Error (similarity):', error);
            throw error;
        }
    },

    // Clear database
    async clearDatabase() {
        try {
            const response = await fetch(`${this.baseURL}/clear-database`, {
                method: 'POST'
            });
            return await response.json();
        } catch (error) {
            console.error('API Error (clear):', error);
            throw error;
        }
    }
};
