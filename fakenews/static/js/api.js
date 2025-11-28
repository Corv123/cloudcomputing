// static/js/api.js
// API module for backend communication - Updated for AWS API Gateway

console.log('🔵 API.JS LOADED - VERSION 2.1 WITH UNWRAPPING');
console.log('🔵 File loaded at:', new Date().toISOString());
console.log('🔵 This is a FRESH upload - old version was deleted!');

const API = {
    baseURL: 'https://bq7y9l5ruf.execute-api.ap-southeast-2.amazonaws.com/prod',

    // Analyze article from URL
    async analyzeArticle(url) {
        try {
            console.log('[API] ===== ANALYZE ARTICLE CALLED =====');
            console.log('[API] Version: 2.0 (with unwrapping)');
            console.log('[API] Calling analyze endpoint:', `${this.baseURL}/analyze`);
            console.log('[API] Request body:', { url });
            
            const response = await fetch(`${this.baseURL}/analyze`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });
            
            console.log('[API] Response status:', response.status);
            console.log('[API] Response headers:', Object.fromEntries(response.headers.entries()));
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('[API] Error response:', errorText);
                let errorData;
                try {
                    errorData = JSON.parse(errorText);
                } catch {
                    errorData = { error: errorText || `HTTP ${response.status}` };
                }
                throw new Error(errorData.error || errorData.message || `HTTP ${response.status}`);
            }
            
            let result;
            const responseText = await response.text();
            console.log('[API] Raw response text:', responseText.substring(0, 200));
            
            try {
                result = JSON.parse(responseText);
            } catch (e) {
                console.error('[API] Failed to parse JSON:', e);
                throw new Error('Invalid JSON response: ' + responseText.substring(0, 100));
            }
            
            console.log('[API] Response received:', result);
            console.log('[API] Response type:', typeof result);
            console.log('[API] Has statusCode?', !!result.statusCode);
            console.log('[API] Has body?', !!result.body);
            console.log('[API] Has success?', !!result.success);
            console.log('[API] Has article?', !!result.article);
            
            // Handle API Gateway response format (if wrapped in {statusCode, headers, body})
            if (result && result.statusCode && result.body) {
                console.log('[API] Response is wrapped - unwrapping...');
                // API Gateway returned the Lambda response format directly
                // Parse the body string as JSON
                let parsedBody;
                if (typeof result.body === 'string') {
                    try {
                        parsedBody = JSON.parse(result.body);
                        console.log('[API] Successfully parsed body string');
                    } catch (e) {
                        console.error('[API] Error parsing body:', e);
                        throw new Error('Failed to parse response body: ' + e.message);
                    }
                } else {
                    parsedBody = result.body;
                    console.log('[API] Body is already an object');
                }
                console.log('[API] Unwrapped response body:', parsedBody);
                console.log('[API] Unwrapped success:', parsedBody.success);
                console.log('[API] Unwrapped article:', parsedBody.article ? 'exists' : 'missing');
                return parsedBody;
            }
            
            // API Gateway already unwrapped it - result should be the body directly
            console.log('[API] Response is already unwrapped by API Gateway');
            console.log('[API] Returning result directly:', {
                success: result.success,
                hasArticle: !!result.article,
                keys: Object.keys(result)
            });
            return result;
        } catch (error) {
            console.error('[API] Error (analyze):', error);
            console.error('[API] Error details:', {
                message: error.message,
                stack: error.stack,
                name: error.name
            });
            throw error;
        }
    },

    // Get database statistics
    async getStats() {
        try {
            console.log('[API] Fetching stats from:', `${this.baseURL}/stats`);
            const response = await fetch(`${this.baseURL}/stats`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            console.log('[API] Stats response status:', response.status);
            console.log('[API] Stats response headers:', Object.fromEntries(response.headers.entries()));
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('[API] Stats error response:', errorText);
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const responseText = await response.text();
            console.log('[API] Stats raw response:', responseText);
            
            let result;
            try {
                result = JSON.parse(responseText);
            } catch (e) {
                // Check if API Gateway wrapped it
                const wrapped = JSON.parse(responseText);
                if (wrapped.statusCode && wrapped.body) {
                    result = typeof wrapped.body === 'string' ? JSON.parse(wrapped.body) : wrapped.body;
                } else {
                    throw e;
                }
            }
            
            console.log('[API] Stats result:', result);
            return result;
        } catch (error) {
            console.error('API Error (stats):', error);
            throw error;
        }
    },

    // Get all articles
    async getArticles() {
        try {
            console.log('🔵 [API.JS] Fetching articles from:', `${this.baseURL}/articles`);
            const response = await fetch(`${this.baseURL}/articles`);
            console.log('🔵 [API.JS] Response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            console.log('🔵 [API.JS] Raw result:', result);
            
            // Unwrap Lambda proxy response if needed
            if (result.statusCode && result.body) {
                console.log('🔵 [API.JS] Unwrapping Lambda response...');
                const bodyData = typeof result.body === 'string' ? JSON.parse(result.body) : result.body;
                console.log('🔵 [API.JS] Unwrapped data:', bodyData);
                return bodyData;
            }
            
            console.log('🔵 [API.JS] Returning direct result');
            return result;
        } catch (error) {
            console.error('❌ [API.JS] Error fetching articles:', error);
            throw error;
        }
    },

    // Get similarity map data (not available in Lambda - returns placeholder)
    async getSimilarityMap(articleId) {
        try {
            // Endpoint doesn't exist in Lambda - return empty result
            return {
                success: false,
                data: [],
                message: 'Similarity map endpoint not available'
            };
        } catch (error) {
            console.error('API Error (similarity):', error);
            return {
                success: false,
                data: [],
                error: error.message
            };
        }
    },

    // Clear database
    async clearDatabase() {
        try {
            const response = await fetch(`${this.baseURL}/clear-database`, {
                method: 'POST'
            });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const result = await response.json();
            
            // Unwrap Lambda proxy response if needed
            if (result.statusCode && result.body) {
                const bodyData = typeof result.body === 'string' ? JSON.parse(result.body) : result.body;
                return bodyData;
            }
            
            return result;
        } catch (error) {
            console.error('API Error (clear):', error);
            throw error;
        }
    },

    // Predict sensationalism score (not available as separate endpoint - already included in analyze)
    async predictSensationalism(text) {
        try {
            // This endpoint doesn't exist - sensationalism is already in analyze response
            console.warn('predictSensationalism called but endpoint not available - sensationalism already in analyze response');
            return {
                sensationalism_bias_likelihood: 0.5,
                analysis_available: false,
                error: 'Endpoint not available - use analyze endpoint instead'
            };
        } catch (error) {
            console.error('API Error (predict):', error);
            return {
                sensationalism_bias_likelihood: 0.5,
                analysis_available: false,
                error: error.message
            };
        }
    },

    // Batch predict sensationalism scores
    async predictBatch(texts) {
        try {
            const response = await fetch(`${this.baseURL}/predict-batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts })
            });
            return await response.json();
        } catch (error) {
            console.error('API Error (predict-batch):', error);
            throw error;
        }
    }
};