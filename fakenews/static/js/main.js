// static/js/main.js
// Main application logic

let currentArticleId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Application initialized');
    loadStats();
});

// Load statistics
async function loadStats() {
    try {
        const result = await API.getStats();
        if (result.success) {
            document.getElementById('totalArticles').textContent = result.total;
            document.getElementById('highCredibility').textContent = result.high_credibility;
            document.getElementById('lowCredibility').textContent = result.low_credibility;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Analyze article (called from HTML button)
async function analyzeArticle() {
    const urlInput = document.getElementById('articleUrl');
    const url = urlInput.value.trim();

    if (!url) {
        showError('Please enter a valid URL');
        return;
    }

    // Show loading
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';

    try {
        console.log('=== STARTING ANALYSIS ===');
        console.log('Analyzing URL:', url);
        const result = await API.analyzeArticle(url);
        console.log('=== BACKEND RESPONSE ===');
        console.log('Full result:', result);
        console.log('Success:', result.success);
        console.log('Cached:', result.cached);
        console.log('Article data:', result.article);
        console.log('Article keys:', result.article ? Object.keys(result.article) : 'No article data');

        if (result.success) {
            currentArticleId = result.article.id;
            
            // Display results
            displayArticleInfo(result.article);
            displayScores(result.article);
            
            // Render charts
            console.log('=== CHART DATA CHECK ===');
            console.log('Chart1 data exists:', !!result.article.chart1_data);
            console.log('Chart2 data exists:', !!result.article.chart2_data);
            console.log('Chart3 data exists:', !!result.article.chart3_data);
            console.log('Chart4 data exists:', !!result.article.chart4_data);
            
            if (result.article.chart1_data) {
                console.log('=== RENDERING CHART 1 ===');
                console.log('Chart1 data:', result.article.chart1_data);
                Charts.renderChart1(result.article.chart1_data);
            } else {
                console.log('❌ Chart1 data missing');
            }
            
            if (result.article.chart2_data) {
                console.log('=== RENDERING CHART 2 ===');
                console.log('Chart2 data:', result.article.chart2_data);
                Charts.renderChart2(result.article.chart2_data);
            } else {
                console.log('❌ Chart2 data missing');
            }
            
            if (result.article.chart3_data) {
                console.log('=== RENDERING CHART 3 ===');
                console.log('Chart3 data:', result.article.chart3_data);
                Charts.renderChart3(result.article.chart3_data);
            } else {
                console.log('❌ Chart3 data missing');
            }
            
            if (result.article.chart4_data) {
                console.log('=== RENDERING CHART 4 ===');
                console.log('Chart4 data:', result.article.chart4_data);
                Charts.renderChart4(result.article.chart4_data);
            } else {
                console.log('❌ Chart4 data missing');
            }
            
            // Render chart 5 (similarity map)
            await Charts.renderChart5(currentArticleId);
            
            // Render chart 6 (related articles)
            if (result.article.chart6_data) {
                console.log('✅ Rendering Chart 6 (Related Articles)');
                Charts.renderChart6(result.article.chart6_data);
            } else {
                console.log('❌ Chart6 data missing');
            }
            
            // Render enhanced credibility metrics
            if (result.article.detailed_metrics && window.EnhancedMetrics) {
                console.log('✅ Rendering Enhanced Credibility Metrics');
                window.EnhancedMetrics.renderCredibilityCards(result.article);
            } else {
                console.log('❌ Enhanced metrics not available');
            }
            
            // Get additional analysis using the new prediction endpoint
            console.log('=== GETTING ADDITIONAL ANALYSIS ===');
            const articleContent = result.article.content || '';
            console.log('Article content length:', articleContent.length);
            
            if (articleContent.length > 0) {
                try {
                    console.log('🔮 Calling prediction API for sensationalism analysis...');
                    console.log('🔮 Article content preview:', articleContent.substring(0, 200) + '...');
                    console.log('🔮 Content length:', articleContent.length);
                    
                    const predictionResult = await API.predictSensationalism(articleContent);
                    console.log('🔮 Full prediction result:', predictionResult);
                    console.log('🔮 Sensationalism score:', predictionResult.sensationalism_bias_likelihood);
                    console.log('🔮 Analysis available:', predictionResult.analysis_available);
                    console.log('🔮 Model confidence:', predictionResult.model_confidence);
                    
                    if (predictionResult.error) {
                        console.error('❌ Prediction API returned error:', predictionResult.error);
                    }
                    
                    if (predictionResult.sensationalism_bias_likelihood === 0.5) {
                        console.warn('⚠️ WARNING: Sensationalism score is 0.5 - this indicates a problem!');
                        console.warn('⚠️ This usually means:');
                        console.warn('   - Model not loaded properly');
                        console.warn('   - Text processing failed');
                        console.warn('   - ML prediction failed');
                        console.warn('   - Fallback to default value');
                    }
                    
                    // Update the article data with prediction results
                    result.article.sensationalism_bias_likelihood = predictionResult.sensationalism_bias_likelihood;
                    result.article.feature_breakdown = predictionResult.feature_breakdown;
                    result.article.interpretation = predictionResult.interpretation;
                    
                    console.log('✅ Updated article with prediction data');
                    console.log('✅ Final sensationalism score:', result.article.sensationalism_bias_likelihood);
                } catch (error) {
                    console.error('❌ Error getting prediction:', error);
                    console.error('❌ Error details:', error.message);
                    console.error('❌ This means the /api/predict endpoint failed');
                }
            } else {
                console.warn('⚠️ No article content available for prediction');
            }
            
            // Render additional charts if data is available
            console.log('=== ADDITIONAL CHARTS CHECK ===');
            console.log('Sensationalism data exists:', result.article.sensationalism_bias_likelihood !== undefined);
            console.log('Word frequency data exists:', !!result.article.word_frequency_data);
            console.log('Sentiment flow data exists:', !!result.article.sentiment_flow_data);
            
            if (result.article.sensationalism_bias_likelihood !== undefined) {
                console.log('=== SENSATIONALISM DATA AVAILABLE ===');
                console.log('Sensationalism value:', result.article.sensationalism_bias_likelihood);
                // Sensationalism score is now displayed in the 2x2 grid, not in overall score
            } else {
                console.log('❌ Sensationalism data missing');
            }
            
            // Word frequency chart removed - not needed
            
            // Sentiment flow chart removed - not needed
            
            // Show results
            document.getElementById('resultsContainer').style.display = 'block';
            
            // Update stats
            loadStats();
            
            // Scroll to results
            document.getElementById('resultsContainer').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        } else {
            showError(result.error || 'Failed to analyze article');
        }
    } catch (error) {
        console.error('Error analyzing article:', error);
        showError('Network error: ' + error.message);
    } finally {
        document.getElementById('loadingIndicator').style.display = 'none';
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = '⚠ ' + message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}


// Load database (called when switching to database tab)
async function loadDatabase() {
    try {
        const result = await API.getArticles();
        
        if (result.success && result.articles.length > 0) {
            renderDatabaseTable(result.articles);
            document.getElementById('databaseTable').style.display = 'block';
            document.getElementById('noDatabaseData').style.display = 'none';
        } else {
            document.getElementById('databaseTable').style.display = 'none';
            document.getElementById('noDatabaseData').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading database:', error);
    }
}

// Render database table
function renderDatabaseTable(articles) {
    // Apply filters
    const minScore = parseInt(document.getElementById('minScoreFilter').value) / 100;
    const sortBy = document.getElementById('sortBy').value;
    
    // Filter
    let filtered = articles.filter(a => a.overall_score >= minScore);
    
    // Sort
    filtered.sort((a, b) => {
        if (sortBy === 'analyzed_at') {
            return new Date(b[sortBy]) - new Date(a[sortBy]);
        }
        return b[sortBy] - a[sortBy];
    });
    
    // Generate table HTML
    let html = `
        <table>
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Source</th>
                    <th>Overall Score</th>
                    <th>Credibility</th>
                    <th>Date</th>
                    <th>URL</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    filtered.forEach(article => {
        const score = Math.round(article.overall_score * 100);
        const credScore = Math.round(article.credibility_score * 100);
        const date = article.analyzed_at ? article.analyzed_at.substring(0, 10) : 'N/A';
        const title = article.title.length > 60 ? article.title.substring(0, 60) + '...' : article.title;
        
        html += `
            <tr>
                <td>${title}</td>
                <td>${article.source || 'N/A'}</td>
                <td style="color: ${score >= 70 ? 'var(--success)' : score >= 40 ? 'var(--warning)' : 'var(--danger)'}">${score}%</td>
                <td>${credScore}%</td>
                <td>${date}</td>
                <td><a href="${article.url}" target="_blank" style="color: var(--accent)">Link</a></td>
            </tr>
        `;
    });
    
    html += '</tbody></table>';
    
    document.getElementById('databaseTable').innerHTML = html;
    
    // Update min score display
    document.getElementById('minScoreValue').textContent = document.getElementById('minScoreFilter').value + '%';
}

// Export database to CSV
async function exportDatabase() {
    try {
        const result = await API.getArticles();
        
        if (!result.success || result.articles.length === 0) {
            alert('No articles to export');
            return;
        }
        
        // Create CSV
        const headers = ['Title', 'Source', 'URL', 'Overall Score', 'Credibility', 'Language', 'Cross-Check', 'Date'];
        const rows = result.articles.map(a => [
            `"${a.title.replace(/"/g, '""')}"`,
            a.source || 'N/A',
            a.url,
            Math.round(a.overall_score * 100),
            Math.round(a.credibility_score * 100),
            Math.round(a.language_score * 100),
            Math.round(a.cross_check_score * 100),
            a.analyzed_at ? a.analyzed_at.substring(0, 10) : 'N/A'
        ]);
        
        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        
        // Download
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fake_news_detector_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error exporting:', error);
        alert('Failed to export database');
    }
}

// Clear database
async function clearDatabase() {
    if (!confirm('Are you sure you want to clear the entire database? This cannot be undone.')) {
        return;
    }
    
    try {
        const result = await API.clearDatabase();
        
        if (result.success) {
            alert('Database cleared successfully');
            currentArticleId = null;
            loadStats();
            loadDatabase();
            
            // Reset analyze tab
            document.getElementById('resultsContainer').style.display = 'none';
            document.getElementById('articleUrl').value = '';
        }
    } catch (error) {
        console.error('Error clearing database:', error);
        alert('Failed to clear database');
    }
}

// Add event listeners for database filters
document.getElementById('minScoreFilter').addEventListener('input', () => {
    if (document.getElementById('databaseTab').classList.contains('active')) {
        loadDatabase();
    }
});

document.getElementById('sortBy').addEventListener('change', () => {
    if (document.getElementById('databaseTab').classList.contains('active')) {
        loadDatabase();
    }
});
