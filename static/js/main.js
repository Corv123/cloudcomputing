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
        console.log('Analyzing:', url);
        const result = await API.analyzeArticle(url);
        console.log('Analysis result:', result);

        if (result.success) {
            currentArticleId = result.article.id;
            
            // Display results
            displayArticleInfo(result.article);
            displayScores(result.article);
            
            // Render charts
            if (result.article.chart1_data) {
                Charts.renderChart1(result.article.chart1_data);
            }
            if (result.article.chart2_data) {
                Charts.renderChart2(result.article.chart2_data);
            }
            if (result.article.chart3_data) {
                // DEBUG: Chart 3 Data Flow
                console.log('🔍 CHART 3 DEBUG:');
                console.log('chart3_data:', result.article.chart3_data);
                console.log('detailed_metrics:', result.article.detailed_metrics);

                if (result.article.detailed_metrics) {
                    console.log('Domain Age:', result.article.detailed_metrics.domain_age);
                    console.log('URL Structure:', result.article.detailed_metrics.url_structure);
                    console.log('Site Structure:', result.article.detailed_metrics.site_structure);
                    console.log('Content Format:', result.article.detailed_metrics.content_format);
                }

                Charts.renderChart3(result.article.chart3_data, result.article.detailed_metrics);
            }
            if (result.article.chart4_data) {
                Charts.renderChart4(result.article.chart4_data);
            }

            // Render chart 5 (similarity map)
            await Charts.renderChart5(currentArticleId);

            // Render chart 6 (related articles)
            if (result.article.chart6_data) {
                Charts.renderChart6(result.article.chart6_data);
            }

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
