// static/js/charts-display.js
// Module for rendering charts using Plotly

console.log('🔵 CHARTS-DISPLAY.JS LOADED - VERSION 2.4');
console.log('🔵 File loaded at:', new Date().toISOString());
console.log('🔵 FIXED: Removed unreachable code and unclosed comments');

const Charts = {
    // Common chart configuration
    config: {
        responsive: true,
        displayModeBar: false
    },

    layout: {
        paper_bgcolor: '#1A1A1A',
        plot_bgcolor: '#1A1A1A',
        font: { color: '#FAFAFA' },
        margin: { t: 30, r: 20, b: 50, l: 50 }
    },

    // Chart 1: Language Quality Bar Chart
    renderChart1(data) {
        console.log('🔍 renderChart1 called');
        console.log('Data received:', data);
        console.log('Data type:', typeof data);
        console.log('Data keys:', data ? Object.keys(data) : 'null');
        
        if (!data) {
            console.log('❌ renderChart1: No data provided');
            document.getElementById('chart1').innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">No data available</p>';
            return;
        }
        
        console.log('✅ renderChart1: Data is valid, proceeding with rendering');

        const trace = {
            x: data.labels,
            y: data.values,
            type: 'bar',
            marker: {
                color: ['#FF4B4B', '#FF6B6B', '#FF8B8B', '#FFABAB']
            }
        };

        const layout = {
            ...this.layout,
            yaxis: { title: 'Score (%)', range: [0, 100] },
            xaxis: { title: '' }
        };

        try {
            console.log('🎨 renderChart1: Creating Plotly chart');
            console.log('Chart container element:', document.getElementById('chart1'));
            console.log('Trace data:', trace);
            console.log('Layout data:', layout);

        Plotly.newPlot('chart1', [trace], layout, this.config);
            console.log('✅ renderChart1: Chart rendered successfully');
        } catch (error) {
            console.error('❌ renderChart1: Error rendering chart:', error);
            document.getElementById('chart1').innerHTML = '<p style="padding: 2rem; text-align: center; color: #FF5252;">Chart rendering error</p>';
        }
    },

    // Chart 2: Sentiment Pie Chart
    renderChart2(data) {
        if (!data) {
            document.getElementById('chart2').innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">No data available</p>';
            return;
        }

        const trace = {
            labels: data.labels,
            values: data.values,
            type: 'pie',
            marker: {
                colors: ['#A0A0A0', '#FF4B4B', '#FF6B6B', '#00C853']
            }
        };

        const layout = {
            ...this.layout,
            showlegend: true,
            legend: { x: 1, y: 0.5 }
        };

        Plotly.newPlot('chart2', [trace], layout, this.config);
    },

    // Chart 3: Credibility Radar Chart
    renderChart3(data) {
        if (!data) {
            document.getElementById('chart3').innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">No data available</p>';
            return;
        }

        const trace = {
            type: 'scatterpolar',
            r: data.values,
            theta: data.labels,
            fill: 'toself',
            fillcolor: 'rgba(255, 75, 75, 0.3)',
            line: {
                color: '#FF4B4B',
                width: 2
            }
        };

        const layout = {
            ...this.layout,
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100],
                    gridcolor: '#3D3D3D'
                },
                angularaxis: {
                    gridcolor: '#3D3D3D'
                }
            }
        };

        Plotly.newPlot('chart3', [trace], layout, this.config);
    },

    // Chart 4: Cross-Check Scatter Plot
    renderChart4(data) {
        if (!data || !data.points || data.points.length === 0) {
            document.getElementById('chart4').innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">Add more articles to enable cross-checking</p>';
            return;
        }

        const trace = {
            x: data.points.map(p => p.x),
            y: data.points.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            text: data.points.map(p => p.title),
            marker: {
                size: 10,
                color: data.points.map(p => p.y),
                colorscale: [
                    [0, '#FF5252'],
                    [0.5, '#FFA726'],
                    [1, '#00C853']
                ],
                showscale: true,
                colorbar: {
                    title: 'Credibility',
                    tickvals: [0, 0.5, 1],
                    ticktext: ['Low', 'Medium', 'High']
                }
            },
            hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f}<br>Credibility: %{y:.2f}<extra></extra>'
        };

        const layout = {
            ...this.layout,
            xaxis: { title: data.x_label || 'Similarity Score', range: [0, 1] },
            yaxis: { title: data.y_label || 'Credibility Score', range: [0, 1] }
        };

        Plotly.newPlot('chart4', [trace], layout, this.config);
    },

    // Chart 5: Similarity Map (from existing database)
    async renderChart5(articleId, chart5Data) {
        try {
            console.log('🔍 renderChart5 called with:', { articleId, chart5Data });
            
            const container = document.getElementById('chart5');
            if (!container) {
                console.error('Chart 5 container not found');
                return;
            }
            
            // If data is provided directly, use it
            let data = chart5Data;
            
            // If no data provided, try to get from articleId (for backward compatibility)
            if (!data && articleId) {
                console.log('No chart5Data provided, data should come from article response');
                container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">Loading similarity map...</p>';
                return;
            }
            
            if (!data || !data.points || data.points.length === 0) {
                container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">Add more articles to view similarity map</p>';
                return;
            }
            
            console.log('📊 Chart 5 data:', data);
            console.log('📊 Points count:', data.points.length);
            
            // Separate current article from other articles
            // Handle both boolean true and string "true" (in case of JSON conversion)
            const currentArticle = data.points.find(p => p.is_current === true || p.is_current === "true" || p.x === 1.0);
            const otherArticles = data.points.filter(p => {
                const isCurrent = p.is_current === true || p.is_current === "true" || p.x === 1.0;
                return !isCurrent;
            });
            
            console.log('🔍 Current article found:', currentArticle ? 'YES' : 'NO');
            if (currentArticle) {
                console.log('   Current article:', currentArticle.title, 'x:', currentArticle.x, 'is_current:', currentArticle.is_current);
            }
            
            const traces = [];
            
            // Add trace for other articles
            if (otherArticles.length > 0) {
                traces.push({
                    x: otherArticles.map(p => p.x),
                    y: otherArticles.map(p => p.y),
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Other Articles',
                    text: otherArticles.map(p => {
                        const source = p.source || 'Unknown';
                        return `${p.title || 'Untitled'}<br>Source: ${source}`;
                    }),
                    marker: {
                        size: 10,
                        color: otherArticles.map(p => p.y),  // Color by credibility
                        colorscale: [
                            [0, '#FF5252'],  // Red for low credibility
                            [0.5, '#FFA726'],  // Orange for medium
                            [1, '#00C853']  // Green for high credibility
                        ],
                        showscale: true,
                        colorbar: {
                            title: 'Credibility',
                            tickvals: [0, 0.5, 1],
                            ticktext: ['Low', 'Medium', 'High'],
                            x: 1.15
                        }
                    },
                    hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f}<br>Credibility: %{y:.2f}<extra></extra>'
                });
            }
            
            // Add trace for current article (highlighted)
            if (currentArticle) {
                traces.push({
                    x: [currentArticle.x],
                    y: [currentArticle.y],
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Your Article',
                    text: [`${currentArticle.title || 'Current Article'}<br>Source: ${currentArticle.source || 'Current'}`],
                    marker: {
                        size: 20,
                        color: '#FF4B4B',  // Red star for current article
                        symbol: 'star',
                        line: {
                            color: '#FFFFFF',
                            width: 2
                        }
                    },
                    hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f} (Your Article)<br>Credibility: %{y:.2f}<extra></extra>'
                });
            }
            
            const layout = {
                ...this.layout,
                xaxis: { 
                    title: 'Content Similarity to Your Article', 
                    range: [0, 1.1]  // Slightly extend to show current article at x=1.0
                },
                yaxis: { title: 'Credibility Score', range: [0, 1] },
                title: 'Similarity Map',
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(255, 255, 255, 0.8)'
                }
            };
            
            Plotly.newPlot('chart5', traces, layout, this.config);
            console.log('✅ Chart 5 rendered successfully');
        } catch (error) {
            console.error('❌ Error rendering chart 5:', error);
            document.getElementById('chart5').innerHTML = '<p style="padding: 2rem; text-align: center; color: #FF5252;">Error loading similarity map</p>';
        }
    },


    // Sensationalism Score Display - COMPLETELY REMOVED to prevent conflict with overall score
    // The overall score is now handled by displayScores() function
    // displayResults function removed entirely

    // Word frequency chart removed - not needed

    // Sentiment flow chart removed - not needed
};

// Add renderChart6 using function assignment (to avoid syntax parsing issues)
Charts.renderChart6 = function(data) {
        console.log('Rendering Chart 6 (Related Articles):', data);
        
        const container = document.getElementById('chart6');
        if (!container) {
            console.error('Chart 6 container not found');
            return;
        }

        // Debug: Check data structure
        // Backend returns chart6_data with 'articles' array
        const articles = (data && data.articles) || (data && data.related_articles) || [];
        console.log('🔍 Chart 6 Debug:', {
            data: data,
            hasRelatedArticles: data && data.related_articles,
            hasArticles: data && data.articles,
            articlesLength: articles.length,
            firstArticle: articles.length > 0 ? articles[0] : 'N/A'
        });

        try {
            if (!data || articles.length === 0) {
                container.innerHTML = `
                    <div class="placeholder-content">
                        <div class="placeholder-icon">📰</div>
                        <p>No related articles found</p>
                        <small>Try analyzing a different article</small>
                    </div>
                `;
                return;
            }

                // Create article cards
                const articlesHtml = articles.map(article => {
                const relevance = article.relevance || 0;
                const relevancePercent = Math.round(relevance * 100);
                
                console.log('📊 Article relevance debug:', {
                    title: article.title,
                    relevance: relevance,
                    relevancePercent: relevancePercent,
                    article: article
                });
                
                return `
                <div class="related-article-card">
                    <div class="article-header">
                        <h4 class="article-title">${article.title || 'Untitled'}</h4>
                        <span class="article-source">${article.source || 'Unknown Source'}</span>
                    </div>
                    <p class="article-summary">${article.snippet || article.summary || 'No summary available'}</p>
                    <div class="article-relevance">
                        <span class="relevance-label">Relevance:</span>
                        <div class="relevance-bar-container">
                            <div class="relevance-bar" style="width: ${relevancePercent}%; background: ${relevancePercent >= 70 ? '#00C853' : relevancePercent >= 40 ? '#FFA726' : '#FF4B4B'};"></div>
                        </div>
                        <span class="relevance-score">${relevancePercent}%</span>
                    </div>
                    <div class="article-meta">
                        <span class="article-date">${article.published_date || ''}</span>
                        <a href="${article.url || '#'}" target="_blank" class="article-link">Read Article →</a>
                    </div>
                </div>
            `;
            }).join('');

            container.innerHTML = `
                <div class="related-articles-container">
                    <h4>Related Articles from Reputable Sources</h4>
                    <div class="articles-grid">
                        ${articlesHtml}
                    </div>
                </div>
            `;

        } catch (error) {
            console.error('Error creating related articles chart:', error);
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #FF5252;">Chart creation error</p>';
        }
};
