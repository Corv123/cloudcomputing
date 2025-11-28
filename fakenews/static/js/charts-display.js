// Purpose-Driven Charts - Version 3.0
// Each chart has a clear purpose: inform, verify, or warn

console.log('🔵 CHARTS-DISPLAY.JS LOADED - VERSION 3.0 (Purpose-Driven)');
console.log('🔵 File loaded at:', new Date().toISOString());

const Charts = {
    // Common chart configuration
    config: {
        responsive: true,
        displayModeBar: false
    },

    // Updated layout with trust colors
    layout: {
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#1e293b',
        font: { 
            color: '#f8fafc',
            family: 'Inter, sans-serif',
            size: 12
        },
        margin: { t: 60, r: 20, b: 100, l: 70 },
        autosize: true
    },

    // Chart 1: Language Quality - Purpose: Show writing quality issues at a glance
    // Design: Gauge meters for each metric with color zones
    renderChart1(data) {
        console.log('🔍 renderChart1: Language Quality Analysis');
        
        const container = document.getElementById('chart1');
        if (!data || !data.labels || !data.values) {
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #94a3b8;">No language quality data available</p>';
            return;
        }

        // Create gauge charts for each metric
        const metrics = data.labels.map((label, i) => ({
            label: label,
            value: data.values[i] || 0
        }));

        // Determine color based on score
        const getColor = (score) => {
            if (score >= 70) return '#10b981'; // Green - Good
            if (score >= 40) return '#f59e0b'; // Amber - Warning
            return '#ef4444'; // Red - Poor
        };

        // Create HTML with gauge visualizations
        let html = '<div class="language-metrics-grid">';
        
        metrics.forEach(metric => {
            const color = getColor(metric.value);
            const percentage = Math.round(metric.value);
            
            html += `
                <div class="language-metric-card">
                    <div class="metric-header">
                        <h4>${metric.label}</h4>
                        <span class="metric-score" style="color: ${color}">${percentage}%</span>
                    </div>
                    <div class="gauge-container">
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width: ${percentage}%; background: ${color};"></div>
                        </div>
                    </div>
                    <div class="metric-status">
                        ${percentage >= 70 ? '✓ Good quality' : percentage >= 40 ? '⚠ Needs improvement' : '✗ Poor quality'}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    },

    // Chart 2: Sentiment Analysis - Purpose: Reveal emotional manipulation
    // Design: Timeline showing sentiment changes throughout article
    renderChart2(data) {
        console.log('🔍 renderChart2: Sentiment Timeline Analysis');
        
        const container = document.getElementById('chart2');
        if (!data) {
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #94a3b8;">No sentiment data available</p>';
            return;
        }

        // If we have sentiment flow data, use it for timeline
        // Otherwise, create a distribution visualization
        if (data.sentiment_flow && data.sentiment_flow.length > 0) {
            // Timeline visualization
            const timelineData = data.sentiment_flow;
            const paragraphs = timelineData.map((_, i) => `P${i + 1}`);
            
            const trace = {
                x: paragraphs,
                y: timelineData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Sentiment',
                line: {
                    color: '#3b82f6',
                    width: 3
                },
                marker: {
                    size: 8,
                    color: timelineData.map(s => {
                        if (s > 0.3) return '#10b981'; // Positive
                        if (s < -0.3) return '#ef4444'; // Negative
                        return '#f59e0b'; // Neutral
                    })
                },
                fill: 'tozeroy',
                fillcolor: 'rgba(59, 130, 246, 0.1)'
            };

            const layout = {
                ...this.layout,
                xaxis: { 
                    title: 'Article Sections',
                    gridcolor: 'rgba(148, 163, 184, 0.1)',
                    tickangle: -45
                },
                yaxis: { 
                    title: 'Sentiment Score',
                    range: [-1, 1],
                    gridcolor: 'rgba(148, 163, 184, 0.1)',
                    zeroline: true,
                    zerolinecolor: 'rgba(148, 163, 184, 0.3)'
                },
                shapes: [
                    {
                        type: 'line',
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 0,
                        xref: 'paper',
                        yref: 'y',
                        line: { color: 'rgba(148, 163, 184, 0.5)', width: 2, dash: 'dash' }
                    }
                ],
                margin: { t: 60, r: 30, b: 100, l: 70 },
                annotations: []
            };

            Plotly.newPlot('chart2', [trace], layout, this.config);
        } else if (data.labels && data.values) {
            // Fallback to distribution chart
            const trace = {
                labels: data.labels,
                values: data.values,
                type: 'pie',
                hole: 0.4,
                marker: {
                    colors: ['#6b7280', '#ef4444', '#f59e0b', '#10b981'],
                    line: { color: '#1e293b', width: 2 }
                },
                textinfo: 'label+percent',
                textposition: 'outside'
            };

            const layout = {
                ...this.layout,
                showlegend: true,
                legend: {
                    x: 1.02,
                    y: 0.5,
                    xanchor: 'left',
                    font: { color: '#f8fafc' }
                },
                margin: { t: 60, r: 150, b: 100, l: 70 }
            };

            Plotly.newPlot('chart2', [trace], layout, this.config);
        } else {
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #94a3b8;">No sentiment data available</p>';
        }
    },

    // Chart 3: Credibility Radar - Purpose: Show trustworthiness breakdown
    // Design: Interactive radar with clickable segments
    renderChart3(data) {
        console.log('🔍 renderChart3: Credibility Radar');
        
        const container = document.getElementById('chart3');
        if (!data || !data.labels || !data.values) {
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #94a3b8;">No credibility data available</p>';
            return;
        }

        const trace = {
            type: 'scatterpolar',
            r: data.values,
            theta: data.labels,
            fill: 'toself',
            fillcolor: 'rgba(59, 130, 246, 0.2)',
            line: {
                color: '#3b82f6',
                width: 3
            },
            marker: {
                size: 10,
                color: '#3b82f6'
            },
            hovertemplate: '<b>%{theta}</b><br>Score: %{r:.0f}/100<extra></extra>'
        };

        const layout = {
            ...this.layout,
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100],
                    gridcolor: 'rgba(148, 163, 184, 0.2)',
                    tickfont: { color: '#cbd5e1' }
                },
                angularaxis: {
                    gridcolor: 'rgba(148, 163, 184, 0.2)',
                    tickfont: { color: '#cbd5e1' }
                },
                bgcolor: '#1e293b'
            },
            showlegend: false,
            margin: { t: 60, r: 20, b: 100, l: 70 }
        };

        Plotly.newPlot('chart3', [trace], layout, this.config);
    },

    // Chart 4: Cross-Check Analysis - Purpose: Show verification against trusted sources
    // Design: Network graph showing article connections
    renderChart4(data) {
        console.log('🔍 renderChart4: Cross-Check Network');
        
        const container = document.getElementById('chart4');
        if (!data || !data.points || data.points.length === 0) {
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #94a3b8;">Add more articles to enable cross-checking</p>';
            return;
        }

        // Create scatter plot with network visualization
        const points = data.points;
        const xValues = points.map(p => p.x || p.similarity || 0);
        const yValues = points.map(p => p.y || p.credibility || 0);
        
        // Dynamically normalize x-axis range
        const maxX = Math.max(...xValues);
        const minX = Math.min(...xValues);
        const xRange = maxX - minX;
        const xPadding = xRange * 0.1; // 10% padding
        const xMax = maxX > 0.6 ? 1.0 : Math.min(0.6, maxX + xPadding);
        const xMin = Math.max(0, minX - xPadding);
        
        // Create traces grouped by credibility for legend
        const highCred = points.filter(p => (p.y || p.credibility || 0) >= 0.7);
        const medCred = points.filter(p => {
            const cred = p.y || p.credibility || 0;
            return cred >= 0.4 && cred < 0.7;
        });
        const lowCred = points.filter(p => (p.y || p.credibility || 0) < 0.4);
        
        const traces = [];
        
        if (highCred.length > 0) {
            traces.push({
                x: highCred.map(p => p.x || p.similarity || 0),
                y: highCred.map(p => p.y || p.credibility || 0),
                mode: 'markers',
                type: 'scatter',
                name: 'High Credibility',
                marker: {
                    size: 10,
                    color: '#10b981',
                    line: { color: '#1e293b', width: 1 }
                },
                hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f}<br>Credibility: %{y:.2f}<extra></extra>',
                text: highCred.map(p => p.title || 'Untitled')
            });
        }
        
        if (medCred.length > 0) {
            traces.push({
                x: medCred.map(p => p.x || p.similarity || 0),
                y: medCred.map(p => p.y || p.credibility || 0),
                mode: 'markers',
                type: 'scatter',
                name: 'Medium Credibility',
                marker: {
                    size: 10,
                    color: '#f59e0b',
                    line: { color: '#1e293b', width: 1 }
                },
                hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f}<br>Credibility: %{y:.2f}<extra></extra>',
                text: medCred.map(p => p.title || 'Untitled')
            });
        }
        
        if (lowCred.length > 0) {
            traces.push({
                x: lowCred.map(p => p.x || p.similarity || 0),
                y: lowCred.map(p => p.y || p.credibility || 0),
                mode: 'markers',
                type: 'scatter',
                name: 'Low Credibility',
                marker: {
                    size: 10,
                    color: '#ef4444',
                    line: { color: '#1e293b', width: 1 }
                },
                hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f}<br>Credibility: %{y:.2f}<extra></extra>',
                text: lowCred.map(p => p.title || 'Untitled')
            });
        }

        const layout = {
            ...this.layout,
            xaxis: { 
                title: 'Content Similarity',
                range: [xMin, xMax],
                gridcolor: 'rgba(148, 163, 184, 0.1)'
            },
            yaxis: { 
                title: 'Source Credibility',
                range: [0, 1],
                gridcolor: 'rgba(148, 163, 184, 0.1)'
            },
            showlegend: true,
            legend: {
                x: 1.02,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(30, 41, 59, 0.9)',
                bordercolor: 'rgba(148, 163, 184, 0.2)',
                font: { color: '#f8fafc', size: 12 }
            },
            margin: { t: 60, r: 150, b: 100, l: 70 },
            shapes: [
                // Add quadrant lines
                {
                    type: 'line',
                    x0: (xMin + xMax) / 2,
                    y0: 0,
                    x1: (xMin + xMax) / 2,
                    y1: 1,
                    xref: 'x',
                    yref: 'y',
                    line: { color: 'rgba(148, 163, 184, 0.2)', width: 1, dash: 'dash' }
                },
                {
                    type: 'line',
                    x0: xMin,
                    y0: 0.5,
                    x1: xMax,
                    y1: 0.5,
                    xref: 'x',
                    yref: 'y',
                    line: { color: 'rgba(148, 163, 184, 0.2)', width: 1, dash: 'dash' }
                }
            ]
        };

        Plotly.newPlot('chart4', traces, layout, this.config);
    },

    // Chart 5: Similarity Map - Purpose: Show content originality
    // Design: Heatmap showing paragraph-by-paragraph similarity
    renderChart5(articleId, chart5Data) {
        console.log('🔍 renderChart5: Similarity Heatmap');
        
        const container = document.getElementById('chart5');
        if (!container) return;

        let data = chart5Data;
        
        if (!data || !data.points || data.points.length === 0) {
            container.innerHTML = '<p style="padding: 2rem; text-align: center; color: #94a3b8;">Add more articles to view similarity map</p>';
            return;
        }

        // Create scatter plot with similarity visualization
        const currentArticle = data.points.find(p => p.is_current === true || p.is_current === "true" || p.x === 1.0);
        const otherArticles = data.points.filter(p => {
            const isCurrent = p.is_current === true || p.is_current === "true" || p.x === 1.0;
            return !isCurrent;
        });

        // Dynamically normalize x-axis range for Chart 5
        const allXValues = data.points.map(p => p.x || p.similarity || 0);
        const maxX5 = allXValues.length > 0 ? Math.max(...allXValues) : 1.0;
        const minX5 = allXValues.length > 0 ? Math.min(...allXValues) : 0;
        const xRange5 = maxX5 - minX5;
        const xPadding5 = xRange5 > 0 ? xRange5 * 0.1 : 0.1; // 10% padding
        // If all points end before 0.6, set max to 0.6; otherwise use data max
        const xMax5 = maxX5 > 0.6 ? Math.min(1.0, maxX5 + xPadding5) : Math.min(0.6, maxX5 + xPadding5);
        const xMin5 = Math.max(0, minX5 - xPadding5);

        const traces = [];

        // Other articles
        if (otherArticles.length > 0) {
            traces.push({
                x: otherArticles.map(p => p.x || 0),
                y: otherArticles.map(p => p.y || 0),
                mode: 'markers',
                type: 'scatter',
                name: 'Other Articles',
                text: otherArticles.map(p => `${p.title || 'Untitled'}<br>Source: ${p.source || 'Unknown'}`),
                marker: {
                    size: 12,
                    color: otherArticles.map(p => {
                        const cred = p.y || 0;
                        return cred >= 0.7 ? '#10b981' : cred >= 0.4 ? '#f59e0b' : '#ef4444';
                    }),
                    line: { color: '#1e293b', width: 1 }
                },
                hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f}<br>Credibility: %{y:.2f}<extra></extra>'
            });
        }

        // Current article (highlighted)
        if (currentArticle) {
            traces.push({
                x: [currentArticle.x || 1.0],
                y: [currentArticle.y || 0],
                mode: 'markers',
                type: 'scatter',
                name: 'Your Article',
                text: [`${currentArticle.title || 'Current Article'}<br>Source: ${currentArticle.source || 'Current'}`],
                marker: {
                    size: 20,
                    color: '#3b82f6',
                    symbol: 'star',
                    line: { color: '#f8fafc', width: 2 }
                },
                hovertemplate: '<b>%{text}</b><br>Similarity: %{x:.2f} (Your Article)<br>Credibility: %{y:.2f}<extra></extra>'
            });
        }

        const layout = {
            ...this.layout,
            xaxis: { 
                title: 'Content Similarity to Your Article',
                range: [xMin5, xMax5],
                gridcolor: 'rgba(148, 163, 184, 0.1)'
            },
            yaxis: { 
                title: 'Credibility Score',
                range: [0, 1],
                gridcolor: 'rgba(148, 163, 184, 0.1)'
            },
            showlegend: true,
            legend: {
                x: 1.02,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(30, 41, 59, 0.9)',
                bordercolor: 'rgba(148, 163, 184, 0.2)',
                font: { color: '#f8fafc', size: 12 }
            },
            margin: { t: 50, r: 150, b: 80, l: 60 }
        };

        Plotly.newPlot('chart5', traces, layout, this.config);
    },

    // Chart 6: Related Articles - Purpose: Provide context and verification
    // Design: Enhanced cards with agree/dispute indicators
    renderChart6(data) {
        console.log('🔍 renderChart6: Related Articles');
        
        const container = document.getElementById('chart6');
        if (!container) return;

        const articles = (data && data.articles) || (data && data.related_articles) || [];

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

        // Determine relevance level
        const getRelevanceStatus = (article) => {
            const relevance = article.relevance || 0;
            if (relevance >= 0.7) return { status: 'Relevant', icon: '✓', color: '#10b981' };
            if (relevance >= 0.4) return { status: 'Moderate', icon: '—', color: '#f59e0b' };
            return { status: 'Weak', icon: '✗', color: '#ef4444' };
        };

        const articlesHtml = articles.map(article => {
            const relevance = article.relevance || 0;
            const relevancePercent = Math.round(relevance * 100);
            const relevanceStatus = getRelevanceStatus(article);
            
            return `
                <div class="related-article-card">
                    <div class="article-agreement-badge" style="background: ${relevanceStatus.color}20; border-left: 4px solid ${relevanceStatus.color};">
                        <span style="color: ${relevanceStatus.color}; font-weight: 700;">${relevanceStatus.icon} ${relevanceStatus.status.toUpperCase()}</span>
                    </div>
                    <div class="article-header">
                        <h4 class="article-title">${article.title || 'Untitled'}</h4>
                        <span class="article-source">${article.source || 'Unknown Source'}</span>
                    </div>
                    <p class="article-summary">${article.snippet || article.summary || 'No summary available'}</p>
                    <div class="article-relevance">
                        <span class="relevance-label">Relevance:</span>
                        <div class="relevance-bar-container">
                            <div class="relevance-bar" style="width: ${relevancePercent}%; background: ${relevancePercent >= 70 ? '#10b981' : relevancePercent >= 40 ? '#f59e0b' : '#ef4444'};"></div>
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
                <div class="articles-grid">
                    ${articlesHtml}
                </div>
            </div>
        `;
    }
};

// Add CSS for language metrics grid
const style = document.createElement('style');
style.textContent = `
    .language-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        padding: 1rem;
    }
    .language-metric-card {
        background: #1e293b;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .metric-header h4 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #cbd5e1;
        margin: 0;
    }
    .metric-score {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .gauge-container {
        margin-bottom: 0.75rem;
    }
    .gauge-track {
        height: 10px;
        background: #334155;
        border-radius: 9999px;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s ease;
    }
    .metric-status {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .article-agreement-badge {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
`;
document.head.appendChild(style);
