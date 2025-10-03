// static/js/charts-display.js
// Module for rendering charts using Plotly

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
        if (!data) {
            document.getElementById('chart1').innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">No data available</p>';
            return;
        }

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

        Plotly.newPlot('chart1', [trace], layout, this.config);
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
    async renderChart5(articleId) {
        try {
            const result = await API.getSimilarityMap(articleId);
            
            if (!result.success || !result.data || result.data.length === 0) {
                document.getElementById('chart5').innerHTML = '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">Add more articles to view similarity map</p>';
                return;
            }

            // Separate user article from others
            const userArticle = result.data.find(d => d.type === 'user');
            const otherArticles = result.data.filter(d => d.type !== 'user');

            const traces = [];

            // Group by type
            const realNews = otherArticles.filter(d => d.type === 'real');
            const fakeNews = otherArticles.filter(d => d.type === 'fake');
            const mixedNews = otherArticles.filter(d => d.type === 'mixed');

            // Add traces for each category
            if (realNews.length > 0) {
                traces.push({
                    x: realNews.map(d => d.similarity),
                    y: realNews.map(d => d.credibility),
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Real News',
                    text: realNews.map(d => d.title),
                    marker: { size: 10, color: '#00C853' },
                    hovertemplate: '<b>%{text}</b><br>Source: ' + realNews.map(d => d.source).join('<br>') + '<extra></extra>'
                });
            }

            if (mixedNews.length > 0) {
                traces.push({
                    x: mixedNews.map(d => d.similarity),
                    y: mixedNews.map(d => d.credibility),
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Mixed',
                    text: mixedNews.map(d => d.title),
                    marker: { size: 10, color: '#FFA726' }
                });
            }

            if (fakeNews.length > 0) {
                traces.push({
                    x: fakeNews.map(d => d.similarity),
                    y: fakeNews.map(d => d.credibility),
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Fake News',
                    text: fakeNews.map(d => d.title),
                    marker: { size: 10, color: '#FF5252' }
                });
            }

            // Add user article
            if (userArticle) {
                traces.push({
                    x: [userArticle.similarity],
                    y: [userArticle.credibility],
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Your Article',
                    text: [userArticle.title],
                    marker: { size: 18, color: '#FF4B4B', symbol: 'star' }
                });
            }

            const layout = {
                ...this.layout,
                xaxis: { title: 'Similarity to Your Article', range: [0, 1] },
                yaxis: { title: 'Credibility Score', range: [0, 1] },
                showlegend: true,
                legend: { x: 0, y: 1 }
            };

            Plotly.newPlot('chart5', traces, layout, this.config);
        } catch (error) {
            console.error('Error rendering chart 5:', error);
            document.getElementById('chart5').innerHTML = '<p style="padding: 2rem; text-align: center; color: #FF5252;">Error loading similarity map</p>';
        }
    },

    // Chart 6: Placeholder (for future implementation)
    renderChart6() {
        // Placeholder is already in HTML - no need to render
        console.log('Chart 6 is a placeholder for future implementation');
    }
};
