# 🔧 Developer Guide - Adding New Features

This guide explains how to extend the Fake News Detector with new charts and analysis modules.

---

## 📊 Project Structure

```
fake-news-detector/
├── app.py                          # Main Flask app
├── config.py                       # Configuration (scoring weights, source lists)
├── database.py                     # Database operations
│
├── analyzers/                      # Scoring modules
│   ├── language_analyzer.py        # Language score (Chart 1 & 2)
│   ├── credibility_analyzer.py     # DOMAIN-BASED credibility (Chart 3) ⭐
│   └── crosscheck_analyzer.py      # Cross-check score (Chart 4)
│
├── extractors/                     # Article extraction
│   └── article_extractor.py        # Extract from URLs
│
└── static/                         # Frontend
    ├── index.html                  # Main page
    ├── css/styles.css             # Dark theme styles
    └── js/
        ├── api.js                 # API communication
        ├── main.js                # Main app logic
        ├── tabs.js                # Tab management
        ├── article-display.js     # Article info display
        ├── scores-display.js      # Scores display
        └── charts-display.js      # Chart rendering (Chart 3 is radar)
```

**⭐ Key Note:** `credibility_analyzer.py` uses a domain-based scoring system that automatically recognizes 50+ known reliable and problematic sources. This drives Chart 3 and accounts for 50% of the overall score.

---

## 🎨 How to Add a New Chart (Chart 6)

###Step 1: Create Python Analyzer (if needed)

If your chart requires new analysis, create a new file in `analyzers/`:

**Example:** `analyzers/sentiment_analyzer.py`

```python
# analyzers/sentiment_analyzer.py

class SentimentAnalyzer:
    """Analyzes article sentiment"""
    
    def analyze(self, content: str) -> dict:
        """
        Analyze sentiment
        
        Returns:
            dict with 'score' and 'chart6_data'
        """
        # Your analysis logic here
        sentiment_score = self._calculate_sentiment(content)
        
        chart6_data = {
            "labels": ["Positive", "Neutral", "Negative"],
            "values": [30, 50, 20]  # Your calculated values
        }
        
        return {
            "score": sentiment_score,
            "chart6_data": chart6_data
        }
    
    def _calculate_sentiment(self, content: str) -> float:
        # Your calculation logic
        return 0.75
```

### Step 2: Import Analyzer in app.py

Add your analyzer to `app.py`:

```python
# In app.py

from analyzers.sentiment_analyzer import SentimentAnalyzer

# Initialize
sentiment_analyzer = SentimentAnalyzer()

# In analyze_article() function, add:
sentiment_result = sentiment_analyzer.analyze(
    article_data['content']
)

# Add to article_data:
article_data.update({
    'chart6_data': sentiment_result['chart6_data']
})
```

### Step 3: Add Chart Rendering Function

In `static/js/charts-display.js`, replace the placeholder function:

```javascript
// In Charts object

renderChart6(data) {
    if (!data) {
        document.getElementById('chart6').innerHTML = 
            '<p style="padding: 2rem; text-align: center; color: #A0A0A0;">No data available</p>';
        return;
    }

    // Example: Bar chart
    const trace = {
        x: data.labels,
        y: data.values,
        type: 'bar',
        marker: {
            color: '#FF4B4B'
        }
    };

    const layout = {
        ...this.layout,
        title: 'Your Chart Title',
        xaxis: { title: 'Categories' },
        yaxis: { title: 'Values' }
    };

    Plotly.newPlot('chart6', [trace], layout, this.config);
}
```

### Step 4: Call Rendering in main.js

In `static/js/main.js`, add to the `analyzeArticle()` function:

```javascript
// After other chart renders
if (result.article.chart6_data) {
    Charts.renderChart6(result.article.chart6_data);
}
```

### Step 5: Update HTML (if needed)

The placeholder is already in `index.html`. Just update the title:

```html
<div class="chart-container">
    <h4>Chart 6: Your Custom Analysis</h4>
    <div id="chart6" class="chart-plot"></div>
</div>
```

### Step 6: Enable in config.py

```python
# In config.py

CHART_SETTINGS = {
    # ... other charts ...
    "chart6": {
        "enabled": True,  # Change to True
        "title": "Your Chart Title",
        "type": "bar"  # or "pie", "scatter", etc.
    }
}
```

---

## 📊 Chart Types Available (Plotly)

### Bar Chart
```javascript
{
    x: ['A', 'B', 'C'],
    y: [10, 20, 30],
    type: 'bar',
    marker: { color: '#FF4B4B' }
}
```

### Pie Chart
```javascript
{
    labels: ['A', 'B', 'C'],
    values: [10, 20, 30],
    type: 'pie',
    marker: { colors: ['#FF4B4B', '#00C853', '#FFA726'] }
}
```

### Line Chart
```javascript
{
    x: [1, 2, 3, 4],
    y: [10, 15, 13, 17],
    type: 'scatter',
    mode: 'lines',
    line: { color: '#FF4B4B' }
}
```

### Scatter Plot
```javascript
{
    x: [1, 2, 3],
    y: [4, 5, 6],
    mode: 'markers',
    type: 'scatter',
    marker: { size: 10, color: '#FF4B4B' }
}
```

### Radar Chart
```javascript
{
    type: 'scatterpolar',
    r: [80, 70, 90, 85, 75],
    theta: ['A', 'B', 'C', 'D', 'E'],
    fill: 'toself'
}
```

---

## 🔧 Adding a New Score Component

**Important Context:** The current system has Chart 3 (credibility) weighted at 50% of the overall score because it uses a domain-based analysis system that automatically recognizes known reliable and problematic sources. This was integrated from a team member's comprehensive credibility scorer.

If you want to add a 4th main score (e.g., "Fact-Check Score"):

### Step 1: Create Analyzer Module

**File:** `analyzers/factcheck_analyzer.py`

```python
class FactCheckAnalyzer:
    def analyze(self, content: str) -> dict:
        # Your logic
        return {
            "score": 0.80,
            "chart7_data": {...}  # If you need a chart
        }
```

### Step 2: Consider Weight Distribution

The current weights heavily favor credibility (Chart 3) at 50%. You have two options:

**Option A: Keep credibility at 50% (recommended)**
```python
SCORE_WEIGHTS = {
    "language": 0.20,      # Reduced from 0.25
    "credibility": 0.50,   # Keep high - domain analysis is comprehensive
    "crosscheck": 0.15,    # Reduced from 0.25
    "factcheck": 0.15      # NEW
}
```

**Option B: Rebalance everything**
```python
SCORE_WEIGHTS = {
    "language": 0.25,
    "credibility": 0.35,   # Reduced but still highest
    "crosscheck": 0.20,
    "factcheck": 0.20      # NEW
}
```

### Step 3: Update app.py

```python
from analyzers.factcheck_analyzer import FactCheckAnalyzer

factcheck_analyzer = FactCheckAnalyzer()

# In analyze_article():
factcheck_result = factcheck_analyzer.analyze(article_data['content'])

overall_score = (
    language_result['score'] * weights['language'] +
    credibility_result['score'] * weights['credibility'] +
    crosscheck_result['score'] * weights['crosscheck'] +
    factcheck_result['score'] * weights['factcheck']  # NEW
)
```

### Step 4: Update HTML

In `index.html`, add a new score card:

```html
<div class="score-card">
    <div class="score-card-header">Fact-Check Score</div>
    <div class="score-card-value" id="factcheckScore">0%</div>
    <div class="score-card-desc">Verified claims</div>
</div>
```

### Step 5: Update JavaScript

In `static/js/scores-display.js`:

```javascript
document.getElementById('factcheckScore').textContent = 
    Math.round(article.factcheck_score * 100) + '%';
```

---

## 💾 Database Modifications

If you need to store additional data:

### Step 1: Update database.py

```python
# In init_database():
cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        -- ... existing columns ...
        new_field TEXT,
        new_score REAL
    )
""")
```

### Step 2: Update insert_article()

```python
cursor.execute("""
    INSERT INTO articles (..., new_field, new_score)
    VALUES (..., ?, ?)
""", (..., article_data.get('new_field'), article_data.get('new_score')))
```

---

## 🎨 Styling Guide

### Color Palette (Dark Theme)

```css
--bg-primary: #0E1117;      /* Main background */
--bg-secondary: #262730;    /* Cards, sidebar */
--bg-tertiary: #1E1E1E;     /* Inputs, hover */
--border-color: #3D3D3D;    /* Borders */
--text-primary: #FAFAFA;    /* Main text */
--text-secondary: #A0A0A0;  /* Secondary text */
--accent: #FF4B4B;          /* Primary accent */
--success: #00C853;         /* Success/high */
--warning: #FFA726;         /* Warning/medium */
--danger: #FF5252;          /* Danger/low */
```

### Adding Custom Styles

Create `static/css/custom.css` and add to HTML:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/custom.css') }}">
```

---

## 🧪 Testing Your Changes

### 1. Test Backend

```python
# test_analyzer.py

from analyzers.your_analyzer import YourAnalyzer

analyzer = YourAnalyzer()
result = analyzer.analyze("test content")
print(result)
```

### 2. Test API Endpoint

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### 3. Test Frontend

Open browser console (`F12`) and check:
- Network tab for API calls
- Console for JavaScript errors
- Elements tab for HTML structure

---

## 📚 Common Patterns

### Pattern 1: Multi-Source Chart

Combine data from multiple analyzers:

```python
# In app.py
combined_data = {
    "chart6_data": {
        "language": language_result['score'],
        "credibility": credibility_result['score'],
        "crosscheck": crosscheck_result['score']
    }
}
```

### Pattern 2: Time-Series Chart

Store historical data:

```python
# Add to database
article_data['analysis_history'] = json.dumps([
    {"date": "2025-01-01", "score": 0.75},
    {"date": "2025-01-02", "score": 0.80}
])
```

### Pattern 3: Comparison Chart

Compare user article with database:

```python
database_avg = sum(a['credibility_score'] for a in database_articles) / len(database_articles)

chart_data = {
    "labels": ["Your Article", "Database Average"],
    "values": [user_score, database_avg]
}
```

---

## 🚀 Deployment Checklist

Before handing off to another developer:

- [ ] All analyzers have docstrings
- [ ] Config.py has clear comments
- [ ] Chart placeholders are clearly marked
- [ ] README.md is updated
- [ ] Test with sample URLs
- [ ] Database schema is documented
- [ ] API endpoints are documented
- [ ] JavaScript modules are commented
- [ ] Error handling is implemented
- [ ] Loading states are shown

---

## 🐛 Debugging Tips

### Backend Issues

```python
# Add logging in app.py
import logging
logging.basicConfig(level=logging.DEBUG)
app.logger.debug(f"Analysis result: {result}")
```

### Frontend Issues

```javascript
// Add console logs
console.log('Chart data:', data);
console.log('API response:', result);
```

### Database Issues

```python
# Check database directly
import sqlite3
conn = sqlite3.connect('articles.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM articles")
print(cursor.fetchall())
```

---

## 📞 API Reference

### POST /api/analyze
**Request:**
```json
{
    "url": "https://example.com/article"
}
```

**Response:**
```json
{
    "success": true,
    "article": {
        "id": 1,
        "title": "Article Title",
        "language_score": 0.75,
        "credibility_score": 0.80,
        "cross_check_score": 0.70,
        "overall_score": 0.76,
        "chart1_data": {...},
        "chart2_data": {...},
        ...
    }
}
```

### GET /api/articles
Returns all articles in database.

### GET /api/stats
Returns database statistics.

### GET /api/similarity-map/:id
Returns similarity data for article.

### POST /api/clear-database
Clears entire database.

---

## 🎓 Learning Resources

- **Plotly Documentation**: https://plotly.com/javascript/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **scikit-learn**: https://scikit-learn.org/stable/
- **BeautifulSoup**: https://www.crummy.com/software/BeautifulSoup/

---

## ✅ Quick Start for New Developers

1. **Clone/Download project**
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run server:** `python app.py`
4. **Open browser:** `http://localhost:5000`
5. **Read this guide**
6. **Make your changes**
7. **Test thoroughly**
8. **Document your additions**

---

## 💡 Example: Adding Chart 6 (Sentiment Analysis)

Full working example in `examples/sentiment_chart_example.md`

---

**Good luck with your development! 🚀**

For questions or issues, check the main README.md or review existing code for patterns.