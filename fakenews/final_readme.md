# 🔍 Fake News Detector - Modular Dark Theme

A modular, extensible web application for analyzing news article credibility with a modern dark interface.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ Features

- 🎨 **Modern Dark Theme** - Streamlit-inspired UI
- 📊 **6 Analysis Charts** - Comprehensive visual analysis
- 🔧 **Modular Architecture** - Easy to extend and customize
- 🚀 **Fast Analysis** - Real-time article credibility scoring
- 💾 **SQLite Database** - Persistent storage of analyzed articles
- 📈 **Similarity Mapping** - Compare articles against database
- 🌐 **REST API** - Clean separation of frontend/backend

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── database.py                     # Database operations
├── requirements.txt                # Python dependencies
│
├── analyzers/                      # Analysis modules
│   ├── __init__.py
│   ├── language_analyzer.py        # Language quality (Chart 1 & 2)
│   ├── credibility_analyzer.py     # Source credibility (Chart 3)
│   └── crosscheck_analyzer.py      # Cross-checking (Chart 4)
│
├── extractors/                     # Article extraction
│   ├── __init__.py
│   └── article_extractor.py        # URL content extraction
│
└── static/                         # Frontend files
    ├── index.html                  # Main HTML page
    ├── css/
    │   └── styles.css             # Dark theme styles
    └── js/
        ├── api.js                 # API communication
        ├── main.js                # Main application logic
        ├── tabs.js                # Tab management
        ├── article-display.js     # Article info display
        ├── scores-display.js      # Scores rendering
        └── charts-display.js      # Chart visualizations
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Create project folder:**
```bash
mkdir fake-news-detector
cd fake-news-detector
```

2. **Create required folders:**
```bash
mkdir -p analyzers extractors static/css static/js
```

3. **Copy all files** to their respective locations (see structure above)

4. **Create `__init__.py` files:**
```bash
touch analyzers/__init__.py
touch extractors/__init__.py
```

5. **Install dependencies:**
```bash
pip install -r requirements.txt
```

6. **Run the application:**
```bash
python app.py
```

7. **Open browser:**
```
http://localhost:5000
```

---

## 📊 How It Works

### Workflow

```
User Input (URL)
    ↓
Check Database (Cached?)
    ↓
Extract Article (Web Scraping)
    ↓
Parallel Analysis:
├── Language Analyzer → Chart 1 & 2
├── Credibility Analyzer → Chart 3
└── CrossCheck Analyzer → Chart 4
    ↓
Calculate Overall Score (Weighted Average)
    ↓
Generate Chart 5 (Similarity Map)
    ↓
Store in Database
    ↓
Display Results
```

### Scoring System

**Overall Score** = Weighted combination of:
- **Language Quality (35%)**: Grammar, style, complexity
- **Source Credibility (40%)**: Domain reputation, transparency
- **Cross-Check (25%)**: Similarity to verified articles

**Category Assignment:**
- 70-100%: **REAL** (High Credibility) 🟢
- 40-69%: **MIXED** (Medium Credibility) 🟡
- 0-39%: **FAKE** (Low Credibility) 🔴

---

## 📈 Charts Explained

### Chart 1: Language Quality Metrics
**Type:** Bar Chart  
**Shows:** Capitalization, Punctuation, Complexity, Grammar scores  
**Purpose:** Identify linguistic red flags

### Chart 2: Sentiment Distribution
**Type:** Pie Chart  
**Shows:** Neutral, Sensational, Negative, Positive tone percentages  
**Purpose:** Detect emotional manipulation

### Chart 3: Credibility Radar (Domain-Based Analysis)
**Type:** Radar Chart  
**Shows:** 4 credibility dimensions scored 0-100
- Domain Age: Reputation and establishment of domain
- URL Structure: Professional patterns and TLD quality  
- Site Structure: Standard pages and navigation presence
- Content Format: Professional formatting quality

**Purpose:** Multi-dimensional domain credibility assessment  
**Known Source Database:** Automatically recognizes 50+ reliable sources (Reuters, BBC, NYT, etc.) and problematic sources (conspiracy sites, misinformation domains)

**Note:** This chart drives 50% of the overall credibility score

### Chart 4: Cross-Check Analysis
**Type:** Scatter Plot  
**Shows:** Similar articles by similarity vs credibility  
**Purpose:** Compare against verified sources

### Chart 5: Similarity Map
**Type:** Scatter Plot  
**Shows:** Your article position among database articles  
**Purpose:** Pattern recognition and clustering

### Chart 6: Placeholder
**Type:** Custom (Your Implementation)  
**Shows:** Your custom analysis  
**Purpose:** Extensibility for future features

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Scoring weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "language": 0.35,
    "credibility": 0.40,
    "crosscheck": 0.25
}

# Add trusted domains
CREDIBLE_DOMAINS = [
    "bbc.com", "reuters.com", ...
]

# Add suspicious domains
SUSPICIOUS_DOMAINS = [
    "fake-news-site.com", ...
]

# Optional: NewsAPI key for better extraction
NEWSAPI_KEY = "your_key_here"
```

---

## 🔧 Extending the System

### Adding a New Chart

See **DEVELOPER_GUIDE.md** for detailed instructions.

**Quick Steps:**
1. Create analyzer in `analyzers/your_analyzer.py`
2. Import in `app.py`
3. Add rendering in `static/js/charts-display.js`
4. Call in `static/js/main.js`

**Example:**
```python
# analyzers/sentiment_analyzer.py
class SentimentAnalyzer:
    def analyze(self, content):
        return {
            "score": 0.8,
            "chart6_data": {"labels": [...], "values": [...]}
        }
```

---

## 🎨 UI Customization

### Color Theme

Edit `static/css/styles.css`:

```css
:root {
    --bg-primary: #0E1117;      /* Main background */
    --accent: #FF4B4B;          /* Primary color */
    --success: #00C853;         /* High credibility */
    --warning: #FFA726;         /* Medium */
    --danger: #FF5252;          /* Low */
}
```

---

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main HTML page |
| `/api/analyze` | POST | Analyze article from URL |
| `/api/articles` | GET | Get all articles |
| `/api/stats` | GET | Get database statistics |
| `/api/similarity-map/:id` | GET | Get similarity data |
| `/api/clear-database` | POST | Clear all articles |

### Example Request

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bbc.com/news/article"}'
```

### Example Response

```json
{
    "success": true,
    "article": {
        "id": 1,
        "title": "Article Title",
        "overall_score": 0.76,
        "language_score": 0.75,
        "credibility_score": 0.80,
        "cross_check_score": 0.70,
        "chart1_data": {...},
        "chart2_data": {...},
        ...
    }
}
```

---

## 🧪 Testing

### Test Backend Analyzer

```python
from analyzers.language_analyzer import LanguageAnalyzer

analyzer = LanguageAnalyzer()
result = analyzer.analyze("Sample Title", "Sample content...")
print(result)
```

### Test API

```bash
# Start server
python app.py

# Test in another terminal
curl http://localhost:5000/api/stats
```

### Test Frontend

1. Open browser to `http://localhost:5000`
2. Open DevTools (F12)
3. Check Console for errors
4. Check Network tab for API calls

---

## 📦 Dependencies

```
flask==3.0.0
flask-cors==4.0.0
requests==2.31.0
beautifulsoup4==4.12.3
scikit-learn==1.4.0
numpy==1.26.3
lxml==5.1.0
```

---

## 🐛 Troubleshooting

### CSS not loading
- Check folder structure: `static/css/styles.css`
- Hard refresh: `Ctrl+Shift+R`
- Check browser console for 404 errors

### Charts not displaying
- Ensure Plotly CDN is accessible
- Check `charts-display.js` for errors
- Verify chart data format

### Database errors
- Delete `articles.db` and restart
- Check write permissions

### Analysis fails
- Verify URL is accessible
- Check internet connection
- Try with different article URL

---

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production (AWS Lambda)
See deployment guide for serverless setup.

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

---

## 📄 File Checklist

Before running, ensure you have:

**Root Directory:**
- [ ] app.py
- [ ] config.py
- [ ] database.py
- [ ] requirements.txt

**analyzers/ folder:**
- [ ] \_\_init\_\_.py
- [ ] language_analyzer.py
- [ ] credibility_analyzer.py
- [ ] crosscheck_analyzer.py

**extractors/ folder:**
- [ ] \_\_init\_\_.py
- [ ] article_extractor.py

**static/ folder:**
- [ ] index.html
- [ ] css/styles.css
- [ ] js/api.js
- [ ] js/main.js
- [ ] js/tabs.js
- [ ] js/article-display.js
- [ ] js/scores-display.js
- [ ] js/charts-display.js

---

## 🤝 Contributing

This project is designed to be extended by team members. See **DEVELOPER_GUIDE.md** for:
- Adding new charts
- Creating custom analyzers
- Modifying UI components
- Database schema changes

---

## 📝 License

MIT License - Feel free to use for educational purposes.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It provides screening assistance but should not be considered a definitive fact-checker. Always verify important information through multiple credible sources.

---

## 📞 Support

For issues or questions:
1. Check **DEVELOPER_GUIDE.md**
2. Review code comments
3. Check browser console for errors
4. Verify all files are in correct locations

---

**Built with ❤️ for fake news detection**

Version 2.0 - Modular Architecture