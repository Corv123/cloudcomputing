# ✅ Setup Checklist

Follow this checklist to set up the Fake News Detector project correctly.

---

## 📁 Step 1: Create Folder Structure

```bash
mkdir fake-news-detector
cd fake-news-detector
mkdir -p analyzers extractors static/css static/js
```

**Verify:**
```
fake-news-detector/
├── analyzers/
├── extractors/
└── static/
    ├── css/
    └── js/
```

---

## 📄 Step 2: Copy Files

### Root Directory (4 files)
- [ ] app.py
- [ ] config.py
- [ ] database.py
- [ ] requirements.txt

### analyzers/ (4 files)
- [ ] \_\_init\_\_.py (empty file)
- [ ] language_analyzer.py
- [ ] credibility_analyzer.py
- [ ] crosscheck_analyzer.py

### extractors/ (2 files)
- [ ] \_\_init\_\_.py (empty file)
- [ ] article_extractor.py

### static/ (1 file)
- [ ] index.html

### static/css/ (1 file)
- [ ] styles.css

### static/js/ (6 files)
- [ ] api.js
- [ ] main.js
- [ ] tabs.js
- [ ] article-display.js
- [ ] scores-display.js
- [ ] charts-display.js

**Total: 19 files**

---

## 🔧 Step 3: Create Empty __init__.py Files

```bash
touch analyzers/__init__.py
touch extractors/__init__.py
```

These files tell Python these folders are modules.

---

## 💻 Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed flask-3.0.0 flask-cors-4.0.0 ...
```

---

## ✅ Step 5: Verify Installation

Run this Python script to verify:

```python
# verify_setup.py
import os

files_to_check = [
    'app.py',
    'config.py',
    'database.py',
    'requirements.txt',
    'analyzers/__init__.py',
    'analyzers/language_analyzer.py',
    'analyzers/credibility_analyzer.py',
    'analyzers/crosscheck_analyzer.py',
    'extractors/__init__.py',
    'extractors/article_extractor.py',
    'static/index.html',
    'static/css/styles.css',
    'static/js/api.js',
    'static/js/main.js',
    'static/js/tabs.js',
    'static/js/article-display.js',
    'static/js/scores-display.js',
    'static/js/charts-display.js'
]

print("Checking files...")
missing = []
for file in files_to_check:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")
        missing.append(file)

if missing:
    print(f"\n⚠️  {len(missing)} files missing!")
else:
    print("\n✅ All files present!")
```

---

## 🚀 Step 6: Run the Application

```bash
python app.py
```

**Expected output:**
```
============================================================
🔍 Fake News Detector - Starting Server
============================================================
Database: articles.db
Score Weights: {'language': 0.35, 'credibility': 0.4, 'crosscheck': 0.25}
============================================================
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

## 🌐 Step 7: Open in Browser

Navigate to: `http://localhost:5000`

**You should see:**
- Dark themed interface
- Sidebar with statistics (all zeros initially)
- Input field for URL
- "Analyze" button

---

## 🧪 Step 8: Test Basic Functionality

### Test 1: Analyze an Article

1. Enter URL: `https://www.bbc.com/news` (should score HIGH - verified reliable source)
2. Click "Analyze"
3. Wait for results
4. Verify:
   - [ ] Article information displayed
   - [ ] Overall score shown (likely 70%+ for BBC)
   - [ ] Three sub-scores displayed
   - [ ] **Chart 3 shows radar with 4 metrics** (Domain Age, URL Structure, Site Structure, Content Format)
   - [ ] Charts 1, 2, 4, 5 rendered
   - [ ] Chart 6 shows placeholder
   - [ ] Category shows "REAL (Verified Reliable Source)" or similar

### Test 1b: Test Known Problematic Source

1. Enter URL: `https://www.infowars.com` (if site is accessible)
2. Click "Analyze"
3. Verify:
   - [ ] Overall score is LOW (likely <40%)
   - [ ] Category shows "FAKE (Known Issue: Conspiracy theories)" or similar
   - [ ] Chart 3 radar shows low scores across all 4 metrics

### Test 2: Check Database Tab

1. Click "Database" button in sidebar
2. Verify:
   - [ ] Table shows analyzed article
   - [ ] Stats in sidebar updated

### Test 3: Analyze Another Article

1. Return to "Analyze" tab
2. Enter different URL
3. Verify:
   - [ ] Second article analyzed
   - [ ] Chart 5 (Similarity Map) now shows comparison

---

## 🔍 Step 9: Verify Each Component

### Backend Check

Open a new terminal and test API:

```bash
curl http://localhost:5000/api/stats
```

**Expected:**
```json
{"success":true,"total":1,"high_credibility":1,"medium_credibility":0,"low_credibility":0}
```

### Frontend Check

Open browser DevTools (F12):

**Console tab:**
- [ ] No red errors
- [ ] Should see: "Application initialized"

**Network tab:**
- [ ] `styles.css` status: 200
- [ ] All JS files status: 200
- [ ] Plotly CDN loaded

### Database Check

```bash
ls -la articles.db
```

Should see `articles.db` file created.

---

## 📊 Step 10: Verify Charts

After analyzing 2+ articles:

- [ ] **Chart 1**: Bar chart with 4 bars (Capitalization, Punctuation, Complexity, Grammar)
- [ ] **Chart 2**: Pie chart with 4 segments (Neutral, Sensational, Negative, Positive)
- [ ] **Chart 3**: **Radar chart with 4 axes** (Domain Age, URL Structure, Site Structure, Content Format) ⭐
- [ ] **Chart 4**: Scatter plot with points (similar articles)
- [ ] **Chart 5**: Scatter plot with colored dots (similarity map)
- [ ] **Chart 6**: Placeholder message

**⭐ Important:** Chart 3 is the credibility radar chart using domain-based analysis. It should show 4 metrics scored 0-100. This chart drives 50% of the overall credibility score.

---

## ⚙️ Step 11: Configuration (Optional)

Edit `config.py`:

```python
# Add NewsAPI key (optional)
NEWSAPI_KEY = "your_key_here"

# Customize trusted domains
CREDIBLE_DOMAINS = [
    "bbc.com",
    "your-trusted-source.com"
]

# Adjust scoring weights
SCORE_WEIGHTS = {
    "language": 0.35,
    "credibility": 0.40,
    "crosscheck": 0.25
}
```

---

## 🎨 Step 12: Verify Dark Theme

Check these elements have dark styling:

- [ ] Background is dark (#0E1117)
- [ ] Sidebar is darker (#262730)
- [ ] Text is light (#FAFAFA)
- [ ] Accent color is red (#FF4B4B)
- [ ] Cards have subtle borders (#3D3D3D)
- [ ] Charts have dark backgrounds (#1A1A1A)
- [ ] Hover effects work on buttons
- [ ] Input fields have dark background
- [ ] Overall looks like the attached reference image

--- color is red (#FF4B4B)
- [ ] Cards have subtle borders
- [ ] Charts have dark backgrounds
- [ ] Hover effects work on buttons

---

## 🐛 Common Issues & Fixes

### Issue 1: "ModuleNotFoundError: No module named 'analyzers'"

**Fix:**
```bash
# Make sure __init__.py exists
touch analyzers/__init__.py
touch extractors/__init__.py
```

### Issue 2: CSS not loading (page looks unstyled)

**Fix:**
```bash
# Verify file location
ls static/css/styles.css

# Hard refresh browser
# Windows/Linux: Ctrl+Shift+R
# Mac: Cmd+Shift+R
```

### Issue 3: Charts not displaying

**Fix:**
- Check browser console for errors
- Verify Plotly CDN is accessible
- Check if data is being returned from API

### Issue 4: "Failed to analyze article"

**Fix:**
- Verify URL is accessible
- Try a different article URL
- Check internet connection
- Review terminal for Python errors

### Issue 5: Database errors

**Fix:**
```bash
# Delete database and restart
rm articles.db
python app.py
```

---

## 📝 Final Verification Checklist

Run through this before considering setup complete:

**Backend:**
- [ ] Flask server starts without errors
- [ ] All analyzers import successfully
- [ ] Database initializes correctly
- [ ] API endpoints respond

**Frontend:**
- [ ] Page loads with dark theme
- [ ] No console errors (F12)
- [ ] All JavaScript files load (200 status)
- [ ] Plotly CDN loads successfully

**Functionality:**
- [ ] Can analyze an article
- [ ] Article info displays correctly
- [ ] All three scores calculated
- [ ] Charts 1-5 render properly
- [ ] Chart 6 shows placeholder
- [ ] Database tab works
- [ ] Can export to CSV
- [ ] Statistics update correctly

**User Interface:**
- [ ] Sidebar navigation works
- [ ] Tab switching works
- [ ] Buttons respond to clicks
- [ ] Loading indicator shows during analysis
- [ ] Error messages display properly
- [ ] Scroll behavior is smooth

---

## 🎯 Success Criteria

Your setup is complete when:

1. ✅ Server runs without errors
2. ✅ Browser shows dark themed interface
3. ✅ Can analyze at least one article successfully
4. ✅ All 5 charts display (Chart 6 is placeholder)
5. ✅ Database tab shows analyzed articles
6. ✅ Statistics update in sidebar
7. ✅ No errors in browser console
8. ✅ No errors in terminal

---

## 📚 Next Steps

After successful setup:

1. **Read DEVELOPER_GUIDE.md** - Learn how to extend the system
2. **Analyze 10+ articles** - Build up your database
3. **Customize config.py** - Add your trusted/suspicious domains
4. **Add Chart 6** - Implement your custom analysis
5. **Test thoroughly** - Try edge cases
6. **Document changes** - Keep track of modifications

---

## 🚀 Quick Start Commands

```bash
# Setup
mkdir -p fake-news-detector/analyzers/extractors/static/css/static/js
cd fake-news-detector
touch analyzers/__init__.py extractors/__init__.py

# Install
pip install -r requirements.txt

# Run
python app.py

# Open browser
http://localhost:5000

# Test API
curl http://localhost:5000/api/stats
```

---

## 📞 Getting Help

If you're stuck:

1. **Check terminal output** - Look for Python errors
2. **Check browser console** - Look for JavaScript errors  
3. **Verify file locations** - Run the verification script
4. **Review this checklist** - Make sure all steps completed
5. **Check DEVELOPER_GUIDE.md** - For detailed explanations

---

## ✨ Completion

Once all checkboxes are ticked and tests pass:

🎉 **Congratulations! Your Fake News Detector is ready!**

You can now:
- Analyze articles
- View credibility scores
- See visual analytics
- Manage database
- Extend with custom features

---

**Setup Time Estimate:** 15-30 minutes

**Last Updated:** 2025-01-02