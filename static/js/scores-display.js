// static/js/scores-display.js
// Module for displaying scores

function displayScores(article) {
    // Overall score
    const overallScore = Math.round(article.overall_score * 100);
    document.getElementById('overallScoreNumber').textContent = overallScore;
    
    const scoreCircle = document.getElementById('overallScoreCircle');
    const scoreVerdict = document.getElementById('scoreVerdict');
    
    // Apply styling based on score
    scoreCircle.classList.remove('high', 'medium', 'low');
    
    if (overallScore >= 70) {
        scoreCircle.classList.add('high');
        scoreVerdict.textContent = '✓ HIGH CREDIBILITY';
        scoreVerdict.style.color = 'var(--success)';
    } else if (overallScore >= 40) {
        scoreCircle.classList.add('medium');
        scoreVerdict.textContent = '⚠ MEDIUM CREDIBILITY';
        scoreVerdict.style.color = 'var(--warning)';
    } else {
        scoreCircle.classList.add('low');
        scoreVerdict.textContent = '✗ LOW CREDIBILITY';
        scoreVerdict.style.color = 'var(--danger)';
    }
    
    // Individual scores
    document.getElementById('languageScore').textContent = 
        Math.round(article.language_score * 100) + '%';
    
    document.getElementById('credibilityScore').textContent = 
        Math.round(article.credibility_score * 100) + '%';
    
    document.getElementById('crosscheckScore').textContent = 
        Math.round(article.cross_check_score * 100) + '%';
}
