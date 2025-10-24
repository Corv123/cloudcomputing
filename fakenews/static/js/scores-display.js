// static/js/scores-display.js
// Module for displaying scores

function displayScores(article) {
    console.log('🎯 DisplayScores called with article:', article);
    
    // Calculate individual scores
    const languageScore = Math.round(article.language_score * 100);
    const credibilityScore = Math.round(article.credibility_score * 100);
    const crosscheckScore = Math.round(article.cross_check_score * 100);
    const sensationalismScore = Math.round(article.sensationalism_bias_likelihood * 100);
    
    // Use backend's calculated overall score (weighted average)
    console.log('🔍 Raw overall_score from backend:', article.overall_score, typeof article.overall_score);
    const overallScore = Math.round(article.overall_score * 100);
    
    console.log('📊 Scores calculated:', {
        language: languageScore,
        credibility: credibilityScore,
        crosscheck: crosscheckScore,
        sensationalism: sensationalismScore,
        overall: overallScore,
        rawOverall: article.overall_score
    });
    
    // Display overall score
    const overallScoreElement = document.getElementById('overallScoreNumber');
    if (overallScoreElement) {
        overallScoreElement.textContent = overallScore;
        console.log('✅ Overall score set to:', overallScore);
        
        // Add a watcher to detect if the score gets changed
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' || mutation.type === 'characterData') {
                    console.log('🚨 Overall score was changed! New value:', overallScoreElement.textContent);
                }
            });
        });
        observer.observe(overallScoreElement, { childList: true, characterData: true, subtree: true });
        
    } else {
        console.error('❌ overallScoreNumber element not found!');
    }
    
    const scoreCircle = document.getElementById('overallScoreCircle');
    const scoreVerdict = document.getElementById('scoreVerdict');
    
    console.log('🔍 Elements found:', {
        scoreCircle: !!scoreCircle,
        scoreVerdict: !!scoreVerdict
    });
    
    if (!scoreCircle || !scoreVerdict) {
        console.error('❌ Missing score elements:', {
            scoreCircle: !!scoreCircle,
            scoreVerdict: !!scoreVerdict
        });
        return;
    }
    
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
    
    // Display individual scores
    document.getElementById('languageScore').textContent = languageScore + '%';
    document.getElementById('credibilityScore').textContent = credibilityScore + '%';
    document.getElementById('crosscheckScore').textContent = crosscheckScore + '%';
    document.getElementById('sensationalismScore').textContent = sensationalismScore + '%';
}
