// Purpose-Driven Score Display - Version 3.0
// Purpose: Show credibility verdict clearly and build trust

function displayScores(article) {
    console.log('🎯 DisplayScores called with article:', article);
    
    // Calculate individual scores with NaN protection
    const languageScore = Math.round((article.language_score ?? 0) * 100);
    const credibilityScore = Math.round((article.credibility_score ?? 0) * 100);
    const crosscheckScore = Math.round((article.cross_check_score ?? 0) * 100);
    
    // Sensationalism: Handle correctly
    let rawSensationalism = article.sensationalism_bias_likelihood ?? 0.5;
    rawSensationalism = typeof rawSensationalism === 'string' ? parseFloat(rawSensationalism) : rawSensationalism;
    const sensationalismScore = Math.round(rawSensationalism * 100);
    
    // Overall score
    const overallScore = Math.round((article.overall_score ?? 0.5) * 100);
    
    console.log('📊 Scores calculated:', {
        language: languageScore,
        credibility: credibilityScore,
        crosscheck: crosscheckScore,
        sensationalism: sensationalismScore,
        overall: overallScore
    });
    
    // Update verdict card
    const verdictCard = document.getElementById('verdictCard');
    const overallScoreCircle = document.getElementById('overallScoreCircle');
    const overallScoreNumber = document.getElementById('overallScoreNumber');
    const verdictBadge = document.getElementById('verdictBadge');
    const verdictText = document.getElementById('verdictText');
    
    if (!verdictCard || !overallScoreCircle || !overallScoreNumber || !verdictBadge || !verdictText) {
        console.error('❌ Missing verdict card elements');
        return;
    }
    
    // Update overall score number
    overallScoreNumber.textContent = overallScore;
    
    // Determine verdict and styling
    let verdictClass = '';
    let verdictLabel = '';
    let badgeClass = '';
    
    if (overallScore >= 70) {
        verdictClass = 'high';
        verdictLabel = 'HIGH CREDIBILITY';
        badgeClass = 'high';
        overallScoreCircle.classList.add('high');
        overallScoreCircle.classList.remove('medium', 'low');
    } else if (overallScore >= 40) {
        verdictClass = 'medium';
        verdictLabel = 'MEDIUM CREDIBILITY';
        badgeClass = 'medium';
        overallScoreCircle.classList.add('medium');
        overallScoreCircle.classList.remove('high', 'low');
    } else {
        verdictClass = 'low';
        verdictLabel = 'LOW CREDIBILITY';
        badgeClass = 'low';
        overallScoreCircle.classList.add('low');
        overallScoreCircle.classList.remove('high', 'medium');
    }
    
    // Update verdict card styling
    verdictCard.classList.remove('high', 'medium', 'low');
    verdictCard.classList.add(verdictClass);
    
    // Update badge
    verdictBadge.classList.remove('high', 'medium', 'low');
    verdictBadge.classList.add(badgeClass);
    verdictText.textContent = verdictLabel;
    
    // Display individual scores in verdict card
    document.getElementById('languageScore').textContent = languageScore + '%';
    document.getElementById('credibilityScore').textContent = credibilityScore + '%';
    document.getElementById('crosscheckScore').textContent = crosscheckScore + '%';
    document.getElementById('sensationalismScore').textContent = sensationalismScore + '%';
    
    // Show verdict card
    if (verdictCard) {
        verdictCard.style.display = 'block';
    }
}
