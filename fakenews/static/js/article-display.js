// static/js/article-display.js
// Module for displaying article information

function displayArticleInfo(article) {
    // Display title
    document.getElementById('infoTitle').textContent = article.title || 'N/A';
    
    // Display source
    document.getElementById('infoSource').textContent = article.source || 'N/A';
    
    // Display URL
    const urlElement = document.getElementById('infoUrl');
    urlElement.href = article.url;
    urlElement.textContent = article.url;
    
    // Display published date
    const publishedDate = article.published_at ? 
        article.published_at.substring(0, 10) : 'N/A';
    document.getElementById('infoPublished').textContent = publishedDate;
    
    // Display content preview (first 300 characters)
    const contentPreview = article.content ? 
        article.content.substring(0, 300) + '...' : 'N/A';
    document.getElementById('infoContent').textContent = contentPreview;
    
    // Display category with source classification if available
    const categoryElement = document.getElementById('infoCategory');
    const overall = article.overall_score;
    
    let categoryText = '';
    let categoryClass = 'category-badge';
    
    // Determine category based on overall score
    if (overall >= 0.7) {
        categoryText = 'REAL';
        categoryClass += ' category-real';
    } else if (overall >= 0.4) {
        categoryText = 'MIXED';
        categoryClass += ' category-mixed';
    } else {
        categoryText = 'FAKE';
        categoryClass += ' category-fake';
    }
    
    // Add source classification if known
    if (article.known_source_classification) {
        categoryText += ` (${article.known_source_classification})`;
    }
    
    categoryElement.textContent = categoryText;
    categoryElement.className = categoryClass;
}