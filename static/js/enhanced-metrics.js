/**
 * Enhanced Metrics Display Module
 * Creates dynamic credibility indicator cards with color-coded status
 * Similar to professional security scanning tools
 */

const EnhancedMetrics = {

    /**
     * Render credibility cards similar to security scanner interface.
     *
     * Creates 4 dynamic cards showing:
     * - Domain Age (with WHOIS verification badge if verified)
     * - URL Structure
     * - Site Structure
     * - Content Format
     *
     * Each card includes:
     * - Color-coded background and border (green=good, red=bad)
     * - Score out of 100 with progress bar
     * - Status badge (Excellent, Good, Moderate, etc.)
     * - Detailed description
     * - WHOIS verification badge (domain age only, when verified)
     *
     * @param {Object} analysisResult - Full analysis result from backend
     */
    renderCredibilityCards(analysisResult) {
        const container = document.getElementById('credibility-cards');
        if (!container) return;

        const metrics = analysisResult.detailed_metrics;
        const enhanced = analysisResult.enhanced_info || {};

        // Card configuration matching backend metrics
        const cards = [
            {
                id: 'domain_age',
                title: 'DOMAIN AGE',
                icon: '📅',
                data: metrics.domain_age,
                verified: enhanced.whois_verified,
                extraInfo: enhanced.domain_created ?
                    `Registered in ${enhanced.domain_created.split('-')[0]}` : ''
            },
            {
                id: 'url_structure',
                title: 'URL STRUCTURE',
                icon: '🔗',
                data: metrics.url_structure,
                verified: false
            },
            {
                id: 'site_structure',
                title: 'SITE STRUCTURE',
                icon: '🏛️',
                data: metrics.site_structure,
                verified: false
            },
            {
                id: 'content_format',
                title: 'CONTENT FORMAT',
                icon: '📄',
                data: metrics.content_format,
                verified: false
            }
        ];

        // Render all cards
        container.innerHTML = cards.map(card => this.createCard(card)).join('');
    },

    /**
     * Create individual credibility card HTML.
     *
     * Color coding logic (matching config.py SCORE_THRESHOLDS):
     * - 85+: Excellent (green)
     * - 70-84: Good (light green)
     * - 50-69: Moderate (yellow)
     * - 30-49: Poor (orange)
     * - 0-29: Critical (red)
     *
     * @param {Object} card - Card configuration object
     * @returns {string} HTML string for card
     */
    createCard(card) {
        const score = card.data.score;
        const colorClass = this.getColorClass(score);
        const bgColor = this.getBackgroundColor(colorClass);
        const verifiedBadge = card.verified ?
            '<span class="verified-badge">✓ WHOIS Verified</span>' : '';

        return `
            <div class="credibility-card ${colorClass}" style="background: ${bgColor}; border-left: 4px solid ${this.getBorderColor(colorClass)};">
                <div class="card-header">
                    <div class="card-title">
                        <span class="card-icon">${card.icon}</span>
                        <h3>${card.title}</h3>
                    </div>
                    ${verifiedBadge}
                </div>

                <div class="card-score">
                    <span class="score-value">${score}/100</span>
                    <div class="progress-bar">
                        <div class="progress-fill ${colorClass}" style="width: ${score}%"></div>
                    </div>
                </div>

                <div class="card-status">
                    <span class="status-badge ${colorClass}">${card.data.status}</span>
                </div>

                <div class="card-description">
                    ${card.data.description}
                    ${card.extraInfo ? `<div class="extra-info">${card.extraInfo}</div>` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Get color class based on score thresholds.
     * Matches backend config.SCORE_THRESHOLDS for consistency.
     *
     * @param {number} score - Score from 0-100
     * @returns {string} Color class name
     */
    getColorClass(score) {
        if (score >= 85) return 'excellent';
        if (score >= 70) return 'good';
        if (score >= 50) return 'moderate';
        if (score >= 30) return 'poor';
        return 'critical';
    },

    /**
     * Get background color for card based on color class.
     * Uses light, subtle colors for readability.
     *
     * @param {string} colorClass - Color class name
     * @returns {string} CSS color value
     */
    getBackgroundColor(colorClass) {
        const colors = {
            'excellent': '#e8f5e9',    // Light green
            'good': '#f1f8e9',         // Pale green
            'moderate': '#fff9c4',     // Light yellow
            'poor': '#fff3e0',         // Light orange
            'critical': '#ffebee'      // Light red
        };
        return colors[colorClass] || '#f5f5f5';
    },

    /**
     * Get border color for card based on color class.
     * Uses vibrant colors for visual hierarchy.
     *
     * @param {string} colorClass - Color class name
     * @returns {string} CSS color value
     */
    getBorderColor(colorClass) {
        const colors = {
            'excellent': '#4caf50',    // Green
            'good': '#8bc34a',         // Light green
            'moderate': '#ffeb3b',     // Yellow
            'poor': '#ff9800',         // Orange
            'critical': '#f44336'      // Red
        };
        return colors[colorClass] || '#9e9e9e';
    }
};

// Export for use in main app
window.EnhancedMetrics = EnhancedMetrics;
