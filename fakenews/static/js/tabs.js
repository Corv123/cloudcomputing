// static/js/tabs.js
// Tab management module

function switchTab(tabName) {
    console.log('🔵 [TABS.JS] Switching to tab:', tabName);
    
    // Remove active from all nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Remove active from all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Activate selected tab
    const targetBtn = document.querySelector(`[onclick="switchTab('${tabName}')"]`);
    const targetTab = document.getElementById(`${tabName}Tab`);
    
    console.log('🔵 [TABS.JS] Target button:', targetBtn ? 'found' : 'NOT FOUND');
    console.log('🔵 [TABS.JS] Target tab:', targetTab ? 'found' : 'NOT FOUND');
    
    if (targetBtn) targetBtn.classList.add('active');
    if (targetTab) targetTab.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'database') {
        console.log('🔵 [TABS.JS] Calling loadDatabase()...');
        if (typeof loadDatabase === 'function') {
            loadDatabase();
        } else {
            console.error('❌ [TABS.JS] loadDatabase is not defined!');
        }
    }
}
