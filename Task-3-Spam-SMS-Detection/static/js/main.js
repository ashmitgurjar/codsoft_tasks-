/* ==========================================================================
   Spam SMS Detector - Main JavaScript Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initClassifier();
    initSampleChips();
    initMetricsAndCharts();
    initBatchUploader();
});

// Global state variables
let metricsDataGlobal = null;
let metricsChartInstance = null;

/* --------------------------------------------------------------------------
   1. Tab Navigation System
   -------------------------------------------------------------------------- */
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

/* --------------------------------------------------------------------------
   2. Real-time SMS Classifier Logic
   -------------------------------------------------------------------------- */
function initClassifier() {
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const smsInput = document.getElementById('smsInput');
    const modelSelect = document.getElementById('modelSelect');

    classifyBtn.addEventListener('click', async () => {
        const text = smsInput.value.trim();
        if (!text) {
            alert('Please enter an SMS message to classify.');
            return;
        }

        classifyBtn.disabled = true;
        classifyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    model_name: modelSelect.value
                })
            });

            const data = await response.json();
            if (response.ok) {
                renderResult(data);
            } else {
                alert(data.error || 'Failed to classify message.');
            }
        } catch (err) {
            console.error('Classification error:', err);
            alert('Error connecting to the classification server.');
        } finally {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Classify Message';
        }
    });

    clearBtn.addEventListener('click', () => {
        smsInput.value = '';
        resetResultCard();
    });
}

function renderResult(data) {
    const resultCard = document.getElementById('resultCard');
    const resultBadge = document.getElementById('resultBadge');
    const resultBody = document.getElementById('resultBody');

    resultCard.classList.remove('result-card-empty');

    const isSpam = data.is_spam;
    const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
    resultBadge.className = `badge ${badgeClass}`;
    resultBadge.innerText = isSpam ? 'SPAM DETECTED' : 'LEGITIMATE (HAM)';

    // Word Highlighting Construction
    let wordHighlightsHTML = '';
    data.word_analysis.forEach(item => {
        if (item.is_spam_indicator) {
            wordHighlightsHTML += `<span class="word-spam-tag" title="Spam Weight: +${item.score}">${escapeHtml(item.token)}</span> `;
        } else {
            wordHighlightsHTML += `${escapeHtml(item.token)} `;
        }
    });

    // All Models Prediction Mini Cards
    let miniCardsHTML = '';
    for (const [modelName, res] of Object.entries(data.all_models)) {
        const miniBadge = res.is_spam ? 'text-danger' : 'text-success';
        miniCardsHTML += `
            <div class="mini-model-card">
                <div class="mini-model-name">${modelName}</div>
                <div class="mini-model-val ${miniBadge}">${res.label.split(' ')[0]}</div>
                <div style="font-size:0.75rem; color:var(--text-dim);">${res.spam_probability}% Spam</div>
            </div>
        `;
    }

    const spamWidth = isSpam ? data.spam_probability : (100 - data.ham_probability);
    const fillClass = isSpam ? 'prob-fill-spam' : 'prob-fill-ham';

    resultBody.innerHTML = `
        <div class="result-box">
            <div class="verdict-header ${isSpam ? 'verdict-spam' : 'verdict-ham'}">
                <div class="verdict-title-box">
                    <h3>${isSpam ? '🚨 SPAM / PHISHING DETECTED' : '✅ LEGITIMATE (HAM) SMS'}</h3>
                    <p class="verdict-subtitle">Classified via ${escapeHtml(data.selected_model)} with ${data.confidence_score}% confidence</p>
                </div>
            </div>

            <!-- Probability Bar -->
            <div class="prob-container">
                <div class="prob-header">
                    <span>Spam Likelihood</span>
                    <span>${data.spam_probability}%</span>
                </div>
                <div class="prob-track">
                    <div class="${fillClass}" style="width: ${spamWidth}%;"></div>
                </div>
            </div>

            <!-- XAI Word Highlighting -->
            <div class="words-box">
                <h4><i class="fa-solid fa-highlighter"></i> Spam Keyword Analysis (Explainable AI):</h4>
                <div class="highlighted-text">
                    ${wordHighlightsHTML}
                </div>
            </div>

            <!-- All Classifiers Comparison -->
            <div>
                <h4 style="font-size:0.85rem; color:var(--text-muted); margin-bottom:10px;">
                    <i class="fa-solid fa-layer-group"></i> Multi-Model Cross-Verification:
                </h4>
                <div class="models-mini-grid">
                    ${miniCardsHTML}
                </div>
            </div>
        </div>
    `;
}

function resetResultCard() {
    const resultCard = document.getElementById('resultCard');
    const resultBadge = document.getElementById('resultBadge');
    const resultBody = document.getElementById('resultBody');

    resultCard.classList.add('result-card-empty');
    resultBadge.className = 'badge';
    resultBadge.innerText = 'Awaiting Input';

    resultBody.innerHTML = `
        <div class="placeholder-state">
            <i class="fa-solid fa-envelope-open-text placeholder-icon"></i>
            <p>Enter an SMS message on the left and click <strong>Classify Message</strong> to view real-time AI predictions, confidence breakdown, and spam keyword highlights.</p>
        </div>
    `;
}

/* --------------------------------------------------------------------------
   3. Quick Sample Chips
   -------------------------------------------------------------------------- */
function initSampleChips() {
    const chips = document.querySelectorAll('.chip');
    const smsInput = document.getElementById('smsInput');
    const classifyBtn = document.getElementById('classifyBtn');

    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const text = chip.getAttribute('data-sample');
            smsInput.value = text;
            classifyBtn.click();
        });
    });
}

/* --------------------------------------------------------------------------
   4. Metrics & Charts Analytics
   -------------------------------------------------------------------------- */
async function initMetricsAndCharts() {
    try {
        const response = await fetch('/api/metrics');
        metricsDataGlobal = await response.json();

        updateNavbarMetrics(metricsDataGlobal);
        renderMetricsChart(metricsDataGlobal.models);
        renderMetricsTable(metricsDataGlobal.models);
        renderConfusionMatrix(metricsDataGlobal.models['Naive Bayes'].confusion_matrix);
        renderKeywordCloud(metricsDataGlobal.top_spam_keywords);
    } catch (err) {
        console.error('Failed to load metrics:', err);
    }
}

function updateNavbarMetrics(data) {
    let bestModel = 'Naive Bayes';
    let bestAcc = 0;

    for (const [name, stats] of Object.entries(data.models)) {
        if (stats.accuracy >= bestAcc) {
            bestAcc = stats.accuracy;
            bestModel = name;
        }
    }

    document.getElementById('topModelName').innerText = bestModel;
    document.getElementById('topModelAcc').innerText = `${(bestAcc * 100).toFixed(1)}%`;
    document.getElementById('statDatasetSize').innerText = `${data.total_samples} SMS`;
    document.getElementById('statBestF1').innerText = `${(data.models[bestModel].f1_score * 100).toFixed(1)}%`;
}

function renderMetricsChart(modelsData) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    const modelNames = Object.keys(modelsData);

    const accuracies = modelNames.map(m => modelsData[m].accuracy * 100);
    const precisions = modelNames.map(m => modelsData[m].precision * 100);
    const recalls = modelNames.map(m => modelsData[m].recall * 100);
    const f1Scores = modelNames.map(m => modelsData[m].f1_score * 100);

    if (metricsChartInstance) {
        metricsChartInstance.destroy();
    }

    metricsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: '#6366f1',
                    borderWidth: 1
                },
                {
                    label: 'Precision (%)',
                    data: precisions,
                    backgroundColor: 'rgba(168, 85, 247, 0.7)',
                    borderColor: '#a855f7',
                    borderWidth: 1
                },
                {
                    label: 'Recall (%)',
                    data: recalls,
                    backgroundColor: 'rgba(6, 182, 212, 0.7)',
                    borderColor: '#06b6d4',
                    borderWidth: 1
                },
                {
                    label: 'F1-Score (%)',
                    data: f1Scores,
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: '#10b981',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f8fafc', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

function renderMetricsTable(modelsData) {
    const tbody = document.getElementById('metricsTableBody');
    tbody.innerHTML = '';

    for (const [name, stats] of Object.entries(modelsData)) {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.addEventListener('click', () => {
            renderConfusionMatrix(stats.confusion_matrix, name);
        });

        tr.innerHTML = `
            <td><strong>${escapeHtml(name)}</strong></td>
            <td>${(stats.accuracy * 100).toFixed(1)}%</td>
            <td>${(stats.precision * 100).toFixed(1)}%</td>
            <td>${(stats.recall * 100).toFixed(1)}%</td>
            <td><span class="badge badge-info">${(stats.f1_score * 100).toFixed(1)}%</span></td>
            <td>${(stats.roc_auc * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    }
}

function renderConfusionMatrix(cm, modelName = 'Naive Bayes') {
    const cmGrid = document.getElementById('cmGrid');
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];

    cmGrid.innerHTML = `
        <div class="cm-cell">
            <div class="cm-cell-val text-success">${tn}</div>
            <div class="cm-cell-label">True Negative (Ham)</div>
        </div>
        <div class="cm-cell">
            <div class="cm-cell-val text-danger">${fp}</div>
            <div class="cm-cell-label">False Positive (False Spam)</div>
        </div>
        <div class="cm-cell">
            <div class="cm-cell-val text-danger">${fn}</div>
            <div class="cm-cell-label">False Negative (Missed Spam)</div>
        </div>
        <div class="cm-cell">
            <div class="cm-cell-val text-success">${tp}</div>
            <div class="cm-cell-label">True Positive (Spam)</div>
        </div>
    `;
}

function renderKeywordCloud(keywords) {
    const container = document.getElementById('keywordCloud');
    container.innerHTML = '';

    keywords.forEach(kw => {
        const tag = document.createElement('div');
        tag.className = 'word-tag';
        tag.innerHTML = `
            <span class="w-text">${escapeHtml(kw.word)}</span>
            <span class="w-weight">+${kw.weight}</span>
        `;
        container.appendChild(tag);
    });
}

/* --------------------------------------------------------------------------
   5. Batch CSV / TXT File Processor
   -------------------------------------------------------------------------- */
function initBatchUploader() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseFileBtn = document.getElementById('browseFileBtn');

    browseFileBtn.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    const batchResults = document.getElementById('batchResults');
    const batchTableBody = document.getElementById('batchTableBody');
    batchTableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;"><i class="fa-solid fa-spinner fa-spin"></i> Processing batch file...</td></tr>';
    batchResults.classList.remove('hidden');

    try {
        const response = await fetch('/api/batch_predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById('batchTotal').innerText = data.total_processed;
            document.getElementById('batchSpam').innerText = data.spam_count;
            document.getElementById('batchHam').innerText = data.ham_count;
            document.getElementById('batchRatio').innerText = `${data.spam_percentage}%`;

            batchTableBody.innerHTML = '';
            data.results.forEach((row, idx) => {
                const tr = document.createElement('tr');
                const badgeClass = row.is_spam ? 'badge-spam' : 'badge-ham';
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td style="max-width:400px; word-break:break-word;">${escapeHtml(row.text)}</td>
                    <td><span class="badge ${badgeClass}">${row.label}</span></td>
                    <td>${row.spam_probability}%</td>
                `;
                batchTableBody.appendChild(tr);
            });
        } else {
            alert(data.error || 'Failed to process file.');
        }
    } catch (err) {
        console.error('Batch upload error:', err);
        alert('Error uploading file to server.');
    }
}

// Utility: Escape HTML
function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
