// ==========================================================================
// DocuSense AI - Frontend Dashboard Controller
// ==========================================================================

let selectedFile = null;

function switchInputMode(mode) {
    document.getElementById('tabPptxMode').classList.toggle('active', mode === 'pptx');
    document.getElementById('tabTextMode').classList.toggle('active', mode === 'text');
    document.getElementById('pptxUploadContainer').classList.toggle('active', mode === 'pptx');
    document.getElementById('textUploadContainer').classList.toggle('active', mode === 'text');
}

function switchResultTab(tabId) {
    document.querySelectorAll('.res-tab-btn').forEach(btn => btn.classList.remove('active'));
    event.currentTarget.classList.add('active');

    document.querySelectorAll('.res-tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
        selectedFile = files[0];
        document.getElementById('chosenFileName').textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / (1024*1024)).toFixed(2)} MB)`;
    }
}

// Drag & drop handlers
const dropzone = document.getElementById('dropzone');
if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--color-cyan)';
        });
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'rgba(6, 182, 212, 0.3)';
        });
    });
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            selectedFile = files[0];
            document.getElementById('chosenFileName').textContent = `Selected: ${selectedFile.name} (${(selectedFile.size / (1024*1024)).toFixed(2)} MB)`;
        }
    });
}

async function uploadAndAnalyzePPTX() {
    if (!selectedFile) {
        alert('Please choose or drag a .pptx presentation first.');
        return;
    }

    const btn = document.getElementById('btnAnalyzePptx');
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing Presentation...';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch('/api/presentation/analyse', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload error');

        renderAnalysisResults(data);
    } catch (err) {
        alert('Analysis Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Analyze Presentation';
    }
}

async function loadSamplePresentation() {
    const btn = event.currentTarget;
    btn.disabled = true;
    btn.textContent = '⏳ Loading Sample Deck...';

    try {
        // Fetch sample file as blob
        const fileRes = await fetch('/static/../sample_presentation.pptx');
        const blob = await fileRes.blob();
        selectedFile = new File([blob], 'sample_presentation.pptx', { type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' });
        document.getElementById('chosenFileName').textContent = 'Selected: sample_presentation.pptx (Demo Deck)';
        await uploadAndAnalyzePPTX();
    } catch (err) {
        alert('Could not load sample deck: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '📁 Load Pre-Built Sample (sample_presentation.pptx)';
    }
}

async function analyzeDirectText() {
    const text = document.getElementById('rawTextInput').value.trim();
    if (!text) {
        alert('Please paste some text to inspect.');
        return;
    }

    try {
        const res = await fetch('/api/text/analyse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Text analysis failed');

        renderDirectTextResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

function renderAnalysisResults(data) {
    const doc = data.doc_summary;
    const slides = data.slides;
    const sentences = data.sentences;

    // Show summary box
    document.getElementById('summaryPlaceholder').style.display = 'none';
    document.getElementById('summaryContent').style.display = 'block';

    document.getElementById('metaFilename').textContent = doc.filename;
    document.getElementById('metaSlides').textContent = doc.slide_count;
    document.getElementById('metaWords').textContent = doc.total_words;
    
    const overallPct = Math.round(doc.overall_ai_prob * 100);
    const scoreEl = document.getElementById('metaOverallScore');
    scoreEl.textContent = `${overallPct}% (${doc.overall_ai_prob >= 0.70 ? 'High' : (doc.overall_ai_prob >= 0.40 ? 'Moderate' : 'Low')})`;
    scoreEl.className = 'font-bold ' + (doc.overall_ai_prob >= 0.70 ? 'text-rose' : (doc.overall_ai_prob >= 0.40 ? 'text-amber' : 'text-emerald'));

    document.getElementById('overallProbFill').style.width = `${overallPct}%`;

    // Tone & Summary
    document.getElementById('toneBadge').textContent = doc.tone.primary_tone;
    document.getElementById('toneIndicators').textContent = doc.tone.indicators.join(' | ') || 'Standard vocabulary';

    const sList = document.getElementById('summaryList');
    sList.innerHTML = doc.summary.map(s => `<li>${escapeHtml(s)}</li>`).join('');

    // Render Sentences
    renderSentences(sentences);

    // Render Slides
    renderSlides(slides);
}

function renderDirectTextResults(data) {
    document.getElementById('summaryPlaceholder').style.display = 'none';
    document.getElementById('summaryContent').style.display = 'block';

    document.getElementById('metaFilename').textContent = 'Direct Text Submission';
    document.getElementById('metaSlides').textContent = 'N/A';
    document.getElementById('metaWords').textContent = data.sentences.reduce((acc, s) => acc + s.sentence.split(/\s+/).length, 0);

    const overallPct = Math.round(data.overall_ai_prob * 100);
    const scoreEl = document.getElementById('metaOverallScore');
    scoreEl.textContent = `${overallPct}%`;
    scoreEl.className = 'font-bold ' + (data.overall_ai_prob >= 0.70 ? 'text-rose' : (data.overall_ai_prob >= 0.40 ? 'text-amber' : 'text-emerald'));
    document.getElementById('overallProbFill').style.width = `${overallPct}%`;

    document.getElementById('toneBadge').textContent = data.tone.primary_tone;
    document.getElementById('toneIndicators').textContent = data.tone.indicators.join(' | ') || 'Standard vocabulary';

    const sList = document.getElementById('summaryList');
    sList.innerHTML = data.summary.map(s => `<li>${escapeHtml(s)}</li>`).join('');

    renderSentences(data.sentences);
}

function renderSentences(sentences) {
    const container = document.getElementById('sentenceHighlightContainer');
    if (!sentences || sentences.length === 0) {
        container.innerHTML = '<p class="text-muted">No sentences extracted.</p>';
        return;
    }

    container.innerHTML = sentences.map((item, idx) => {
        let probDisplay = item.ai_probability !== null ? `${Math.round(item.ai_probability * 100)}% AI Likeness` : 'Insufficient Text (< 5 tokens)';
        return `
            <div class="sentence-item ${item.band}">
                <div>${escapeHtml(item.sentence)}</div>
                <div class="sentence-meta">
                    <span>Sentence #${idx + 1}</span>
                    <span class="font-bold">${probDisplay}</span>
                </div>
            </div>
        `;
    }).join('');
}

function renderSlides(slides) {
    const container = document.getElementById('slidesListContainer');
    if (!slides || slides.length === 0) {
        container.innerHTML = '<p class="text-muted">No slides extracted.</p>';
        return;
    }

    container.innerHTML = slides.map(s => {
        const warningsHtml = s.warnings.length > 0
            ? s.warnings.map(w => `<div class="slide-warning-tag">⚠️ ${escapeHtml(w)}</div>`).join('')
            : '<div class="text-emerald" style="font-size:0.75rem; margin-top:4px">✓ Passed all structure heuristics</div>';

        return `
            <div class="slide-card">
                <div class="slide-card-header">
                    <strong>Slide #${s.slide}</strong>
                    <span class="badge-mini">${s.word_count} words | ${s.has_picture ? '🖼️ Visual' : '📝 Text-only'}</span>
                </div>
                <div class="slide-preview-text">${escapeHtml(s.text || '[Blank or Non-Text Slide]')}</div>
                <div class="slide-audit-box">
                    ${warningsHtml}
                </div>
            </div>
        `;
    }).join('');
}

async function sendRAGQuery() {
    const input = document.getElementById('ragInput');
    const query = input.value.trim();
    if (!query) return;

    const history = document.getElementById('ragChatHistory');

    // Append user message
    const uMsg = document.createElement('div');
    uMsg.className = 'rag-msg user';
    uMsg.innerHTML = `<div class="bubble">${escapeHtml(query)}</div>`;
    history.appendChild(uMsg);
    input.value = '';
    history.scrollTop = history.scrollHeight;

    // Append bot thinking
    const bMsg = document.createElement('div');
    bMsg.className = 'rag-msg bot';
    bMsg.innerHTML = `<div class="bubble text-muted">Searching slides and asking local Phi...</div>`;
    history.appendChild(bMsg);
    history.scrollTop = history.scrollHeight;

    try {
        const res = await fetch('/api/presentation/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'The local RAG request failed.');
        }
        bMsg.innerHTML = `<div class="bubble">${escapeHtml(data.answer)}</div>`;
    } catch (err) {
        bMsg.innerHTML = `<div class="bubble text-rose">Error: ${err.message}</div>`;
    }
    history.scrollTop = history.scrollHeight;
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;');
}
