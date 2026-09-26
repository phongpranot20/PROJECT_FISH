let selectedFiles = [];
let speciesChart = null;
let timelineChart = null;
let allHistoryLogs = [];
let webcamStream = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initSpecies();
  initModelStatus();
  initUploadHandlers();
  initCharts();
  loadStatsAndHistory();
  initWebcam();

  // Search input filter
  document.getElementById('logSearchInput').addEventListener('input', (e) => {
    filterHistoryLogs(e.target.value);
  });

  // Clear History
  document.getElementById('btnClearHistory').addEventListener('click', async () => {
    if (confirm("Are you sure you want to clear all analysis history?")) {
      await fetch('/api/history', { method: 'DELETE' });
      loadStatsAndHistory();
    }
  });
});

// Load Species Cards
async function initSpecies() {
  try {
    const res = await fetch('/api/species');
    const data = await res.json();
    const grid = document.getElementById('speciesGrid');
    grid.innerHTML = '';

    data.species.forEach(sp => {
      const card = document.createElement('div');
      card.className = 'species-card';
      card.innerHTML = `
        <div class="species-img-wrapper">
          <img src="${sp.image}" alt="${sp.name}" class="species-img" onerror="this.src='https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=400'">
          <span class="species-badge">${sp.family}</span>
        </div>
        <div class="species-info">
          <div class="species-name">${sp.name}</div>
          <div class="species-sci">${sp.scientific}</div>
          <button type="button" class="btn-sample" onclick="runSampleInference('${sp.sample_file}')">
            <span>⚡ Test Sample</span>
          </button>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load species:", err);
  }
}

// Check Model Status
async function initModelStatus() {
  try {
    const res = await fetch('/api/model-status');
    const data = await res.json();
    const badgeText = document.getElementById('modelStatusText');
    if (data.ready) {
      badgeText.textContent = `Model Ready (${data.model_size_mb} MB)`;
    } else {
      badgeText.textContent = "Model Initializing...";
      badgeText.parentElement.style.borderColor = "#f59e0b";
      badgeText.parentElement.style.color = "#f59e0b";
    }
  } catch (e) {
    console.warn("Status check failed:", e);
  }
}

// Setup Upload & Drag-and-Drop
function initUploadHandlers() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowse');
  const btnClearSelection = document.getElementById('btnClearSelection');
  const btnRunInference = document.getElementById('btnRunInference');

  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
  });

  btnClearSelection.addEventListener('click', () => {
    selectedFiles = [];
    fileInput.value = '';
    renderPreviews();
  });

  btnRunInference.addEventListener('click', () => {
    if (selectedFiles.length === 0) return;
    runInference(selectedFiles);
  });
}

function addFiles(files) {
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    if (f.type.startsWith('image/')) {
      selectedFiles.push(f);
    }
  }
  renderPreviews();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderPreviews();
}

function renderPreviews() {
  const section = document.getElementById('previewSection');
  const grid = document.getElementById('previewGrid');
  const countSpan = document.getElementById('selectedCount');

  if (selectedFiles.length === 0) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';
  countSpan.textContent = selectedFiles.length;
  grid.innerHTML = '';

  selectedFiles.forEach((file, idx) => {
    const card = document.createElement('div');
    card.className = 'preview-card';

    const reader = new FileReader();
    reader.onload = (e) => {
      card.innerHTML = `
        <img src="${e.target.result}" class="preview-thumb" alt="${file.name}">
        <button type="button" class="preview-remove" onclick="removeFile(${idx})">×</button>
        <div class="preview-name">${file.name}</div>
      `;
    };
    reader.readAsDataURL(file);
    grid.appendChild(card);
  });
}

// Run Inference for Multiple Uploaded Files
async function runInference(files) {
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.add('active');

  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    
    // Clear selection
    selectedFiles = [];
    renderPreviews();

    // Render results
    renderResults(data.results);

    // Refresh charts and history
    await loadStatsAndHistory();

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert("Inference failed: " + err.message);
  } finally {
    overlay.classList.remove('active');
  }
}

// Run Quick Sample Inference
async function runSampleInference(sampleFile) {
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.add('active');

  try {
    const res = await fetch('/api/predict-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_file: sampleFile })
    });
    const data = await res.json();

    renderResults([data], `/images/${sampleFile}`);
    await loadStatsAndHistory();
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert("Sample inference failed: " + err.message);
  } finally {
    overlay.classList.remove('active');
  }
}

// Render Results Grid
function renderResults(results, fallbackImg = null) {
  const section = document.getElementById('resultsSection');
  const container = document.getElementById('resultCards');
  container.innerHTML = '';

  if (!results || results.length === 0) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';

  results.forEach(item => {
    if (item.error) {
      const errCard = document.createElement('div');
      errCard.className = 'result-card';
      errCard.innerHTML = `<div style="color: #ef4444;">Error processing ${item.filename}: ${item.error}</div>`;
      container.appendChild(errCard);
      return;
    }

    const confClass = item.confidence >= 80 ? 'conf-high' : item.confidence >= 50 ? 'conf-med' : 'conf-low';
    const imgSrc = fallbackImg || `/images/${item.filename}`;

    const card = document.createElement('div');
    card.className = 'result-card';

    // Format probability rows
    const scoresHtml = (item.scores || []).map(s => {
      return `
        <div class="score-row">
          <span class="score-label" title="${s.species}">${s.species}</span>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: ${s.confidence}%"></div>
          </div>
          <span class="score-val">${s.confidence}%</span>
        </div>
      `;
    }).join('');

    card.innerHTML = `
      <div class="result-img-box">
        <img src="${imgSrc}" alt="${item.filename}" onerror="this.src='https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=400'">
      </div>
      <div class="result-content">
        <div class="result-header">
          <div class="result-species">${item.species}</div>
          <div class="result-confidence-badge ${confClass}">${item.confidence}% Match</div>
        </div>
        <div class="result-meta">
          <span>📁 ${item.filename}</span>
          <span>⚡ ${item.latency_ms} ms</span>
          <span>🕒 ${item.timestamp.split(' ')[1]}</span>
        </div>
        <div class="scores-container">
          ${scoresHtml}
        </div>
      </div>
    `;

    container.appendChild(card);
  });
}

// Initialize Charts
function initCharts() {
  const doughnutCtx = document.getElementById('speciesDoughnutChart').getContext('2d');
  speciesChart = new Chart(doughnutCtx, {
    type: 'doughnut',
    data: {
      labels: [],
      datasets: [{
        data: [],
        backgroundColor: [
          '#00f2fe', '#4facfe', '#7928ca', '#10b981', '#f59e0b', '#ec4899'
        ],
        borderWidth: 0,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
        }
      },
      cutout: '70%'
    }
  });

  const timelineCtx = document.getElementById('confidenceTimelineChart').getContext('2d');
  timelineChart = new Chart(timelineCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Confidence (%)',
        data: [],
        borderColor: '#00f2fe',
        backgroundColor: 'rgba(0, 242, 254, 0.12)',
        fill: true,
        tension: 0.35,
        pointBackgroundColor: '#4facfe',
        pointBorderColor: '#070c18',
        pointHoverRadius: 6,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', maxRotation: 45, maxTicksLimit: 8 }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b' }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Load Telemetry & History
async function loadStatsAndHistory() {
  try {
    const statsRes = await fetch('/api/stats');
    const stats = await statsRes.json();

    document.getElementById('statTotalAnalyzed').textContent = stats.total_analyzed || 0;
    document.getElementById('statAvgConfidence').textContent = (stats.avg_confidence || 0).toFixed(1) + '%';

    // Top detected species
    const counts = stats.species_counts || {};
    let topName = 'N/A';
    let topCount = 0;
    for (const [sp, c] of Object.entries(counts)) {
      if (c > topCount) {
        topCount = c;
        topName = sp;
      }
    }
    document.getElementById('statTopSpecies').textContent = topName;

    // Update Species Doughnut Chart
    const speciesLabels = Object.keys(counts);
    const speciesData = Object.values(counts);
    if (speciesChart) {
      speciesChart.data.labels = speciesLabels.length > 0 ? speciesLabels : ['No Data'];
      speciesChart.data.datasets[0].data = speciesData.length > 0 ? speciesData : [1];
      speciesChart.data.datasets[0].backgroundColor = speciesData.length > 0 ? [
        '#00f2fe', '#4facfe', '#7928ca', '#10b981', '#f59e0b', '#ec4899'
      ] : ['rgba(255,255,255,0.1)'];
      speciesChart.update();
    }

    // Update Timeline Line Chart
    if (timelineChart && stats.timeline) {
      timelineChart.data.labels = stats.timeline.map(t => t.Timestamp.split(' ')[1] || t.Timestamp);
      timelineChart.data.datasets[0].data = stats.timeline.map(t => t.Confidence);
      timelineChart.update();
    }

    // History Table
    const histRes = await fetch('/api/history');
    const hist = await histRes.json();
    allHistoryLogs = hist.logs || [];
    renderHistoryTable(allHistoryLogs);
  } catch (err) {
    console.error("Failed to load stats/history:", err);
  }
}

function renderHistoryTable(logs) {
  const tbody = document.getElementById('historyTableBody');
  tbody.innerHTML = '';

  if (!logs || logs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; padding: 24px; color: var(--text-muted);">
          No classification records found. Upload an image above to start.
        </td>
      </tr>
    `;
    return;
  }

  logs.forEach(row => {
    const tr = document.createElement('tr');
    const confClass = row.Confidence >= 80 ? 'conf-high' : row.Confidence >= 50 ? 'conf-med' : 'conf-low';
    tr.innerHTML = `
      <td style="color: var(--text-primary); font-weight: 500;">${row.Timestamp}</td>
      <td>${row.Filename}</td>
      <td style="color: #38bdf8; font-weight: 600;">${row.Species}</td>
      <td>
        <span class="result-confidence-badge ${confClass}" style="font-size: 0.78rem;">
          ${Number(row.Confidence).toFixed(2)}%
        </span>
      </td>
      <td><span style="color: #34d399;">✓ Processed</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterHistoryLogs(query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    renderHistoryTable(allHistoryLogs);
    return;
  }
  const filtered = allHistoryLogs.filter(row => 
    (row.Species && row.Species.toLowerCase().includes(q)) ||
    (row.Filename && row.Filename.toLowerCase().includes(q)) ||
    (row.Timestamp && row.Timestamp.toLowerCase().includes(q))
  );
  renderHistoryTable(filtered);
}

// Camera / Webcam Support
function initWebcam() {
  const btnOpen = document.getElementById('btnOpenWebcam');
  const btnClose = document.getElementById('btnCloseWebcam');
  const btnCapture = document.getElementById('btnCapturePhoto');
  const modal = document.getElementById('webcamModal');
  const video = document.getElementById('webcamVideo');
  const canvas = document.getElementById('webcamCanvas');

  btnOpen.addEventListener('click', async () => {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      video.srcObject = webcamStream;
      modal.classList.add('active');
    } catch (e) {
      alert("Camera access denied or unavailable: " + e.message);
    }
  });

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    modal.classList.remove('active');
  };

  btnClose.addEventListener('click', stopWebcam);

  btnCapture.addEventListener('click', () => {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
      stopWebcam();
      selectedFiles.push(file);
      renderPreviews();
      runInference([file]);
    }, 'image/jpeg', 0.95);
  });
}
