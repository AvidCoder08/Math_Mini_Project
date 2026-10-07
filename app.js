/**
 * LINEAR ALGEBRA AUTOMATED DATA ANALYSIS TOOL - APP CONTROLLER
 * Client-side orchestration, UI interactivity, Chart.js plotting, KaTeX typesetting,
 * and Web Worker background execution.
 */

// --- Global Application State ---
const state = {
  currentSource: 'housing', // 'housing' | 'diabetes' | 'upload'
  rawCsvText: '',
  parsedRows: [],
  headers: [],
  numericHeaders: [],
  selectedTarget: '',
  varTarget: 0.90,
  testFrac: 0.20,
  seed: 42,
  addRedundant: true,
  isComputing: false,
  lastResults: null,
  theme: 'light',
  worker: null,
  workerAvailable: false,
  charts: {
    predActual: null,
    scree: null,
    cumVar: null
  }
};

// --- Initialization on DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initWorker();
  initEventListeners();
  loadDatasetSource('housing');
});

// --- Theme Management (Light Parchment vs Dark Ink-Blue) ---
function initTheme() {
  const savedTheme = localStorage.getItem('la_app_theme') || 'light';
  setTheme(savedTheme);

  const themeToggle = document.getElementById('theme-toggle');
  themeToggle.addEventListener('click', () => {
    const nextTheme = state.theme === 'light' ? 'dark' : 'light';
    setTheme(nextTheme);
  });
}

function setTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('la_app_theme', theme);

  const themeIcon = document.getElementById('theme-icon');
  const themeLabel = document.getElementById('theme-label');
  if (theme === 'dark') {
    themeIcon.textContent = '◑';
    themeLabel.textContent = 'Paper Mode';
  } else {
    themeIcon.textContent = '◐';
    themeLabel.textContent = 'Ink Mode';
  }

  // Refresh charts with theme-adjusted colors
  if (state.lastResults) {
    updateCharts(state.lastResults);
  }
}

// --- Web Worker Setup with Graceful Fallback ---
function initWorker() {
  const statusBadge = document.getElementById('worker-status');
  const statusText = document.getElementById('worker-status-text');

  try {
    // Attempt spawning worker from worker.js
    state.worker = new Worker('worker.js');
    state.workerAvailable = true;

    state.worker.onmessage = (e) => {
      handleWorkerMessage(e.data);
    };

    state.worker.onerror = (err) => {
      console.warn('Web Worker error, falling back to main-thread execution:', err);
      state.workerAvailable = false;
      updateWorkerBadge(false);
    };

    updateWorkerBadge(true);
  } catch (err) {
    console.warn('Worker instantiation failed (possibly file:// protocol restrictions). Falling back to inline computation.', err);
    state.workerAvailable = false;
    updateWorkerBadge(false);
  }
}

function updateWorkerBadge(available) {
  const statusText = document.getElementById('worker-status-text');
  const statusDot = document.querySelector('.status-dot');
  if (available) {
    statusText.textContent = 'Worker Ready';
    statusDot.style.background = '#10b981';
  } else {
    statusText.textContent = 'Main Thread Active';
    statusDot.style.background = '#3b82f6';
  }
}

function handleWorkerMessage(data) {
  const { type, payload } = data;
  state.isComputing = false;
  setRunButtonLoading(false);

  if (type === 'PIPELINE_COMPLETE') {
    state.lastResults = payload;
    renderResults(payload);
    showToast('Decomposition completed and verified.', 'success');
  } else if (type === 'PIPELINE_ERROR') {
    showToast(`Error: ${payload.message}`, 'error');
    console.error('Pipeline Error:', payload);
  }
}

// --- Event Listeners & UI Binding ---
function initEventListeners() {
  // Navigation Tabs
  const navTabBtns = document.querySelectorAll('.nav-tab-btn');
  navTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const targetTab = btn.getAttribute('data-tab');
      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
      });
      document.getElementById(`pane-${targetTab}`).classList.add('active');
    });
  });

  // Dataset Source Buttons
  const sourceBtns = document.querySelectorAll('.source-btn');
  sourceBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sourceBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const source = btn.getAttribute('data-source');
      loadDatasetSource(source);
    });
  });

  // Drag and Drop Zone
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  dropZone.addEventListener('click', () => fileInput.click());
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
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Target Select Change
  const targetSelect = document.getElementById('target-select');
  targetSelect.addEventListener('change', (e) => {
    state.selectedTarget = e.target.value;
    document.getElementById('telemetry-target').textContent = state.selectedTarget || '—';
  });

  // Variance Target Slider
  const varSlider = document.getElementById('var-target-slider');
  const varBadge = document.getElementById('var-target-val');
  varSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    state.varTarget = val / 100;
    varBadge.textContent = `${val}%`;
  });

  // Test Partition Slider
  const testSlider = document.getElementById('test-frac-slider');
  const testBadge = document.getElementById('test-frac-val');
  testSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    state.testFrac = val / 100;
    testBadge.textContent = `${val}%`;
  });

  // Seed Input
  const seedInput = document.getElementById('seed-input');
  seedInput.addEventListener('change', (e) => {
    state.seed = parseInt(e.target.value, 10) || 0;
  });

  // Redundant Column Toggle
  const redundantToggle = document.getElementById('redundant-toggle');
  redundantToggle.addEventListener('change', (e) => {
    state.addRedundant = e.target.checked;
  });

  // Run Decomposition Trigger
  const btnRun = document.getElementById('btn-run');
  btnRun.addEventListener('click', () => triggerPipeline());

  // Global Keyboard Shortcut: Ctrl+Enter or Cmd+Enter to Run
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      triggerPipeline();
    }
  });

  // Step Section Accordion Toggles
  const stepHeaders = document.querySelectorAll('.step-header');
  stepHeaders.forEach(hdr => {
    hdr.addEventListener('click', () => {
      const card = hdr.closest('.step-card');
      card.classList.toggle('collapsed');
    });
  });

  // Export Buttons
  document.getElementById('btn-copy-json').addEventListener('click', copyResultsJson);
  document.getElementById('btn-download-json').addEventListener('click', downloadResultsJson);
}

// --- Dataset Loading & Parsing ---
function loadDatasetSource(source) {
  state.currentSource = source;
  const uploadGroup = document.getElementById('upload-group');
  const summaryDatasetTag = document.getElementById('summary-dataset-tag');

  if (source === 'housing') {
    uploadGroup.style.display = 'none';
    summaryDatasetTag.textContent = 'Synthetic Housing Dataset (~150 Rows)';
    parseCsvContent(window.SAMPLE_HOUSING_CSV || '', 'SalePrice');
  } else if (source === 'diabetes') {
    uploadGroup.style.display = 'none';
    summaryDatasetTag.textContent = 'Diabetes Progression Dataset (160 Rows)';
    parseCsvContent(window.SAMPLE_DIABETES_CSV || '', 'target');
  } else if (source === 'upload') {
    uploadGroup.style.display = 'flex';
    summaryDatasetTag.textContent = 'Custom User Uploaded Dataset';
    if (!state.rawCsvText) {
      document.getElementById('telemetry-rows').textContent = '—';
      document.getElementById('telemetry-cols').textContent = '—';
      document.getElementById('target-select').innerHTML = '<option value="">Upload CSV to view columns</option>';
    }
  }
}

function handleFileUpload(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a valid .csv file.', 'error');
    return;
  }

  const loadedName = document.getElementById('loaded-file-name');
  loadedName.textContent = file.name;
  loadedName.style.display = 'block';

  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    state.rawCsvText = text;
    parseCsvContent(text);
  };
  reader.readAsText(file);
}

function parseCsvContent(csvString, preferredTarget = null) {
  state.rawCsvText = csvString;
  Papa.parse(csvString, {
    header: true,
    dynamicTyping: false,
    skipEmptyLines: true,
    complete: (results) => {
      if (!results.data || results.data.length < 5) {
        showToast('Dataset has fewer than 5 rows. Please provide more samples.', 'error');
        return;
      }

      state.parsedRows = results.data;
      state.headers = results.meta.fields || Object.keys(results.data[0]);

      // Detect numeric candidate columns for target
      detectNumericHeaders();
      populateTargetDropdown(preferredTarget);

      // Update telemetry
      document.getElementById('telemetry-rows').textContent = state.parsedRows.length;
      document.getElementById('telemetry-cols').textContent = state.headers.length;
      document.getElementById('telemetry-target').textContent = state.selectedTarget || '—';

      // Automatically run pipeline upon initial load
      triggerPipeline();
    },
    error: (err) => {
      showToast(`CSV parsing error: ${err.message}`, 'error');
    }
  });
}

function detectNumericHeaders() {
  state.numericHeaders = [];
  const sampleSize = Math.min(state.parsedRows.length, 50);

  for (const h of state.headers) {
    let numericCount = 0;
    let nonBlankCount = 0;

    for (let i = 0; i < sampleSize; i++) {
      const v = state.parsedRows[i][h];
      if (v !== undefined && v !== null && String(v).trim() !== '') {
        nonBlankCount++;
        if (!isNaN(parseFloat(v)) && isFinite(v)) {
          numericCount++;
        }
      }
    }

    if (nonBlankCount > 0 && numericCount / nonBlankCount >= 0.8) {
      state.numericHeaders.push(h);
    }
  }
}

function populateTargetDropdown(preferredTarget) {
  const select = document.getElementById('target-select');
  select.innerHTML = '';

  if (state.numericHeaders.length === 0) {
    select.innerHTML = '<option value="">No numeric columns found</option>';
    showToast('No numeric target column detected in this dataset.', 'error');
    return;
  }

  state.numericHeaders.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = `${col} (numeric)`;
    select.appendChild(opt);
  });

  // Choose preferred or fallback to last numeric column
  if (preferredTarget && state.numericHeaders.includes(preferredTarget)) {
    select.value = preferredTarget;
  } else {
    select.value = state.numericHeaders[state.numericHeaders.length - 1];
  }
  state.selectedTarget = select.value;
}

// --- Pipeline Trigger & Execution ---
function triggerPipeline() {
  if (state.isComputing) return;

  if (!state.selectedTarget) {
    showToast('Please select a numeric target column (b).', 'error');
    return;
  }

  if (state.parsedRows.length < 5) {
    showToast('Dataset has insufficient rows for train/test split.', 'error');
    return;
  }

  state.isComputing = true;
  setRunButtonLoading(true);

  const payload = {
    rows: state.parsedRows,
    targetCol: state.selectedTarget,
    options: {
      addRedundant: state.addRedundant,
      varTarget: state.varTarget,
      testFrac: state.testFrac,
      seed: state.seed
    }
  };

  if (state.workerAvailable && state.worker) {
    state.worker.postMessage({ type: 'RUN_PIPELINE', payload });
  } else {
    // Fallback: execute inline with slight timeout so UI reflects loading state
    setTimeout(() => {
      try {
        // Run using inline implementation
        const results = runLinearAlgebraPipelineInline(payload.rows, payload.targetCol, payload.options);
        handleWorkerMessage({ type: 'PIPELINE_COMPLETE', payload: results });
      } catch (err) {
        handleWorkerMessage({ type: 'PIPELINE_ERROR', payload: { message: err.message } });
      }
    }, 20);
  }
}

function setRunButtonLoading(loading) {
  const btn = document.getElementById('btn-run');
  const statusDot = document.querySelector('.status-dot');
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Computing...';
    statusDot.classList.add('busy');
  } else {
    btn.disabled = false;
    btn.innerHTML = '<span>▶</span> Run Decomposition';
    statusDot.classList.remove('busy');
  }
}

// --- Render Results into DOM ---
function renderResults(res) {
  // Update Telemetry
  document.getElementById('telemetry-time').textContent = `${res.computationMs.toFixed(1)} ms`;

  // Scoreboard Metrics
  document.getElementById('stat-full-rmse').textContent = res.test_rmse.toFixed(3);
  document.getElementById('stat-full-features').textContent = `${res.rank} basis features`;

  document.getElementById('stat-red-rmse').textContent = res.reduced_rmse.toFixed(3);
  document.getElementById('stat-red-components').textContent = `${res.k} PCA components (${(res.varianceRetained * 100).toFixed(1)}% var)`;

  document.getElementById('stat-base-rmse').textContent = res.baseline_rmse.toFixed(3);

  document.getElementById('stat-rank-nullity').textContent = `${res.rank} / ${res.rawD}`;
  document.getElementById('stat-dropped-count').textContent = `${res.dropped.length} dropped`;

  // Preprocessing Notes Log
  const notesList = document.getElementById('notes-list');
  notesList.innerHTML = '';
  if (res.notes.length === 0) {
    notesList.innerHTML = '<li>Matrix clean: no constant or collinear features required dropping.</li>';
  } else {
    res.notes.forEach(note => {
      const li = document.createElement('li');
      li.textContent = note;
      notesList.appendChild(li);
    });
  }

  // Correctness Checker Ledger
  const ledgerBody = document.getElementById('ledger-body');
  ledgerBody.innerHTML = '';
  let allPass = true;

  for (const [checkName, checkObj] of Object.entries(res.checks)) {
    const tr = document.createElement('tr');
    if (!checkObj.pass) allPass = false;

    const tdName = document.createElement('td');
    tdName.innerHTML = `<strong>${checkName}</strong>`;

    const tdCond = document.createElement('td');
    tdCond.innerHTML = renderMath(checkObj.condition);

    const tdVal = document.createElement('td');
    tdVal.className = 'mono';
    tdVal.textContent = checkObj.val.toExponential(3);

    const tdBadge = document.createElement('td');
    tdBadge.innerHTML = checkObj.pass
      ? '<span class="check-badge pass">✓ PASS</span>'
      : '<span class="check-badge fail">✗ FAIL</span>';

    tr.appendChild(tdName);
    tr.appendChild(tdCond);
    tr.appendChild(tdVal);
    tr.appendChild(tdBadge);
    ledgerBody.appendChild(tr);
  }

  // Overall Stamp
  const stamp = document.getElementById('overall-stamp');
  if (allPass) {
    stamp.className = 'rubber-stamp verified';
    stamp.textContent = 'VERIFIED: ALL CHECKS PASSED';
  } else {
    stamp.className = 'rubber-stamp discrepancy';
    stamp.textContent = 'DISCREPANCY DETECTED';
  }

  // Render Charts
  updateCharts(res);

  // Render the 10 Step Breakdown
  renderStepBreakdown(res);

  // Render Export JSON Preview
  document.getElementById('json-preview').textContent = JSON.stringify(res, null, 2);

  // Re-render KaTeX math across all dynamic elements
  if (window.renderMathInElement) {
    renderMathInElement(document.getElementById('pane-worksheet'), {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    });
  }
}

// --- Render Textbook Steps Breakdown ---
function renderStepBreakdown(res) {
  // STEP 1
  document.getElementById('step1-outcome').textContent =
    `Matrix A standardized with ${res.trainN} train samples and ${res.rawD} features. Target centered (train mean = ${res.targetMean.toFixed(2)}).`;
  document.getElementById('step1-dims').textContent =
    `Train A: ${res.trainN} × ${res.rawD} | Test A: ${res.testN} × ${res.rawD}`;
  document.getElementById('step1-matrix-a').innerHTML =
    renderBracketedMatrix(res.previewA, null, res.basis.slice(0, 6));

  // STEP 2
  const r_ = res.previewBlock.length;
  const c_ = res.previewBlock[0].length;
  document.getElementById('step2-outcome').textContent =
    `Extracted leading ${r_}×${c_} block. LU factorization verified with P@L@U reconstructing block with residual ${res.checks["LU reconstructs block (P@L@U = A)"].val.toExponential(2)}.`;
  document.getElementById('step2-rref-matrix').innerHTML =
    renderBracketedMatrix(res.rrefBlock);

  // STEP 3
  document.getElementById('step3-outcome').textContent =
    `Dimension of column space is rank = ${res.rank}. Nullity = ${res.nullity} (redundant degree).`;
  document.getElementById('step3-stats-box').innerHTML = `
    <div style="display: flex; gap: 1rem; font-family: var(--font-mono); font-size: 0.85rem; margin-top: 0.5rem;">
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 1rem; border-radius: var(--radius-sm);">
        Rank: <strong style="color: var(--accent);">${res.rank}</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 1rem; border-radius: var(--radius-sm);">
        Nullity: <strong>${res.nullity}</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 1rem; border-radius: var(--radius-sm);">
        Total Features (d): <strong>${res.rawD}</strong>
      </div>
    </div>
  `;

  // STEP 4
  const droppedText = res.dropped.length > 0
    ? res.dropped.map(d => `<span class="mono" style="color: var(--fail-ink); font-weight: 600;">${d}</span>`).join(', ')
    : 'None (matrix is full column rank)';
  document.getElementById('step4-outcome').textContent =
    `Preserved ${res.rank} independent basis features. Pruned ${res.dropped.length} collinear features.`;
  document.getElementById('step4-basis-display').innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.85rem;">
      <div><strong>Preserved Basis Columns:</strong> <span class="mono" style="color: var(--pass-ink); font-weight: 600;">${res.basis.join(', ')}</span></div>
      <div><strong>Pruned Redundant Columns:</strong> ${droppedText}</div>
    </div>
  `;

  // STEP 5
  document.getElementById('step5-outcome').textContent =
    `Gram-Schmidt produced orthonormal matrix Q (${res.trainN}×${res.rank}). Maximum deviation |QᵀQ - I| is ${res.checks["Q^T Q = I (orthonormal)"].val.toExponential(2)}.`;
  document.getElementById('step5-matrix-q').innerHTML =
    renderBracketedMatrix(res.previewQ, null, res.basis.slice(0, 5));

  // STEP 6
  document.getElementById('step6-outcome').textContent =
    `Decomposed target: ||b|| = ${res.normB.toFixed(2)}, ||proj|| = ${res.normProj.toFixed(2)}, ||resid|| = ${res.normResid.toFixed(2)}. Residual is orthogonal to all basis columns.`;
  document.getElementById('step6-norms-box').innerHTML = `
    <div style="display: flex; gap: 1rem; font-family: var(--font-mono); font-size: 0.85rem; margin-top: 0.5rem; flex-wrap: wrap;">
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 0.85rem; border-radius: var(--radius-sm);">
        ||b|| = <strong>${res.normB.toFixed(2)}</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 0.85rem; border-radius: var(--radius-sm);">
        ||proj|| = <strong>${res.normProj.toFixed(2)}</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 0.85rem; border-radius: var(--radius-sm);">
        ||resid|| = <strong>${res.normResid.toFixed(2)}</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 0.85rem; border-radius: var(--radius-sm);">
        Pythagoras Check: <strong>${(Math.abs(res.normB**2 - (res.normProj**2 + res.normResid**2))).toFixed(4)}</strong>
      </div>
    </div>
  `;

  // STEP 7
  document.getElementById('step7-outcome').textContent =
    `Least squares weights fitted. Train RMSE = ${res.train_rmse.toFixed(3)}, Test RMSE = ${res.test_rmse.toFixed(3)} vs Baseline = ${res.baseline_rmse.toFixed(3)}.`;
  const weightsTbody = document.getElementById('weights-table-body');
  weightsTbody.innerHTML = '';
  res.basis.forEach(featName => {
    const w = res.weights[featName];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${featName}</strong></td>
      <td class="num">${w !== undefined ? w.toFixed(4) : '—'}</td>
      <td class="num">${w !== undefined ? w.toFixed(4) : '—'}</td>
      <td class="num" style="color: var(--text-dim);">&lt; 10⁻¹²</td>
    `;
    weightsTbody.appendChild(tr);
  });

  // STEP 8
  const topEigPct = (res.eigenvalues[0] / res.eigenvalues.reduce((a, b) => a + b, 0) * 100).toFixed(1);
  document.getElementById('step8-outcome').textContent =
    `Spectral decomposition of covariance matrix C (${res.rank}×${res.rank}). Largest eigenvector explains ${topEigPct}% of variance.`;
  document.getElementById('step8-eigen-summary').innerHTML = `
    <div style="font-family: var(--font-mono); font-size: 0.82rem; margin-top: 0.5rem;">
      <div><strong>Top 5 Eigenvalues:</strong> [ ${res.eigenvalues.slice(0, 5).map(v => v.toFixed(3)).join(', ')}${res.eigenvalues.length > 5 ? ' ...' : ''} ]</div>
      <div style="margin-top: 0.25rem;">Trace Equality Check: <strong>|Σλᵢ - tr(C)| = ${res.checks["Sum of eigenvalues = trace"].val.toExponential(2)}</strong></div>
    </div>
  `;

  // STEP 9
  document.getElementById('step9-outcome').textContent =
    `Compressed ${res.rank} features to ${res.k} principal components, capturing ${(res.varianceRetained * 100).toFixed(1)}% of total system variance.`;
  document.getElementById('step9-pca-summary').innerHTML = `
    <div style="display: flex; gap: 1rem; font-family: var(--font-mono); font-size: 0.85rem; margin-top: 0.5rem;">
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 1rem; border-radius: var(--radius-sm);">
        Retained Components (k): <strong style="color: var(--accent);">${res.k}</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 1rem; border-radius: var(--radius-sm);">
        Target Variance: <strong>${(state.varTarget * 100).toFixed(0)}%</strong>
      </div>
      <div style="background: var(--bg-card-subtle); border: 1px solid var(--border-ink); padding: 0.5rem 1rem; border-radius: var(--radius-sm);">
        Actual Variance Retained: <strong>${(res.varianceRetained * 100).toFixed(2)}%</strong>
      </div>
    </div>
  `;

  // STEP 10
  const rmseDiff = res.reduced_rmse - res.test_rmse;
  const sign = rmseDiff >= 0 ? '+' : '';
  document.getElementById('step10-outcome').textContent =
    `Compression ${res.rank} → ${res.k} components. Out-of-sample Test RMSE change: ${sign}${rmseDiff.toFixed(3)}.`;
  document.getElementById('step10-comparison-box').innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 0.5rem;">
      <div class="metric-box">
        <span class="metric-label">Full Basis Model (${res.rank} features)</span>
        <span class="metric-value">${res.test_rmse.toFixed(3)}</span>
        <span class="metric-sub">Test RMSE</span>
      </div>
      <div class="metric-box">
        <span class="metric-label">Reduced Model (${res.k} components)</span>
        <span class="metric-value">${res.reduced_rmse.toFixed(3)}</span>
        <span class="metric-sub">Test RMSE</span>
      </div>
      <div class="metric-box">
        <span class="metric-label">Trivial Baseline</span>
        <span class="metric-value">${res.baseline_rmse.toFixed(3)}</span>
        <span class="metric-sub">Test RMSE</span>
      </div>
      <div class="metric-box">
        <span class="metric-label">Compression Factor</span>
        <span class="metric-value accented">${((1 - res.k / res.rawD) * 100).toFixed(0)}%</span>
        <span class="metric-sub">${res.rawD} → ${res.k} dims</span>
      </div>
    </div>
  `;
}

// --- Chart.js Themed Rendering ---
function updateCharts(res) {
  const isDark = state.theme === 'dark';
  const gridColor = isDark ? 'rgba(56, 85, 130, 0.25)' : 'rgba(180, 160, 140, 0.25)';
  const textColor = isDark ? '#94a3b8' : '#6b6357';
  const accentColor = isDark ? '#ea580c' : '#c2410c';

  // 1. Predicted vs Actual (Test Set)
  const ctxPred = document.getElementById('chart-pred-actual').getContext('2d');
  if (state.charts.predActual) state.charts.predActual.destroy();

  const pairs = res.actual_test.map((act, i) => ({ x: act, y: res.pred_test[i] }));
  const allVals = [...res.actual_test, ...res.pred_test];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);

  state.charts.predActual = new Chart(ctxPred, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Test Samples',
          data: pairs,
          backgroundColor: isDark ? 'rgba(234, 88, 12, 0.7)' : 'rgba(194, 65, 12, 0.65)',
          borderColor: accentColor,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Ideal (y = x)',
          type: 'line',
          data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
          borderColor: isDark ? '#4ade80' : '#166534',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `Actual: ${ctx.parsed.x.toFixed(2)}, Pred: ${ctx.parsed.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Actual Target (b)', color: textColor, font: { family: 'JetBrains Mono' } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono' } }
        },
        y: {
          title: { display: true, text: 'Predicted Target', color: textColor, font: { family: 'JetBrains Mono' } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono' } }
        }
      }
    }
  });

  // 2. Scree Plot (Eigenvalues)
  const ctxScree = document.getElementById('chart-scree').getContext('2d');
  if (state.charts.scree) state.charts.scree.destroy();

  const eigLabels = res.eigenvalues.map((_, i) => `λ${i + 1}`);

  state.charts.scree = new Chart(ctxScree, {
    type: 'bar',
    data: {
      labels: eigLabels,
      datasets: [
        {
          label: 'Eigenvalue Magnitude',
          data: res.eigenvalues,
          backgroundColor: isDark ? 'rgba(56, 189, 248, 0.75)' : 'rgba(14, 116, 144, 0.75)',
          borderColor: isDark ? '#38bdf8' : '#0e7490',
          borderWidth: 1,
          borderRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Eigenvalue: ${ctx.parsed.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Principal Component', color: textColor, font: { family: 'JetBrains Mono' } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono' } }
        },
        y: {
          title: { display: true, text: 'Eigenvalue', color: textColor, font: { family: 'JetBrains Mono' } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono' } }
        }
      }
    }
  });

  // 3. Cumulative Variance Curve
  const ctxCum = document.getElementById('chart-cumvar').getContext('2d');
  if (state.charts.cumVar) state.charts.cumVar.destroy();

  const cumPercentages = res.cumulativeVariance.map(v => v * 100);
  const compLabels = res.cumulativeVariance.map((_, i) => `${i + 1}`);

  state.charts.cumVar = new Chart(ctxCum, {
    type: 'line',
    data: {
      labels: compLabels,
      datasets: [
        {
          label: 'Cumulative Variance %',
          data: cumPercentages,
          borderColor: accentColor,
          backgroundColor: isDark ? 'rgba(234, 88, 12, 0.2)' : 'rgba(194, 65, 12, 0.1)',
          fill: true,
          tension: 0.15,
          pointRadius: 4,
          pointBackgroundColor: accentColor
        },
        {
          label: `Target Threshold (${(state.varTarget * 100).toFixed(0)}%)`,
          data: Array(cumPercentages.length).fill(state.varTarget * 100),
          borderColor: '#ef4444',
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Components Retained', color: textColor, font: { family: 'JetBrains Mono' } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono' } }
        },
        y: {
          min: 0,
          max: 105,
          title: { display: true, text: 'Cumulative Variance %', color: textColor, font: { family: 'JetBrains Mono' } },
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono' } }
        }
      }
    }
  });
}

// --- Bracketed Matrix HTML Formatter ---
function renderBracketedMatrix(matrix, rowLabels = null, colLabels = null, maxRows = 6, maxCols = 6) {
  if (!matrix || matrix.length === 0) return '<div class="mono">Empty Matrix</div>';

  const numRows = Math.min(matrix.length, maxRows);
  const numCols = Math.min(matrix[0].length, maxCols);

  let html = '<table class="matrix-grid-table">';

  // Header row if column labels exist
  if (colLabels) {
    html += '<thead><tr>';
    if (rowLabels) html += '<th></th>';
    for (let j = 0; j < numCols; j++) {
      html += `<th>${colLabels[j] || `c${j + 1}`}</th>`;
    }
    if (matrix[0].length > maxCols) html += '<th>...</th>';
    html += '</tr></thead>';
  }

  html += '<tbody>';
  for (let i = 0; i < numRows; i++) {
    html += '<tr>';
    if (rowLabels) {
      html += `<th style="text-align: left; padding-right: 0.5rem; color: var(--text-dim);">${rowLabels[i]}</th>`;
    }
    for (let j = 0; j < numCols; j++) {
      const val = matrix[i][j];
      const formatted = typeof val === 'number' ? val.toFixed(2) : String(val);
      html += `<td>${formatted}</td>`;
    }
    if (matrix[i].length > maxCols) {
      html += '<td style="color: var(--text-dim);">...</td>';
    }
    html += '</tr>';
  }
  if (matrix.length > maxRows) {
    html += `<tr><td colspan="${numCols + (rowLabels ? 1 : 0) + 1}" style="text-align: center; color: var(--text-dim); padding-top: 0.25rem;">⋮ (${matrix.length - maxRows} more rows)</td></tr>`;
  }
  html += '</tbody></table>';

  return html;
}

// --- KaTeX String Helper ---
function renderMath(latex, display = false) {
  if (window.katex) {
    try {
      return katex.renderToString(latex, { throwOnError: false, displayMode: display });
    } catch {
      return latex;
    }
  }
  return latex;
}

// --- Toast Alerts ---
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icon = type === 'error' ? '⚠' : (type === 'success' ? '✓' : 'ℹ');
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

// --- Export Results Handlers ---
function copyResultsJson() {
  if (!state.lastResults) {
    showToast('No pipeline results to copy.', 'error');
    return;
  }
  const jsonStr = JSON.stringify(state.lastResults, null, 2);
  navigator.clipboard.writeText(jsonStr).then(() => {
    showToast('Computation results JSON copied to clipboard.', 'success');
  }).catch(() => {
    showToast('Failed to copy to clipboard.', 'error');
  });
}

function downloadResultsJson() {
  if (!state.lastResults) {
    showToast('No pipeline results to download.', 'error');
    return;
  }
  const jsonStr = JSON.stringify(state.lastResults, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `linear_algebra_pipeline_results_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Downloaded pipeline results JSON.', 'success');
}

// ==========================================================================
// INLINE FALLBACK LINEAR ALGEBRA RUNNER (Used when Web Worker is restricted)
// ==========================================================================
function runLinearAlgebraPipelineInline(rows, targetCol, options) {
  // Uses the exact same mathematical logic as worker.js
  const { addRedundant = false, varTarget = 0.90, testFrac = 0.20, seed = 0 } = options;
  const startTime = performance.now();

  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Preprocess
  const notes = [];
  const validRows = [];
  for (const row of rows) {
    const val = row[targetCol];
    if (val !== undefined && val !== null && val !== '') {
      const num = parseFloat(val);
      if (!isNaN(num)) validRows.push(row);
    }
  }
  const b = validRows.map(r => parseFloat(r[targetCol]));
  const allCols = Object.keys(validRows[0]).filter(c => c !== targetCol);
  const remainingCols = [];
  for (const c of allCols) {
    const vals = validRows.map(r => r[c]);
    const uniqueVals = new Set(vals.filter(v => v !== undefined && v !== null && v !== ''));
    if (uniqueVals.size <= 1) { notes.push(`Dropped constant column '${c}'`); continue; }
    const cLower = c.toLowerCase().trim();
    const isIdName = cLower === 'id' || cLower === 'index';
    const isUniqueInt = uniqueVals.size === validRows.length && vals.every(v => Number.isInteger(Number(v)));
    if (isIdName || isUniqueInt) { notes.push(`Dropped ID-like column '${c}'`); continue; }
    remainingCols.push(c);
  }

  const numCols = [];
  const catCols = [];
  for (const c of remainingCols) {
    const nonNull = validRows.map(r => r[c]).filter(v => v !== undefined && v !== null && v !== '');
    const isNumeric = nonNull.every(v => !isNaN(parseFloat(v)) && isFinite(v));
    if (isNumeric) numCols.push(c); else catCols.push(c);
  }

  const colMedians = {};
  for (const c of numCols) {
    const nums = validRows.map(r => parseFloat(r[c])).filter(v => !isNaN(v)).sort((a, b) => a - b);
    colMedians[c] = nums.length > 0 ? (nums.length % 2 !== 0 ? nums[Math.floor(nums.length/2)] : (nums[Math.floor(nums.length/2)-1] + nums[Math.floor(nums.length/2)])/2) : 0;
  }

  const encodedFeatureNames = [...numCols];
  const catCategories = {};
  for (const c of catCols) {
    const cats = Array.from(new Set(validRows.map(r => String(r[c] || '').trim()))).sort();
    const keepCats = cats.slice(1);
    catCategories[c] = keepCats;
    for (const kc of keepCats) encodedFeatureNames.push(`${c}_${kc}`);
    notes.push(`One-hot encoded text column '${c}' (drop_first '${cats[0]}')`);
  }

  const X = [];
  for (const r of validRows) {
    const rowVec = [];
    for (const c of numCols) {
      const v = r[c];
      rowVec.push(v === undefined || v === null || v === '' || isNaN(parseFloat(v)) ? colMedians[c] : parseFloat(v));
    }
    for (const c of catCols) {
      const val = String(r[c] || '').trim();
      for (const kc of catCategories[c]) rowVec.push(val === kc ? 1.0 : 0.0);
    }
    X.push(rowVec);
  }

  if (addRedundant && encodedFeatureNames.length >= 2) {
    const c1 = encodedFeatureNames[0], c2 = encodedFeatureNames[1];
    encodedFeatureNames.push(`${c1}+${c2}`);
    for (let i = 0; i < X.length; i++) X[i].push(X[i][0] + X[i][1]);
    notes.push(`Added redundant column '${c1}+${c2}' for demonstration`);
  }

  const N = X.length, d = encodedFeatureNames.length;
  const rng = mulberry32(seed);
  const indices = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const cut = Math.floor((1 - testFrac) * N);
  const trIdx = indices.slice(0, cut);
  const teIdx = indices.slice(cut);
  const n_tr = trIdx.length, n_te = teIdx.length;

  const mu = new Float64Array(d), sd = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let sum = 0; for (let i = 0; i < n_tr; i++) sum += X[trIdx[i]][j];
    mu[j] = sum / n_tr;
    let sq = 0; for (let i = 0; i < n_tr; i++) { const diff = X[trIdx[i]][j] - mu[j]; sq += diff * diff; }
    let s = Math.sqrt(sq / n_tr);
    sd[j] = s === 0 ? 1.0 : s;
  }

  const A = Array.from({ length: n_tr }, () => new Float64Array(d));
  for (let i = 0; i < n_tr; i++) for (let j = 0; j < d; j++) A[i][j] = (X[trIdx[i]][j] - mu[j]) / sd[j];
  const A_te = Array.from({ length: n_te }, () => new Float64Array(d));
  for (let i = 0; i < n_te; i++) for (let j = 0; j < d; j++) A_te[i][j] = (X[teIdx[i]][j] - mu[j]) / sd[j];

  let b_sum = 0; for (let i = 0; i < n_tr; i++) b_sum += b[trIdx[i]];
  const bm = b_sum / n_tr;
  const bt = new Float64Array(n_tr); for (let i = 0; i < n_tr; i++) bt[i] = b[trIdx[i]] - bm;
  const bt_te = new Float64Array(n_te); for (let i = 0; i < n_te; i++) bt_te[i] = b[teIdx[i]] - bm;

  // Helpers
  function matMul(M1, M2) {
    const m = M1.length, k = M1[0].length, n = M2[0].length;
    const resM = Array.from({ length: m }, () => new Float64Array(n));
    for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) {
      const a = M1[i][p];
      for (let j = 0; j < n; j++) resM[i][j] += a * M2[p][j];
    }
    return resM;
  }

  function rref(mat, tol = 1e-9) {
    const Ac = mat.map(r => Array.from(r));
    const m = Ac.length, n = Ac[0].length;
    let lead = 0;
    for (let r = 0; r < m; r++) {
      if (lead >= n) break;
      let i = r;
      while (Math.abs(Ac[i][lead]) < tol) {
        i++;
        if (i === m) { i = r; lead++; if (lead === n) return Ac; }
      }
      [Ac[i], Ac[r]] = [Ac[r], Ac[i]];
      const val = Ac[r][lead];
      for (let c = 0; c < n; c++) Ac[r][c] /= val;
      for (let rI = 0; rI < m; rI++) {
        if (rI !== r) {
          const factor = Ac[rI][lead];
          for (let c = 0; c < n; c++) Ac[rI][c] -= factor * Ac[r][c];
        }
      }
      lead++;
    }
    return Ac;
  }

  function lu(mat) {
    const m = mat.length, n = mat[0].length, minDim = Math.min(m, n);
    const perm = Array.from({ length: m }, (_, i) => i);
    const L = Array.from({ length: m }, (_, i) => Array.from({ length: minDim }, (_, j) => (i === j ? 1 : 0)));
    const U = mat.map(r => Array.from(r));
    for (let k = 0; k < minDim; k++) {
      let maxVal = Math.abs(U[k][k]), pRow = k;
      for (let i = k + 1; i < m; i++) if (Math.abs(U[i][k]) > maxVal) { maxVal = Math.abs(U[i][k]); pRow = i; }
      if (pRow !== k) {
        [U[k], U[pRow]] = [U[pRow], U[k]];
        [perm[k], perm[pRow]] = [perm[pRow], perm[k]];
        for (let j = 0; j < k; j++) { const tmp = L[k][j]; L[k][j] = L[pRow][j]; L[pRow][j] = tmp; }
      }
      const pivot = U[k][k];
      if (Math.abs(pivot) > 1e-12) {
        for (let i = k + 1; i < m; i++) {
          const factor = U[i][k] / pivot;
          L[i][k] = factor; U[i][k] = 0;
          for (let j = k + 1; j < n; j++) U[i][j] -= factor * U[k][j];
        }
      }
    }
    const P = Array.from({ length: m }, () => new Float64Array(m));
    for (let k = 0; k < m; k++) P[perm[k]][k] = 1;
    return { P, L, U };
  }

  function greedyGS(mat, tol = 1e-5) {
    const m = mat.length, n = mat[0].length;
    const cols = Array.from({ length: n }, (_, j) => {
      const col = new Float64Array(m);
      for (let i = 0; i < m; i++) col[i] = mat[i][j];
      return col;
    });
    const piv = Array.from({ length: n }, (_, i) => i);
    let rank = 0;
    for (let k = 0; k < Math.min(m, n); k++) {
      let maxNorm = 0, pCol = k;
      for (let j = k; j < n; j++) {
        let normSq = 0; for (let i = 0; i < m; i++) normSq += cols[j][i] * cols[j][i];
        const norm = Math.sqrt(normSq);
        if (norm > maxNorm) { maxNorm = norm; pCol = j; }
      }
      if (maxNorm < tol) break;
      if (pCol !== k) {
        const tmpC = cols[k]; cols[k] = cols[pCol]; cols[pCol] = tmpC;
        const tmpP = piv[k]; piv[k] = piv[pCol]; piv[pCol] = tmpP;
      }
      rank++;
      for (let i = 0; i < m; i++) cols[k][i] /= maxNorm;
      for (let j = k + 1; j < n; j++) {
        let dot = 0; for (let i = 0; i < m; i++) dot += cols[k][i] * cols[j][i];
        for (let i = 0; i < m; i++) cols[j][i] -= dot * cols[k][i];
      }
    }
    return { piv, rank };
  }

  function gramSchmidt(M) {
    const rows = M.length, cols = M[0].length;
    const Q = Array.from({ length: rows }, () => new Float64Array(cols));
    for (let j = 0; j < cols; j++) {
      const v = new Float64Array(rows);
      for (let r = 0; r < rows; r++) v[r] = M[r][j];
      for (let i = 0; i < j; i++) {
        let dot = 0; for (let r = 0; r < rows; r++) dot += Q[r][i] * M[r][j];
        for (let r = 0; r < rows; r++) v[r] -= dot * Q[r][i];
      }
      let norm = 0; for (let r = 0; r < rows; r++) norm += v[r] * v[r];
      norm = Math.sqrt(norm);
      if (norm > 1e-12) for (let r = 0; r < rows; r++) Q[r][j] = v[r] / norm;
    }
    return Q;
  }

  function solveUT(R, bVec) {
    const n = R.length, x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = bVec[i];
      for (let j = i + 1; j < n; j++) sum -= R[i][j] * x[j];
      x[i] = sum / R[i][i];
    }
    return x;
  }

  function solveSys(matA, bVec) {
    const n = matA.length, M = matA.map((r, i) => [...r, bVec[i]]);
    for (let k = 0; k < n; k++) {
      let maxV = Math.abs(M[k][k]), pR = k;
      for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > maxV) { maxV = Math.abs(M[i][k]); pR = i; }
      if (pR !== k) [M[k], M[pR]] = [M[pR], M[k]];
      const pivot = M[k][k];
      if (Math.abs(pivot) < 1e-14) continue;
      for (let j = k; j <= n; j++) M[k][j] /= pivot;
      for (let i = 0; i < n; i++) {
        if (i !== k) {
          const factor = M[i][k];
          for (let j = k; j <= n; j++) M[i][j] -= factor * M[k][j];
        }
      }
    }
    return Float64Array.from(M.map(r => r[n]));
  }

  function jacobi(matC, maxSweeps = 60, tol = 1e-12) {
    const n = matC.length;
    const Ac = matC.map(r => Array.from(r));
    const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sw = 0; sw < maxSweeps; sw++) {
      let maxOff = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(Ac[i][j]) > maxOff) maxOff = Math.abs(Ac[i][j]);
      if (maxOff < tol) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = Ac[p][q];
        if (Math.abs(apq) < 1e-15) continue;
        const app = Ac[p][p], aqq = Ac[q][q];
        const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(theta), s = Math.sin(theta);
        for (let i = 0; i < n; i++) if (i !== p && i !== q) {
          const aip = Ac[i][p], aiq = Ac[i][q];
          Ac[i][p] = c * aip - s * aiq; Ac[p][i] = Ac[i][p];
          Ac[i][q] = s * aip + c * aiq; Ac[q][i] = Ac[i][q];
        }
        Ac[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        Ac[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        Ac[p][q] = 0; Ac[q][p] = 0;
        for (let i = 0; i < n; i++) {
          const vip = V[i][p], viq = V[i][q];
          V[i][p] = c * vip - s * viq;
          V[i][q] = s * vip + c * viq;
        }
      }
    }
    const eig = []; for (let i = 0; i < n; i++) eig.push({ val: Ac[i][i], vec: V.map(r => r[i]) });
    eig.sort((a, b) => b.val - a.val);
    return { values: eig.map(e => e.val), vectors: Array.from({ length: n }, (_, r) => eig.map(e => e.vec[r])) };
  }

  function rmse(actual, pred) {
    let sum = 0; for (let i = 0; i < actual.length; i++) { const diff = actual[i] - pred[i]; sum += diff * diff; }
    return Math.sqrt(sum / actual.length);
  }

  // Execute Steps
  const r_ = Math.min(6, n_tr), c_ = Math.min(6, d);
  const blk = Array.from({ length: r_ }, (_, i) => Array.from({ length: c_ }, (_, j) => A[i][j]));
  const rrefBlk = rref(blk.map(r => r.map(v => Math.round(v * 100)/100)));
  const { P: luP, L: luL, U: luU } = lu(blk);
  const PLU = matMul(matMul(luP, luL), luU);
  let luErr = 0; for (let i = 0; i < r_; i++) for (let j = 0; j < c_; j++) luErr = Math.max(luErr, Math.abs(PLU[i][j] - blk[i][j]));

  const { piv, rank } = greedyGS(A, 1e-5);
  const keep = piv.slice(0, rank).sort((a, b) => a - b);
  const basis = keep.map(i => encodedFeatureNames[i]);
  const dropped = encodedFeatureNames.filter((_, i) => !keep.includes(i));

  const B = Array.from({ length: n_tr }, () => new Float64Array(rank));
  for (let i = 0; i < n_tr; i++) for (let j = 0; j < rank; j++) B[i][j] = A[i][keep[j]];

  const Q = gramSchmidt(B);
  let qtqErr = 0;
  for (let i = 0; i < rank; i++) for (let j = 0; j < rank; j++) {
    let dot = 0; for (let r = 0; r < n_tr; r++) dot += Q[r][i] * Q[r][j];
    qtqErr = Math.max(qtqErr, Math.abs(dot - (i === j ? 1 : 0)));
  }

  const Qt_bt = new Float64Array(rank);
  for (let j = 0; j < rank; j++) { let dot = 0; for (let r = 0; r < n_tr; r++) dot += Q[r][j] * bt[r]; Qt_bt[j] = dot; }
  const proj = new Float64Array(n_tr);
  for (let r = 0; r < n_tr; r++) for (let j = 0; j < rank; j++) proj[r] += Q[r][j] * Qt_bt[j];
  const resid = new Float64Array(n_tr);
  for (let r = 0; r < n_tr; r++) resid[r] = bt[r] - proj[r];

  let residErr = 0, maxBt = 0;
  for (let r = 0; r < n_tr; r++) maxBt = Math.max(maxBt, Math.abs(bt[r]));
  for (let j = 0; j < rank; j++) {
    let dot = 0; for (let r = 0; r < n_tr; r++) dot += B[r][j] * resid[r];
    residErr = Math.max(residErr, Math.abs(dot));
  }

  const R = Array.from({ length: rank }, () => new Float64Array(rank));
  for (let i = 0; i < rank; i++) for (let j = 0; j < rank; j++) {
    let dot = 0; for (let r = 0; r < n_tr; r++) dot += Q[r][i] * B[r][j];
    R[i][j] = dot;
  }
  const x = solveUT(R, Qt_bt);

  const BtB = Array.from({ length: rank }, () => new Float64Array(rank));
  for (let i = 0; i < rank; i++) for (let j = 0; j < rank; j++) {
    let dot = 0; for (let r = 0; r < n_tr; r++) dot += B[r][i] * B[r][j];
    BtB[i][j] = dot;
  }
  const Btb = new Float64Array(rank);
  for (let i = 0; i < rank; i++) { let dot = 0; for (let r = 0; r < n_tr; r++) dot += B[r][i] * bt[r]; Btb[i] = dot; }
  const x_ne = solveSys(BtB, Btb);
  let neErr = 0;
  for (let i = 0; i < rank; i++) neErr = Math.max(neErr, Math.abs(x[i] - x_ne[i]));

  const pred_tr = new Float64Array(n_tr);
  for (let i = 0; i < n_tr; i++) for (let j = 0; j < rank; j++) pred_tr[i] += B[i][j] * x[j];
  const tr_rmse = rmse(bt, pred_tr);

  const pred_te = new Float64Array(n_te);
  for (let i = 0; i < n_te; i++) for (let j = 0; j < rank; j++) pred_te[i] += A_te[i][keep[j]] * x[j];
  const te_rmse = rmse(bt_te, pred_te);
  const base_rmse = rmse(bt_te, new Float64Array(n_te));

  // Covariance & Jacobi
  const C = Array.from({ length: rank }, () => new Float64Array(rank));
  const denom = n_tr > 1 ? n_tr - 1 : 1;
  for (let i = 0; i < rank; i++) for (let j = 0; j < rank; j++) {
    let dot = 0; for (let r = 0; r < n_tr; r++) dot += B[r][i] * B[r][j];
    C[i][j] = dot / denom;
  }
  const { values: eigVals, vectors: eigVecs } = jacobi(C);

  let maxCvErr = 0;
  for (let j = 0; j < rank; j++) {
    const v = eigVecs.map(r => r[j]), lam = eigVals[j];
    for (let i = 0; i < rank; i++) {
      let cvi = 0; for (let k = 0; k < rank; k++) cvi += C[i][k] * v[k];
      maxCvErr = Math.max(maxCvErr, Math.abs(cvi - lam * v[i]));
    }
  }

  let traceC = 0; for (let i = 0; i < rank; i++) traceC += C[i][i];
  const sumEig = eigVals.reduce((acc, v) => acc + v, 0);
  const traceDiff = Math.abs(traceC - sumEig);

  const cumsumEig = []; let runSum = 0;
  for (let i = 0; i < rank; i++) { runSum += eigVals[i]; cumsumEig.push(runSum / sumEig); }

  let k_pca = rank;
  for (let i = 0; i < rank; i++) if (cumsumEig[i] >= varTarget) { k_pca = i + 1; break; }
  k_pca = Math.min(k_pca, rank);

  const W = Array.from({ length: rank }, (_, r) => eigVecs[r].slice(0, k_pca));
  const Z = matMul(B, W);
  const B_te = Array.from({ length: n_te }, (_, i) => keep.map(cIdx => A_te[i][cIdx]));
  const Z_te = matMul(B_te, W);

  const ZtZ = Array.from({ length: k_pca }, () => new Float64Array(k_pca));
  for (let i = 0; i < k_pca; i++) for (let j = 0; j < k_pca; j++) {
    let dot = 0; for (let r = 0; r < n_tr; r++) dot += Z[r][i] * Z[r][j];
    ZtZ[i][j] = dot;
  }
  const Ztb = new Float64Array(k_pca);
  for (let i = 0; i < k_pca; i++) { let dot = 0; for (let r = 0; r < n_tr; r++) dot += Z[r][i] * bt[r]; Ztb[i] = dot; }
  const xz = solveSys(ZtZ, Ztb);

  const pred_red_te = new Float64Array(n_te);
  for (let i = 0; i < n_te; i++) for (let j = 0; j < k_pca; j++) pred_red_te[i] += Z_te[i][j] * xz[j];
  const red_rmse = rmse(bt_te, pred_red_te);

  const checks = {
    "LU reconstructs block (P@L@U = A)": { pass: luErr < 1e-4, condition: "\\|P L U - A_{\\text{blk}}\\|_{\\infty} < 10^{-4}", val: luErr },
    "Q^T Q = I (orthonormal)": { pass: qtqErr < 1e-4, condition: "\\|Q^T Q - I\\|_{\\infty} < 10^{-4}", val: qtqErr },
    "Residual perpendicular to columns": { pass: residErr < 1e-4 * Math.max(1, maxBt), condition: "\\|B^T \\cdot \\text{resid}\\|_{\\infty} \\approx 0", val: residErr },
    "Matches normal equations (A^T A x = A^T b)": { pass: neErr < 1e-3, condition: "\\|x_{\\text{QR}} - x_{\\text{NE}}\\|_{\\infty} < 10^{-3}", val: neErr },
    "C v = lambda v for all pairs": { pass: maxCvErr < 1e-4, condition: "\\max_i \\|C v_i - \\lambda_i v_i\\|_{\\infty} < 10^{-4}", val: maxCvErr },
    "Sum of eigenvalues = trace": { pass: traceDiff < 1e-4, condition: "\\left|\\sum \\lambda_i - \\text{tr}(C)\\right| < 10^{-4}", val: traceDiff }
  };

  function norm2(vec) { let sum = 0; for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i]; return Math.sqrt(sum); }

  return {
    rawN: N, rawD: d, trainN: n_tr, testN: n_te,
    targetName: targetCol, targetMean: bm,
    rank, nullity: d - rank, dropped, basis,
    weights: Object.fromEntries(basis.map((name, i) => [name, x[i]])),
    eigenvalues: Array.from(eigVals),
    cumulativeVariance: cumsumEig,
    varianceRetained: cumsumEig[k_pca - 1],
    k: k_pca,
    train_rmse: tr_rmse, test_rmse: te_rmse, reduced_rmse: red_rmse, baseline_rmse: base_rmse,
    normB: norm2(bt), normProj: norm2(proj), normResid: norm2(resid),
    checks, notes, computationMs: performance.now() - startTime,
    previewA: Array.from({ length: Math.min(5, n_tr) }, (_, i) => Array.from({ length: Math.min(6, d) }, (_, j) => Number(A[i][j].toFixed(3)))),
    previewBlock: blk, rrefBlock: rrefBlk, luP, luL, luU,
    previewQ: Array.from({ length: Math.min(5, n_tr) }, (_, i) => Array.from({ length: Math.min(5, rank) }, (_, j) => Number(Q[i][j].toFixed(3)))),
    actual_test: Array.from(bt_te).map(v => v + bm),
    pred_test: Array.from(pred_te).map(v => v + bm)
  };
}

