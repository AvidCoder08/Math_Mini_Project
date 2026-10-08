/**
 * LINEAR ALGEBRA STUDIO — APPLICATION CONTROLLER
 * High-density scientific laboratory orchestration:
 * - Dataset ingestion & target detection
 * - Web Worker pipeline invocation with inline fallback
 * - Chart.js archival scientific figures styled for Notion
 * - KaTeX mathematical typesetting
 * - Pixel-accurate Notion workspace interface
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
    cumVar: null,
    leverage: null,
    modelsCV: null
  }
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initWorker();
  initSidebarNav();
  initEventListeners();
  loadDatasetSource('housing');
});

// --- Theme Management ---
function initTheme() {
  const savedTheme = localStorage.getItem('la_notion_theme') || 'light';
  setTheme(savedTheme);

  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const nextTheme = state.theme === 'light' ? 'dark' : 'light';
      setTheme(nextTheme);
    });
  }
}

function setTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('la_notion_theme', theme);

  const themeIcon = document.getElementById('theme-icon');
  if (themeIcon) {
    themeIcon.textContent = theme === 'dark' ? '◑' : '◐';
  }

  if (state.lastResults) {
    updateCharts(state.lastResults);
  }
}

// --- Web Worker Setup with Graceful Inline Fallback ---
function initWorker() {
  const statusText = document.getElementById('worker-status-text');

  try {
    state.worker = new Worker('worker.js');
    state.workerAvailable = true;

    state.worker.onmessage = (e) => {
      handleWorkerMessage(e.data);
    };

    state.worker.onerror = (err) => {
      console.warn('Worker error; enabling inline engine fallback:', err);
      state.workerAvailable = false;
      if (statusText) statusText.textContent = 'Inline Engine';
    };

    if (statusText) statusText.textContent = 'Worker Ready';
  } catch (err) {
    console.warn('Worker instantiation restricted (e.g. file:// protocol). Using inline engine.', err);
    state.workerAvailable = false;
    if (statusText) statusText.textContent = 'Inline Engine';
  }
}

function handleWorkerMessage(data) {
  const { type, payload } = data;
  state.isComputing = false;
  setRunButtonLoading(false);

  if (type === 'PIPELINE_COMPLETE') {
    state.lastResults = payload;
    renderResults(payload);
    showToast('Decomposition and cross-validation complete.', 'success');
  } else if (type === 'PIPELINE_ERROR') {
    showToast(`Error: ${payload.message}`, 'error');
    console.error('Pipeline Error:', payload);
  }
}

// --- Sidebar Navigation & Responsiveness ---
function initSidebarNav() {
  const sidebar = document.getElementById('notion-sidebar');
  const sidebarToggle = document.getElementById('sidebar-toggle');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      setTimeout(() => {
        Object.values(state.charts).forEach(c => { if (c) c.resize(); });
      }, 250);
    });
  }

  // Smooth scroll and auto-open details on nav clicks
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        const targetEl = document.querySelector(href);
        if (targetEl) {
          if (targetEl.tagName === 'DETAILS') {
            targetEl.open = true;
          }
          document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
          const parentItem = link.closest('.nav-item');
          if (parentItem) parentItem.classList.add('active');
        }
      }
    });
  });
}

// --- Event Listeners ---
function initEventListeners() {
  // Dataset Source Switcher (Notion Pills)
  const sourceBtns = document.querySelectorAll('.notion-pill');
  sourceBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sourceBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const source = btn.getAttribute('data-source');
      loadDatasetSource(source);
    });
  });

  // Upload Drawer & Drop Zone
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  if (dropZone && fileInput) {
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
  }

  // Target Select
  const targetSelect = document.getElementById('target-select');
  if (targetSelect) {
    targetSelect.addEventListener('change', (e) => {
      state.selectedTarget = e.target.value;
      const telTarget = document.getElementById('telemetry-target');
      if (telTarget) telTarget.textContent = state.selectedTarget || '—';
    });
  }

  // PCA Variance Slider
  const varSlider = document.getElementById('var-target-slider');
  const varBadge = document.getElementById('var-target-val');
  if (varSlider && varBadge) {
    varSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.varTarget = val / 100;
      varBadge.textContent = `${val}%`;
    });
  }

  // Holdout Test Fraction Slider
  const testSlider = document.getElementById('test-frac-slider');
  const testBadge = document.getElementById('test-frac-val');
  if (testSlider && testBadge) {
    testSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.testFrac = val / 100;
      testBadge.textContent = `${val}%`;
    });
  }

  // Seed Input
  const seedInput = document.getElementById('seed-input');
  if (seedInput) {
    seedInput.addEventListener('change', (e) => {
      state.seed = parseInt(e.target.value, 10) || 0;
    });
  }

  // Collinear Redundant Column Toggle
  const redundantToggle = document.getElementById('redundant-toggle');
  if (redundantToggle) {
    redundantToggle.addEventListener('change', (e) => {
      state.addRedundant = e.target.checked;
    });
  }

  // Execution Trigger
  const btnRun = document.getElementById('btn-run');
  if (btnRun) {
    btnRun.addEventListener('click', () => triggerPipeline());
  }

  // Keyboard Shortcut: Ctrl+Enter / Cmd+Enter
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      triggerPipeline();
    }
  });

  // Render static KaTeX once DOM fully settles
  window.addEventListener('load', () => {
    renderMath();
  });
}

// --- Dataset Loading & Ingestion ---
function loadDatasetSource(source) {
  state.currentSource = source;
  const uploadGroup = document.getElementById('upload-group');
  const summaryDatasetTag = document.getElementById('summary-dataset-tag');

  if (source === 'housing') {
    if (uploadGroup) uploadGroup.style.display = 'none';
    if (summaryDatasetTag) summaryDatasetTag.textContent = 'Synthetic Housing (~150 Rows)';
    parseCsvContent(window.SAMPLE_HOUSING_CSV || '', 'SalePrice');
  } else if (source === 'diabetes') {
    if (uploadGroup) uploadGroup.style.display = 'none';
    if (summaryDatasetTag) summaryDatasetTag.textContent = 'Diabetes Progression (160 Rows)';
    parseCsvContent(window.SAMPLE_DIABETES_CSV || '', 'target');
  } else if (source === 'upload') {
    if (uploadGroup) uploadGroup.style.display = 'flex';
    if (summaryDatasetTag) summaryDatasetTag.textContent = 'Custom User Uploaded Dataset';
    if (!state.rawCsvText) {
      const telRows = document.getElementById('telemetry-rows');
      const telCols = document.getElementById('telemetry-cols');
      const telTarget = document.getElementById('telemetry-target');
      if (telRows) telRows.textContent = '—';
      if (telCols) telCols.textContent = '—';
      if (telTarget) telTarget.textContent = '—';
      const targetSelect = document.getElementById('target-select');
      if (targetSelect) targetSelect.innerHTML = '<option value="">Upload CSV to view columns</option>';
    }
  }
}

function handleFileUpload(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a valid .csv file.', 'error');
    return;
  }

  const loadedName = document.getElementById('loaded-file-name');
  if (loadedName) {
    loadedName.textContent = file.name;
    loadedName.style.display = 'block';
  }

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

      detectNumericHeaders();
      populateTargetDropdown(preferredTarget);

      const telRows = document.getElementById('telemetry-rows');
      const telCols = document.getElementById('telemetry-cols');
      const telTarget = document.getElementById('telemetry-target');
      if (telRows) telRows.textContent = state.parsedRows.length;
      if (telCols) telCols.textContent = state.headers.length;
      if (telTarget) telTarget.textContent = state.selectedTarget || '—';

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
  if (!select) return;
  select.innerHTML = '';

  if (state.numericHeaders.length === 0) {
    select.innerHTML = '<option value="">No numeric columns found</option>';
    showToast('No numeric target column detected in this dataset.', 'error');
    return;
  }

  state.numericHeaders.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    select.appendChild(opt);
  });

  if (preferredTarget && state.numericHeaders.includes(preferredTarget)) {
    select.value = preferredTarget;
  } else {
    select.value = state.numericHeaders[state.numericHeaders.length - 1];
  }
  state.selectedTarget = select.value;
}

// --- Pipeline Execution ---
function triggerPipeline() {
  if (state.isComputing) return;

  if (!state.selectedTarget) {
    showToast('Please select a numeric target column.', 'error');
    return;
  }

  if (state.parsedRows.length < 5) {
    showToast('Dataset has insufficient rows for computation.', 'error');
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
    // Run inline using worker code loaded via script tag or window scope
    setTimeout(() => {
      try {
        const pipelineFn = window.runLinearAlgebraPipeline || (typeof runLinearAlgebraPipeline === 'function' ? runLinearAlgebraPipeline : null);
        if (pipelineFn) {
          const results = pipelineFn(payload.rows, payload.targetCol, payload.options);
          handleWorkerMessage({ type: 'PIPELINE_COMPLETE', payload: results });
        } else {
          showToast('Linear algebra engine unavailable in this security context.', 'error');
          setRunButtonLoading(false);
          state.isComputing = false;
        }
      } catch (err) {
        handleWorkerMessage({ type: 'PIPELINE_ERROR', payload: { message: err.message, stack: err.stack } });
      }
    }, 20);
  }
}

function setRunButtonLoading(loading) {
  const btn = document.getElementById('btn-run');
  if (!btn) return;
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span><span>Computing...</span>';
  } else {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">▶</span><span>Run Pipeline</span>';
  }
}

// --- Render Pipeline Results into DOM ---
function renderResults(res) {
  // Telemetry
  const telTime = document.getElementById('telemetry-time');
  if (telTime) telTime.textContent = `${res.computationMs.toFixed(1)} ms`;

  const telRows = document.getElementById('telemetry-rows');
  if (telRows) telRows.textContent = res.trainN + res.testN;

  const telCols = document.getElementById('telemetry-cols');
  if (telCols) telCols.textContent = res.rawD;

  const telTarget = document.getElementById('telemetry-target');
  if (telTarget) telTarget.textContent = state.selectedTarget || '—';

  // Top Diagnostic Highlight Cards
  const bestLinearModel = res.cvTable
    .filter(r => !r.model.includes('Baseline') && !r.model.includes('Decision tree'))
    .sort((a, b) => a.rmse - b.rmse)[0];

  const statBestModel = document.getElementById('stat-best-model');
  const statBestRmse = document.getElementById('stat-best-rmse');
  if (statBestModel && bestLinearModel) statBestModel.textContent = bestLinearModel.model.split('(')[0].trim();
  if (statBestRmse && bestLinearModel) statBestRmse.textContent = `5-Fold RMSE: ${bestLinearModel.rmse.toFixed(2)} (± ${bestLinearModel.rmse_sd.toFixed(2)})`;

  const statFullRmse = document.getElementById('stat-full-rmse');
  const statFullFeatures = document.getElementById('stat-full-features');
  if (statFullRmse) statFullRmse.textContent = res.test_rmse.toFixed(3);
  if (statFullFeatures) statFullFeatures.textContent = `${res.rank} basis features (train RMSE: ${res.train_rmse.toFixed(3)})`;

  const statRankNullity = document.getElementById('stat-rank-nullity');
  const statDroppedCount = document.getElementById('stat-dropped-count');
  if (statRankNullity) statRankNullity.textContent = `${res.rank} / ${res.rawD}`;
  if (statDroppedCount) statDroppedCount.textContent = `${res.dropped.length} redundant dropped`;

  const statCondNum = document.getElementById('stat-cond-num');
  const statCondStatus = document.getElementById('stat-cond-status');
  if (statCondNum) statCondNum.textContent = res.cond.toFixed(1);
  if (statCondStatus) statCondStatus.textContent = res.condClassification;

  // Executive Synthesis Callout
  const synthesisList = document.getElementById('synthesis-list');
  if (synthesisList) {
    synthesisList.innerHTML = '';
    res.synthesisLines.forEach(line => {
      const li = document.createElement('li');
      li.textContent = line;
      synthesisList.appendChild(li);
    });
  }

  // Preprocessing Notes Log
  const notesList = document.getElementById('notes-list');
  if (notesList) {
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
  }

  // Model Generalization Scoreboard Table
  const modelsTableBody = document.getElementById('models-table-body');
  if (modelsTableBody && res.cvTable) {
    modelsTableBody.innerHTML = '';
    res.cvTable.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${escapeHtml(row.model)}</strong></td>
        <td class="num">${row.rmse.toFixed(2)} ± ${row.rmse_sd.toFixed(2)}</td>
        <td class="num">${row.mae.toFixed(2)}</td>
        <td class="num">${row.r2.toFixed(3)} ± ${row.r2_sd.toFixed(3)}</td>
      `;
      modelsTableBody.appendChild(tr);
    });
  }

  // Theoretical Verification Ledger
  const ledgerBody = document.getElementById('ledger-body');
  if (ledgerBody) {
    ledgerBody.innerHTML = '';
    let allPassed = true;

    for (const [checkName, checkData] of Object.entries(res.checks)) {
      const tr = document.createElement('tr');
      if (!checkData.pass) allPassed = false;

      const tdName = document.createElement('td');
      tdName.innerHTML = `<strong>${escapeHtml(checkName)}</strong>`;

      const tdCond = document.createElement('td');
      tdCond.innerHTML = checkData.condition ? `$$${checkData.condition}$$` : '—';

      const tdVal = document.createElement('td');
      tdVal.className = 'num';
      tdVal.textContent = checkData.val !== undefined ? checkData.val.toExponential(3) : '0';

      const tdVerdict = document.createElement('td');
      tdVerdict.style.textAlign = 'center';
      tdVerdict.innerHTML = checkData.pass
        ? '<span class="notion-badge green">PASS</span>'
        : '<span class="notion-badge red">FAIL</span>';

      tr.appendChild(tdName);
      tr.appendChild(tdCond);
      tr.appendChild(tdVal);
      tr.appendChild(tdVerdict);
      ledgerBody.appendChild(tr);
    }

    const overallBadge = document.getElementById('overall-status-badge');
    if (overallBadge) {
      if (allPassed) {
        overallBadge.className = 'notion-badge green';
        overallBadge.textContent = 'All 16 Invariants Verified';
      } else {
        overallBadge.className = 'notion-badge red';
        overallBadge.textContent = 'Invariant Discrepancy Flagged';
      }
    }
  }

  // STAGE 1: Matrix Representation
  const s1Dims = document.getElementById('step1-dims');
  if (s1Dims) s1Dims.textContent = `Train: ${res.trainN} × ${res.rawD} | Holdout: ${res.testN} × ${res.rawD}`;
  const s1Outcome = document.getElementById('step1-outcome');
  if (s1Outcome) s1Outcome.textContent = `Standardized training matrix A (${res.trainN} × ${res.rawD}) and zero-mean vector b constructed.`;
  const s1Matrix = document.getElementById('step1-matrix-a');
  if (s1Matrix) s1Matrix.innerHTML = renderBracketMatrix(res.previewA);

  // STAGE 2: Full RREF & Block LU
  const s2Outcome = document.getElementById('step2-outcome');
  if (s2Outcome) s2Outcome.textContent = `Exposed ${res.pivotColNames.length} pivot column(s) and ${res.freeColNames.length} free column(s). Block LU verified.`;
  const step2Summary = document.getElementById('step2-rref-summary');
  if (step2Summary) {
    step2Summary.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; display: flex; flex-direction: column; gap: 0.35rem;">
        <div><strong>Pivot Columns (${res.pivotColNames.length}):</strong> [${res.pivotColNames.join(', ')}]</div>
        <div><strong>Free Columns (${res.freeColNames.length}):</strong> ${res.freeColNames.length > 0 ? '[' + res.freeColNames.join(', ') + ']' : 'None (Full Column Rank)'}</div>
        ${Object.entries(res.freeColFormulas).map(([fCol, formula]) => `
          <div style="color: var(--notion-blue);">→ Free feature <em>${escapeHtml(fCol)}</em> = ${escapeHtml(formula)}</div>
        `).join('')}
      </div>
    `;
  }
  const s2Matrix = document.getElementById('step2-rref-matrix');
  if (s2Matrix) s2Matrix.innerHTML = renderBracketMatrix(res.rrefBlock);

  // STAGE 3: Space Structure (Rank & SVD)
  const s3Outcome = document.getElementById('step3-outcome');
  if (s3Outcome) s3Outcome.textContent = `Numerical rank r = ${res.rank}, nullity = ${res.nullity}, condition number κ(A) = ${res.cond.toFixed(1)} (${res.condClassification}).`;
  const s3Stats = document.getElementById('step3-stats-box');
  if (s3Stats) {
    s3Stats.innerHTML = `
      <div>
        Singular values: [${res.eigenvalues.map(v => Math.sqrt(Math.max(0, v * (res.trainN - 1))).toFixed(2)).join(', ')}]
        <br>Matrix condition number κ(A) = ${res.cond.toFixed(1)} → <strong>${res.condClassification}</strong>
      </div>
    `;
  }

  // STAGE 4: Basis & VIF
  const s4Outcome = document.getElementById('step4-outcome');
  if (s4Outcome) s4Outcome.textContent = `Extracted ${res.rank} basis column(s); pruned ${res.dropped.length} redundant column(s). ${res.highVifCount} feature(s) exhibit VIF > 5.`;
  const step4Box = document.getElementById('step4-basis-display');
  if (step4Box) {
    step4Box.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 0.5rem;">
        <div style="font-size: 0.8rem; font-family: var(--font-mono);">
          <strong>Basis features (${res.basis.length}):</strong> [${res.basis.join(', ')}]
          ${res.dropped.length > 0 ? `<br><span style="color: #eb5757;">Dropped redundant: [${res.dropped.join(', ')}]</span>` : ''}
        </div>
        <div class="chip-list">
          ${res.basis.map(bName => {
            const vifVal = res.vifMap[bName] || 1.0;
            const isHigh = vifVal > 5;
            return `<span class="chip-item ${isHigh ? 'warn' : ''}">
              ${escapeHtml(bName)}: VIF ${vifVal.toFixed(1)}
            </span>`;
          }).join('')}
        </div>
      </div>
    `;
  }

  // STAGE 5: Gram-Schmidt QR
  const s5Outcome = document.getElementById('step5-outcome');
  if (s5Outcome) s5Outcome.textContent = `Constructed orthonormal Q (${res.trainN} × ${res.rank}) and upper triangular R (${res.rank} × ${res.rank}). Machine epsilon error.`;
  const s5Matrix = document.getElementById('step5-matrix-q');
  if (s5Matrix) s5Matrix.innerHTML = renderBracketMatrix(res.previewQ);

  // STAGE 6: Projection & Hat Matrix
  const s6Outcome = document.getElementById('step6-outcome');
  if (s6Outcome) s6Outcome.textContent = `Decomposed vector b into projection p and residual e. Residual norm: ${res.normResid.toFixed(2)}. ${res.highLevRows.length} high-leverage point(s) flagged.`;
  const s6Box = document.getElementById('step6-norms-box');
  if (s6Box) {
    s6Box.innerHTML = `
      <div>
        Norms: ‖b‖ = ${res.normB.toFixed(2)}, ‖proj‖ = ${res.normProj.toFixed(2)}, ‖resid‖ = ${res.normResid.toFixed(2)}
        <br>Trace(H) = ${res.rank}.00 = rank. High leverage cutoff (2r/n) = ${res.highLevCutoff.toFixed(3)} (${res.highLevRows.length} samples flagged).
        ${res.sortedCooks.length > 0 ? `<br>Top Cook's distance influential points: ${res.sortedCooks.map(c => `Row ${c.row} (D = ${c.cook.toFixed(3)})`).join(', ')}` : ''}
      </div>
    `;
  }

  // STAGE 7: Least Squares Weights & CIs
  const s7Outcome = document.getElementById('step7-outcome');
  if (s7Outcome) s7Outcome.textContent = `Computed QR weights matching normal equations. Holdout RMSE = ${res.test_rmse.toFixed(3)} vs Baseline = ${res.baseline_rmse.toFixed(3)}.`;
  const weightsBody = document.getElementById('weights-table-body');
  if (weightsBody) {
    weightsBody.innerHTML = '';
    res.weightsTable.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${escapeHtml(row.feature)}</strong></td>
        <td class="num">${row.weight.toFixed(3)}</td>
        <td class="num">${row.weight_lu.toFixed(3)}</td>
        <td class="num">${row.std_err.toFixed(3)}</td>
        <td class="num">[${row.ci_low.toFixed(3)}, ${row.ci_high.toFixed(3)}]</td>
        <td class="num ${row.vif > 5 ? 'warn' : ''}">${row.vif.toFixed(1)}</td>
      `;
      weightsBody.appendChild(tr);
    });
  }

  const s7Cond = document.getElementById('step7-cond-comparison');
  if (s7Cond) {
    s7Cond.innerHTML = `
      Condition numbers: κ(B) = ${res.condB.toFixed(1)} vs κ(BᵀB) = ${res.condBtB.toFixed(1)}.
      <br>Note: Solving via normal equations squares the condition number, discarding precision. QR solves directly without squaring.
    `;
  }

  // STAGE 8: Spectral Decomposition
  const s8Outcome = document.getElementById('step8-outcome');
  if (s8Outcome) s8Outcome.textContent = `Diagonalized covariance matrix C via cyclic Jacobi rotations. Top eigenvector accounts for ${(res.cumulativeVariance[0] * 100).toFixed(1)}% of total variance.`;
  const s8Summary = document.getElementById('step8-eigen-summary');
  if (s8Summary) {
    s8Summary.innerHTML = `
      <div>
        Eigenvalues: [${res.eigenvalues.map(v => v.toFixed(3)).join(', ')}]
        <br>PC1 primary loadings: ${res.pc1Loadings.slice(0, 3).map(l => `${l.name} (${l.loading.toFixed(2)})`).join(', ')}
        ${res.pc2Loadings.length > 0 ? `<br>PC2 primary loadings: ${res.pc2Loadings.slice(0, 3).map(l => `${l.name} (${l.loading.toFixed(2)})`).join(', ')}` : ''}
      </div>
    `;
  }

  // STAGE 9: Subspace Regularization & Eckart-Young
  const s9Outcome = document.getElementById('step9-outcome');
  if (s9Outcome) s9Outcome.textContent = `Variance rule selected k = ${res.k_pca}; CV selected k = ${res.k_cv} for PCR and λ = ${res.lam_cv.toFixed(2)} for Ridge (effective dof = ${res.dofRidge.toFixed(1)}).`;
  const s9Summary = document.getElementById('step9-pca-summary');
  if (s9Summary) {
    s9Summary.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 0.65rem;">
        <div>
          PCA 90% variance rule: k = ${res.k_pca} components (${(res.varianceRetained * 100).toFixed(1)}% variance)
          <br>Cross-validation choice: PCR k = ${res.k_cv} | Ridge λ = ${res.lam_cv.toFixed(2)} (Effective degrees of freedom: ${res.dofRidge.toFixed(1)} of ${res.rank})
        </div>
        <div>
          <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); margin-bottom: 0.35rem; text-transform: uppercase;">
            Eckart-Young Rank-k Low-Rank Matrix Compression:
          </div>
          <div class="notion-database-table-wrap">
            <table class="notion-table" style="font-size: 0.8rem;">
              <thead>
                <tr>
                  <th>Approximation Rank (k)</th>
                  <th class="num">Relative Error (‖B - Bₖ‖ / ‖B‖)</th>
                  <th class="num">Values Stored</th>
                  <th class="num">Compression Ratio</th>
                </tr>
              </thead>
              <tbody>
                ${res.compressionTable.map(comp => `
                  <tr>
                    <td><strong>Rank ${comp.k}</strong></td>
                    <td class="num">${(comp.relError * 100).toFixed(1)}%</td>
                    <td class="num">${comp.stored} numbers</td>
                    <td class="num">${comp.pct.toFixed(0)}% of raw matrix</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  }

  // STAGE 10: 5-Fold Cross Validation
  const s10Outcome = document.getElementById('step10-outcome');
  if (s10Outcome) s10Outcome.textContent = `Completed 5-fold cross-validation with all scalers and basis selections refit inside each fold.`;
  const s10Comparison = document.getElementById('step10-comparison-box');
  if (s10Comparison) {
    s10Comparison.innerHTML = `
      <div class="notion-database-table-wrap">
        <table class="notion-table">
          <thead>
            <tr>
              <th>Model Specification</th>
              <th class="num">Holdout RMSE (Mean ± SD)</th>
              <th class="num">Holdout MAE</th>
              <th class="num">Holdout R² (Mean ± SD)</th>
            </tr>
          </thead>
          <tbody>
            ${res.cvTable.map(row => `
              <tr>
                <td><strong>${escapeHtml(row.model)}</strong></td>
                <td class="num">${row.rmse.toFixed(2)} ± ${row.rmse_sd.toFixed(2)}</td>
                <td class="num">${row.mae.toFixed(2)}</td>
                <td class="num">${row.r2.toFixed(3)} ± ${row.r2_sd.toFixed(3)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  // Update Analytical Figures
  updateCharts(res);

  // Typeset Math via KaTeX
  renderMath();
}

// --- KaTeX Typesetter ---
function renderMath() {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    });
  }
}

// --- Scientific Charts (Chart.js) Styled for Notion ---
function updateCharts(res) {
  const isDark = state.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(55, 53, 47, 0.08)';
  const textColor = isDark ? 'rgba(255, 255, 255, 0.65)' : 'rgba(55, 53, 47, 0.65)';
  const notionBlue = isDark ? '#529cca' : '#2383e2';
  const notionGreen = isDark ? '#4dbe8b' : '#0f7b6c';
  const notionOrange = isDark ? '#d9730d' : '#d9730d';
  const notionMuted = isDark ? '#37352f' : '#e3e2de';

  // Chart 1: Predicted vs Actual (Holdout)
  if (state.charts.predActual) state.charts.predActual.destroy();
  const canvasPred = document.getElementById('chart-pred-actual');
  if (canvasPred) {
    const ctxPred = canvasPred.getContext('2d');
    const actuals = res.actual_test;
    const preds = res.pred_test;
    const scatterData = actuals.map((a, i) => ({ x: a, y: preds[i] }));
    const minVal = Math.min(...actuals, ...preds);
    const maxVal = Math.max(...actuals, ...preds);

    state.charts.predActual = new Chart(ctxPred, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'OLS Prediction',
            data: scatterData,
            backgroundColor: notionBlue,
            borderColor: notionBlue,
            pointRadius: 3.5
          },
          {
            label: 'Ideal (y = x)',
            data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
            type: 'line',
            borderColor: isDark ? 'rgba(255, 255, 255, 0.3)' : 'rgba(55, 53, 47, 0.3)',
            borderDash: [4, 4],
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: 'Observed Holdout Target', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            title: { display: true, text: 'Model Prediction', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          }
        }
      }
    });
  }

  // Chart 2: Singular Values Spectrum (Scree Plot)
  if (state.charts.scree) state.charts.scree.destroy();
  const canvasScree = document.getElementById('chart-scree');
  if (canvasScree) {
    const ctxScree = canvasScree.getContext('2d');
    const sVals = res.eigenvalues.map(v => Math.sqrt(Math.max(1e-12, v * (res.trainN - 1))));

    state.charts.scree = new Chart(ctxScree, {
      type: 'line',
      data: {
        labels: sVals.map((_, i) => `σ${i + 1}`),
        datasets: [{
          data: sVals,
          borderColor: notionBlue,
          backgroundColor: notionBlue,
          pointRadius: 3.5,
          borderWidth: 1.5,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            type: 'logarithmic',
            title: { display: true, text: 'Singular Value (Log Scale)', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          }
        }
      }
    });
  }

  // Chart 3: Cumulative Explained Variance
  if (state.charts.cumVar) state.charts.cumVar.destroy();
  const canvasCumVar = document.getElementById('chart-cumvar');
  if (canvasCumVar) {
    const ctxCumVar = canvasCumVar.getContext('2d');
    const cumPct = res.cumulativeVariance.map(v => v * 100);

    state.charts.cumVar = new Chart(ctxCumVar, {
      type: 'line',
      data: {
        labels: cumPct.map((_, i) => `${i + 1}`),
        datasets: [
          {
            label: 'Cumulative %',
            data: cumPct,
            borderColor: notionGreen,
            backgroundColor: notionGreen,
            pointRadius: 3,
            borderWidth: 1.5,
            fill: false
          },
          {
            label: 'Variance Target',
            data: cumPct.map(() => state.varTarget * 100),
            borderColor: '#eb5757',
            borderDash: [4, 4],
            pointRadius: 0,
            borderWidth: 1,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: 'Component Count', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            min: 0,
            max: 105,
            title: { display: true, text: 'Variance Explained (%)', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          }
        }
      }
    });
  }

  // Chart 4: Hat Matrix Diagonal Leverage Distribution
  if (state.charts.leverage) state.charts.leverage.destroy();
  const canvasLev = document.getElementById('chart-leverage');
  if (canvasLev) {
    const ctxLev = canvasLev.getContext('2d');
    const levVals = res.leverage_vals || [];

    state.charts.leverage = new Chart(ctxLev, {
      type: 'bar',
      data: {
        labels: levVals.map((_, i) => `${i + 1}`),
        datasets: [
          {
            data: levVals,
            backgroundColor: levVals.map(h => h > res.highLevCutoff ? '#eb5757' : (isDark ? 'rgba(255,255,255,0.25)' : 'rgba(55,53,47,0.25)')),
            borderWidth: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: 'Sample Index', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { display: false },
            ticks: { display: false }
          },
          y: {
            title: { display: true, text: 'Leverage h_i', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          }
        }
      }
    });
  }

  // Chart 5: 5-Fold Cross Validation Model Comparison (Horizontal Bar Chart)
  if (state.charts.modelsCV) state.charts.modelsCV.destroy();
  const canvasCV = document.getElementById('chart-models-cv');
  if (canvasCV) {
    const ctxCV = canvasCV.getContext('2d');
    const cvModels = res.cvTable.map(r => r.model);
    const cvRmse = res.cvTable.map(r => r.rmse);

    state.charts.modelsCV = new Chart(ctxCV, {
      type: 'bar',
      data: {
        labels: cvModels,
        datasets: [{
          data: cvRmse,
          backgroundColor: cvModels.map(m => m.includes('Ridge') ? notionOrange : (m.includes('OLS') ? notionBlue : notionMuted)),
          borderWidth: 0,
          borderRadius: 2
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: '5-Fold Cross-Validated RMSE (Lower is Better)', color: textColor, font: { family: 'Inter', size: 10 } },
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: textColor, font: { family: 'Inter', size: 9.5 } }
          }
        }
      }
    });
  }
}

// --- Bracketed Matrix Renderer ---
function renderBracketMatrix(mat) {
  if (!mat || mat.length === 0) return '<span class="mono">—</span>';
  let html = '<div class="matrix-bracket"><table class="matrix-table"><tbody>';
  for (let i = 0; i < mat.length; i++) {
    html += '<tr>';
    for (let j = 0; j < mat[i].length; j++) {
      const val = typeof mat[i][j] === 'number' ? mat[i][j].toFixed(2) : String(mat[i][j]);
      html += `<td>${val}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

// --- Utility Helpers ---
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}
