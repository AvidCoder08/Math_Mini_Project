/**
 * LINEAR ALGEBRA PIPELINE WEB WORKER
 * Performs the complete 10-stage linear algebra pipeline in a background thread.
 * Pure JavaScript - zero external dependencies.
 */

// 1. Seeded Random Number Generator (mulberry32)
function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 2. Linear Algebra Primitive Helpers
function matMul(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = Array.from({ length: m }, () => new Float64Array(n));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      for (let j = 0; j < n; j++) {
        C[i][j] += a * B[p][j];
      }
    }
  }
  return C;
}

function transpose(A) {
  const m = A.length, n = A[0].length;
  const T = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

function norm2(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  return Math.sqrt(sum);
}

function rmse(actual, pred) {
  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = actual[i] - pred[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum / actual.length);
}

// 3. Preprocessing (Data Cleaning, Drop Constant/ID, One-Hot Encode, Median Impute)
function preprocess(rows, targetCol) {
  const notes = [];
  
  // A. Drop rows missing the target
  const validRows = [];
  for (const row of rows) {
    const val = row[targetCol];
    if (val !== undefined && val !== null && val !== '') {
      const num = parseFloat(val);
      if (!isNaN(num)) {
        validRows.push(row);
      }
    }
  }
  const droppedTargetCount = rows.length - validRows.length;
  if (droppedTargetCount > 0) {
    notes.push(`Dropped ${droppedTargetCount} rows with missing or non-numeric target '${targetCol}'`);
  }

  if (validRows.length < 5) {
    throw new Error(`Insufficient valid data rows (${validRows.length}). At least 5 rows required.`);
  }

  const b = validRows.map(r => parseFloat(r[targetCol]));

  // B. Examine feature columns
  const allCols = Object.keys(validRows[0]).filter(c => c !== targetCol);
  const remainingCols = [];

  for (const c of allCols) {
    const vals = validRows.map(r => r[c]);
    const uniqueVals = new Set(vals.filter(v => v !== undefined && v !== null && v !== ''));
    if (uniqueVals.size <= 1) {
      notes.push(`Dropped constant column '${c}' (unique values ≤ 1)`);
      continue;
    }
    const cLower = c.toLowerCase().trim();
    const isIdName = cLower === 'id' || cLower === 'index';
    const isUniqueInt = uniqueVals.size === validRows.length && vals.every(v => Number.isInteger(Number(v)));
    if (isIdName || isUniqueInt) {
      notes.push(`Dropped ID-like column '${c}'`);
      continue;
    }
    remainingCols.push(c);
  }

  if (remainingCols.length === 0) {
    throw new Error("No usable feature columns remaining after dropping constant and ID-like columns.");
  }

  // C. Categorical vs Numeric Detection
  const numCols = [];
  const catCols = [];
  for (const c of remainingCols) {
    const nonNull = validRows.map(r => r[c]).filter(v => v !== undefined && v !== null && v !== '');
    const isNumeric = nonNull.every(v => !isNaN(parseFloat(v)) && isFinite(v));
    if (isNumeric) {
      numCols.push(c);
    } else {
      catCols.push(c);
    }
  }

  // D. Column Medians for Imputation
  const colMedians = {};
  for (const c of numCols) {
    const nums = validRows
      .map(r => parseFloat(r[c]))
      .filter(v => !isNaN(v))
      .sort((a, b) => a - b);
    let med = 0;
    if (nums.length > 0) {
      const mid = Math.floor(nums.length / 2);
      med = nums.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
    }
    colMedians[c] = med;
  }

  // E. One-hot encoding (drop_first)
  const encodedFeatureNames = [...numCols];
  const catCategories = {};
  for (const c of catCols) {
    const cats = Array.from(new Set(validRows.map(r => String(r[c] || '').trim()))).sort();
    const keepCats = cats.slice(1); // drop first category
    catCategories[c] = keepCats;
    for (const kc of keepCats) {
      encodedFeatureNames.push(`${c}_${kc}`);
    }
    notes.push(`One-hot encoded '${c}' with categories [${cats.join(', ')}], dropping baseline '${cats[0]}'`);
  }

  // F. Build Dense Matrix X
  let missingImputed = 0;
  const X = [];
  for (const r of validRows) {
    const rowVec = [];
    for (const c of numCols) {
      const v = r[c];
      if (v === undefined || v === null || v === '' || isNaN(parseFloat(v))) {
        missingImputed++;
        rowVec.push(colMedians[c]);
      } else {
        rowVec.push(parseFloat(v));
      }
    }
    for (const c of catCols) {
      const val = String(r[c] || '').trim();
      const keepCats = catCategories[c];
      for (const kc of keepCats) {
        rowVec.push(val === kc ? 1.0 : 0.0);
      }
    }
    X.push(rowVec);
  }

  if (missingImputed > 0) {
    notes.push(`Imputed ${missingImputed} missing feature values using training column medians`);
  }

  return { X, b, featureNames: encodedFeatureNames, notes, validCount: validRows.length };
}

// 4. Gauss-Jordan Elimination for RREF
function rref(matrix, tol = 1e-9) {
  const A = matrix.map(row => Array.from(row));
  const m = A.length, n = A[0].length;
  let lead = 0;
  const pivots = [];

  for (let r = 0; r < m; r++) {
    if (lead >= n) break;
    let i = r;
    while (Math.abs(A[i][lead]) < tol) {
      i++;
      if (i === m) {
        i = r;
        lead++;
        if (lead === n) return { rref: A, pivots };
      }
    }
    [A[i], A[r]] = [A[r], A[i]];
    const val = A[r][lead];
    for (let c = 0; c < n; c++) A[r][c] /= val;
    for (let rowIdx = 0; rowIdx < m; rowIdx++) {
      if (rowIdx !== r) {
        const factor = A[rowIdx][lead];
        for (let c = 0; c < n; c++) A[rowIdx][c] -= factor * A[r][c];
      }
    }
    pivots.push(lead);
    lead++;
  }

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (Math.abs(A[i][j]) < tol) A[i][j] = 0;
    }
  }
  return { rref: A, pivots };
}

// 5. LU Decomposition with Partial Pivoting: A = P @ L @ U
function luDecomposition(matrix) {
  const m = matrix.length, n = matrix[0].length;
  const minDim = Math.min(m, n);
  const perm = Array.from({ length: m }, (_, i) => i);
  const L = Array.from({ length: m }, (_, i) => Array.from({ length: minDim }, (_, j) => (i === j ? 1 : 0)));
  const U = matrix.map(row => Array.from(row));

  for (let k = 0; k < minDim; k++) {
    let maxVal = Math.abs(U[k][k]);
    let pRow = k;
    for (let i = k + 1; i < m; i++) {
      if (Math.abs(U[i][k]) > maxVal) {
        maxVal = Math.abs(U[i][k]);
        pRow = i;
      }
    }

    if (pRow !== k) {
      [U[k], U[pRow]] = [U[pRow], U[k]];
      [perm[k], perm[pRow]] = [perm[pRow], perm[k]];
      for (let j = 0; j < k; j++) {
        const tmp = L[k][j];
        L[k][j] = L[pRow][j];
        L[pRow][j] = tmp;
      }
    }

    const pivot = U[k][k];
    if (Math.abs(pivot) > 1e-12) {
      for (let i = k + 1; i < m; i++) {
        const factor = U[i][k] / pivot;
        L[i][k] = factor;
        U[i][k] = 0;
        for (let j = k + 1; j < n; j++) {
          U[i][j] -= factor * U[k][j];
        }
      }
    }
  }

  // Permutation matrix P such that P @ L @ U = matrix
  const P = Array.from({ length: m }, () => new Float64Array(m));
  for (let k = 0; k < m; k++) {
    P[perm[k]][k] = 1;
  }
  return { P, L, U };
}

// 6. Basis Selection via Greedy Gram-Schmidt with Column Pivoting
function greedyGSPivoting(matrix, tol = 1e-5) {
  const m = matrix.length, n = matrix[0].length;
  const cols = Array.from({ length: n }, (_, j) => {
    const col = new Float64Array(m);
    for (let i = 0; i < m; i++) col[i] = matrix[i][j];
    return col;
  });
  const piv = Array.from({ length: n }, (_, i) => i);
  const maxRank = Math.min(m, n);
  let rank = 0;

  for (let k = 0; k < maxRank; k++) {
    let maxNorm = 0;
    let pCol = k;
    for (let j = k; j < n; j++) {
      let normSq = 0;
      for (let i = 0; i < m; i++) normSq += cols[j][i] * cols[j][i];
      const norm = Math.sqrt(normSq);
      if (norm > maxNorm) {
        maxNorm = norm;
        pCol = j;
      }
    }

    if (maxNorm < tol) break;

    if (pCol !== k) {
      const tmpC = cols[k];
      cols[k] = cols[pCol];
      cols[pCol] = tmpC;

      const tmpP = piv[k];
      piv[k] = piv[pCol];
      piv[pCol] = tmpP;
    }

    rank++;

    for (let i = 0; i < m; i++) cols[k][i] /= maxNorm;

    for (let j = k + 1; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < m; i++) dot += cols[k][i] * cols[j][i];
      for (let i = 0; i < m; i++) cols[j][i] -= dot * cols[k][i];
    }
  }

  return { piv, rank };
}

// 7. Classical Gram-Schmidt (Q)
function gramSchmidt(M) {
  const rows = M.length, cols = M[0].length;
  const Q = Array.from({ length: rows }, () => new Float64Array(cols));
  for (let j = 0; j < cols; j++) {
    const v = new Float64Array(rows);
    for (let r = 0; r < rows; r++) v[r] = M[r][j];
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let r = 0; r < rows; r++) dot += Q[r][i] * M[r][j];
      for (let r = 0; r < rows; r++) v[r] -= dot * Q[r][i];
    }
    let norm = 0;
    for (let r = 0; r < rows; r++) norm += v[r] * v[r];
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      for (let r = 0; r < rows; r++) Q[r][j] = v[r] / norm;
    }
  }
  return Q;
}

// 8. Solvers: Upper Triangular & General Linear System
function solveUpperTriangular(R, b) {
  const n = R.length;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) {
      sum -= R[i][j] * x[j];
    }
    x[i] = sum / R[i][i];
  }
  return x;
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < n; k++) {
    let maxVal = Math.abs(M[k][k]);
    let pRow = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > maxVal) {
        maxVal = Math.abs(M[i][k]);
        pRow = i;
      }
    }
    if (pRow !== k) [M[k], M[pRow]] = [M[pRow], M[k]];
    const pivot = M[k][k];
    if (Math.abs(pivot) < 1e-14) continue;
    for (let j = k; j <= n; j++) M[k][j] /= pivot;
    for (let i = 0; i < n; i++) {
      if (i !== k) {
        const factor = M[i][k];
        for (let j = k; j <= n; j++) {
          M[i][j] -= factor * M[k][j];
        }
      }
    }
  }
  return Float64Array.from(M.map(row => row[n]));
}

// 9. Cyclic Jacobi Eigenvalue Algorithm for Real Symmetric Matrices
function jacobiEigen(matrix, maxSweeps = 60, tol = 1e-12) {
  const n = matrix.length;
  const A = matrix.map(row => Array.from(row));
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxOff = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const off = Math.abs(A[i][j]);
        if (off > maxOff) maxOff = off;
      }
    }
    if (maxOff < tol) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-15) continue;

        const app = A[p][p], aqq = A[q][q];
        const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(theta), s = Math.sin(theta);

        for (let i = 0; i < n; i++) {
          if (i !== p && i !== q) {
            const aip = A[i][p];
            const aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[p][i] = A[i][p];
            A[i][q] = s * aip + c * aiq;
            A[q][i] = A[i][q];
          }
        }
        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[p][q] = 0;
        A[q][p] = 0;

        for (let i = 0; i < n; i++) {
          const vip = V[i][p];
          const viq = V[i][q];
          V[i][p] = c * vip - s * viq;
          V[i][q] = s * vip + c * viq;
        }
      }
    }
  }

  const eig = [];
  for (let i = 0; i < n; i++) {
    eig.push({
      val: A[i][i],
      vec: V.map(row => row[i])
    });
  }
  eig.sort((a, b) => b.val - a.val);

  const values = eig.map(e => e.val);
  const vectors = Array.from({ length: n }, (_, r) => eig.map(e => e.vec[r]));
  return { values, vectors };
}

// 10. Complete Linear Algebra Execution Pipeline
function runLinearAlgebraPipeline(rows, targetCol, options = {}) {
  const {
    addRedundant = false,
    varTarget = 0.90,
    testFrac = 0.20,
    seed = 0
  } = options;

  const startTime = performance.now();

  // Preprocessing
  let { X, b, featureNames, notes, validCount } = preprocess(rows, targetCol);

  // Optional redundant column demo
  if (addRedundant && featureNames.length >= 2) {
    const c1 = featureNames[0], c2 = featureNames[1];
    const redName = `${c1}+${c2}`;
    featureNames.push(redName);
    for (let i = 0; i < X.length; i++) {
      X[i].push(X[i][0] + X[i][1]);
    }
    notes.push(`Added synthetic collinear feature '${redName}' (sum of '${c1}' and '${c2}') to demonstrate basis pruning`);
  }

  const N = X.length;
  const d = featureNames.length;

  // Train / Test split with mulberry32 seeded RNG
  const rng = mulberry32(seed);
  const indices = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const cut = Math.floor((1 - testFrac) * N);
  const trIdx = indices.slice(0, cut);
  const teIdx = indices.slice(cut);
  const n_tr = trIdx.length;
  const n_te = teIdx.length;

  // Compute mean and standard deviation from training set only
  const mu = new Float64Array(d);
  const sd = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n_tr; i++) sum += X[trIdx[i]][j];
    mu[j] = sum / n_tr;

    let sq = 0;
    for (let i = 0; i < n_tr; i++) {
      const diff = X[trIdx[i]][j] - mu[j];
      sq += diff * diff;
    }
    let s = Math.sqrt(sq / n_tr);
    sd[j] = s === 0 ? 1.0 : s;
  }

  // Standardize training and testing matrices
  const A = Array.from({ length: n_tr }, () => new Float64Array(d));
  for (let i = 0; i < n_tr; i++) {
    for (let j = 0; j < d; j++) {
      A[i][j] = (X[trIdx[i]][j] - mu[j]) / sd[j];
    }
  }

  const A_te = Array.from({ length: n_te }, () => new Float64Array(d));
  for (let i = 0; i < n_te; i++) {
    for (let j = 0; j < d; j++) {
      A_te[i][j] = (X[teIdx[i]][j] - mu[j]) / sd[j];
    }
  }

  // Target centering using train target mean
  let b_sum = 0;
  for (let i = 0; i < n_tr; i++) b_sum += b[trIdx[i]];
  const bm = b_sum / n_tr;

  const bt = new Float64Array(n_tr);
  for (let i = 0; i < n_tr; i++) bt[i] = b[trIdx[i]] - bm;

  const bt_te = new Float64Array(n_te);
  for (let i = 0; i < n_te; i++) bt_te[i] = b[teIdx[i]] - bm;

  const checks = {};

  // STEP 1: Matrix Representation
  // A is n_tr x d, b has n_tr targets
  const previewA = Array.from({ length: Math.min(5, n_tr) }, (_, i) =>
    Array.from({ length: Math.min(6, d) }, (_, j) => Number(A[i][j].toFixed(3)))
  );
  const previewB = Array.from({ length: Math.min(5, n_tr) }, (_, i) => Number(bt[i].toFixed(2)));

  // STEP 2: Matrix Simplification (RREF + LU on first 6x6 block)
  const r_ = Math.min(6, n_tr);
  const c_ = Math.min(6, d);
  const blk = Array.from({ length: r_ }, (_, i) => Array.from({ length: c_ }, (_, j) => A[i][j]));

  // RREF of rounded block
  const blkRounded = blk.map(row => row.map(v => Math.round(v * 100) / 100));
  const { rref: rrefBlk } = rref(blkRounded, 1e-5);

  // LU decomposition of block
  const { P: luP, L: luL, U: luU } = luDecomposition(blk);
  const PL = matMul(luP, luL);
  const PLU = matMul(PL, luU);
  let luMaxErr = 0;
  for (let i = 0; i < r_; i++) {
    for (let j = 0; j < c_; j++) {
      luMaxErr = Math.max(luMaxErr, Math.abs(PLU[i][j] - blk[i][j]));
    }
  }
  checks["LU reconstructs block (P@L@U = A)"] = {
    pass: luMaxErr < 1e-4,
    condition: "\\|P L U - A_{\\text{blk}}\\|_{\\infty} < 10^{-4}",
    val: luMaxErr
  };

  // STEP 3 & 4: Rank, Nullity & Basis Selection (Column-Pivoted Gram-Schmidt)
  const { piv, rank } = greedyGSPivoting(A, 1e-5);
  const nullity = d - rank;

  const keep = piv.slice(0, rank).sort((a, b) => a - b);
  const basis = keep.map(i => featureNames[i]);
  const dropped = featureNames.filter((_, i) => !keep.includes(i));

  // Construct Basis Matrix B (n_tr x rank)
  const B = Array.from({ length: n_tr }, () => new Float64Array(rank));
  for (let i = 0; i < n_tr; i++) {
    for (let j = 0; j < rank; j++) {
      B[i][j] = A[i][keep[j]];
    }
  }

  // STEP 5: Gram-Schmidt Orthogonalization (Q)
  const Q = gramSchmidt(B);
  let maxQtQErr = 0;
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) {
      let dot = 0;
      for (let r = 0; r < n_tr; r++) dot += Q[r][i] * Q[r][j];
      const expected = (i === j ? 1 : 0);
      maxQtQErr = Math.max(maxQtQErr, Math.abs(dot - expected));
    }
  }
  checks["Q^T Q = I (orthonormal)"] = {
    pass: maxQtQErr < 1e-4,
    condition: "\\|Q^T Q - I\\|_{\\infty} < 10^{-4}",
    val: maxQtQErr
  };

  // STEP 6: Projection and Residual
  const Qt_bt = new Float64Array(rank);
  for (let j = 0; j < rank; j++) {
    let dot = 0;
    for (let r = 0; r < n_tr; r++) dot += Q[r][j] * bt[r];
    Qt_bt[j] = dot;
  }
  const proj = new Float64Array(n_tr);
  for (let r = 0; r < n_tr; r++) {
    for (let j = 0; j < rank; j++) {
      proj[r] += Q[r][j] * Qt_bt[j];
    }
  }
  const resid = new Float64Array(n_tr);
  for (let r = 0; r < n_tr; r++) resid[r] = bt[r] - proj[r];

  let maxBtResid = 0;
  let maxBtAbs = 0;
  for (let r = 0; r < n_tr; r++) maxBtAbs = Math.max(maxBtAbs, Math.abs(bt[r]));
  for (let j = 0; j < rank; j++) {
    let dot = 0;
    for (let r = 0; r < n_tr; r++) dot += B[r][j] * resid[r];
    maxBtResid = Math.max(maxBtResid, Math.abs(dot));
  }
  const residThreshold = 1e-4 * Math.max(1, maxBtAbs);
  checks["Residual perpendicular to columns"] = {
    pass: maxBtResid < residThreshold,
    condition: "\\|B^T \\cdot \\text{resid}\\|_{\\infty} \\approx 0",
    val: maxBtResid
  };

  const normB = norm2(bt);
  const normProj = norm2(proj);
  const normResid = norm2(resid);

  // STEP 7: Least Squares via QR vs Normal Equations
  const R = Array.from({ length: rank }, () => new Float64Array(rank));
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) {
      let dot = 0;
      for (let r = 0; r < n_tr; r++) dot += Q[r][i] * B[r][j];
      R[i][j] = dot;
    }
  }
  const x = solveUpperTriangular(R, Qt_bt);

  // Normal equations: (B^T B) x_ne = B^T bt
  const BtB = Array.from({ length: rank }, () => new Float64Array(rank));
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) {
      let dot = 0;
      for (let r = 0; r < n_tr; r++) dot += B[r][i] * B[r][j];
      BtB[i][j] = dot;
    }
  }
  const Btb = new Float64Array(rank);
  for (let i = 0; i < rank; i++) {
    let dot = 0;
    for (let r = 0; r < n_tr; r++) dot += B[r][i] * bt[r];
    Btb[i] = dot;
  }
  const x_ne = solveLinearSystem(BtB, Btb);

  let maxXDiff = 0;
  let maxXVal = 0;
  for (let i = 0; i < rank; i++) {
    maxXDiff = Math.max(maxXDiff, Math.abs(x[i] - x_ne[i]));
    maxXVal = Math.max(maxXVal, Math.abs(x[i]));
  }
  const neThreshold = 1e-3 * Math.max(1, maxXVal);
  checks["Matches normal equations (A^T A x = A^T b)"] = {
    pass: maxXDiff < neThreshold,
    condition: "\\|x_{\\text{QR}} - x_{\\text{NE}}\\|_{\\infty} < 10^{-3} \\|x\\|",
    val: maxXDiff
  };

  // Model Predictions and RMSE
  const pred_tr = new Float64Array(n_tr);
  for (let i = 0; i < n_tr; i++) {
    for (let j = 0; j < rank; j++) pred_tr[i] += B[i][j] * x[j];
  }
  const tr_rmse = rmse(bt, pred_tr);

  const pred_te = new Float64Array(n_te);
  for (let i = 0; i < n_te; i++) {
    for (let j = 0; j < rank; j++) {
      pred_te[i] += A_te[i][keep[j]] * x[j];
    }
  }
  const te_rmse = rmse(bt_te, pred_te);
  const base_rmse = rmse(bt_te, new Float64Array(n_te));

  // STEP 8: Covariance & Jacobi Eigen-decomposition
  const C = Array.from({ length: rank }, () => new Float64Array(rank));
  const denom = n_tr > 1 ? n_tr - 1 : 1;
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) {
      let dot = 0;
      for (let r = 0; r < n_tr; r++) dot += B[r][i] * B[r][j];
      C[i][j] = dot / denom;
    }
  }
  const { values: eigVals, vectors: eigVecs } = jacobiEigen(C);

  // Check C v = lambda v
  let maxCvErr = 0;
  for (let j = 0; j < rank; j++) {
    const v = eigVecs.map(row => row[j]);
    const lam = eigVals[j];
    for (let i = 0; i < rank; i++) {
      let cv_i = 0;
      for (let k = 0; k < rank; k++) cv_i += C[i][k] * v[k];
      maxCvErr = Math.max(maxCvErr, Math.abs(cv_i - lam * v[i]));
    }
  }
  checks["C v = lambda v for all pairs"] = {
    pass: maxCvErr < 1e-4,
    condition: "\\max_i \\|C v_i - \\lambda_i v_i\\|_{\\infty} < 10^{-4}",
    val: maxCvErr
  };

  // Check sum(eigenvalues) = trace(C)
  let traceC = 0;
  for (let i = 0; i < rank; i++) traceC += C[i][i];
  const sumEig = eigVals.reduce((acc, v) => acc + v, 0);
  const traceDiff = Math.abs(traceC - sumEig);
  checks["Sum of eigenvalues = trace"] = {
    pass: traceDiff < 1e-4,
    condition: "\\left|\\sum \\lambda_i - \\text{tr}(C)\\right| < 10^{-4}",
    val: traceDiff
  };

  const cumsumEig = [];
  let runSum = 0;
  for (let i = 0; i < rank; i++) {
    runSum += eigVals[i];
    cumsumEig.push(runSum / sumEig);
  }

  // STEP 9: PCA Selection
  let k_pca = rank;
  for (let i = 0; i < rank; i++) {
    if (cumsumEig[i] >= varTarget) {
      k_pca = i + 1;
      break;
    }
  }
  k_pca = Math.min(k_pca, rank);

  // Projection matrix W: rank x k_pca
  const W = Array.from({ length: rank }, (_, r) => eigVecs[r].slice(0, k_pca));
  // Z = B @ W (n_tr x k_pca)
  const Z = matMul(B, W);
  // Z_te = B_te @ W (n_te x k_pca)
  const B_te = Array.from({ length: n_te }, (_, i) => keep.map(cIdx => A_te[i][cIdx]));
  const Z_te = matMul(B_te, W);

  // STEP 10: Reduced Model vs Full Model vs Baseline
  const ZtZ = Array.from({ length: k_pca }, () => new Float64Array(k_pca));
  for (let i = 0; i < k_pca; i++) {
    for (let j = 0; j < k_pca; j++) {
      let dot = 0;
      for (let r = 0; r < n_tr; r++) dot += Z[r][i] * Z[r][j];
      ZtZ[i][j] = dot;
    }
  }
  const Ztb = new Float64Array(k_pca);
  for (let i = 0; i < k_pca; i++) {
    let dot = 0;
    for (let r = 0; r < n_tr; r++) dot += Z[r][i] * bt[r];
    Ztb[i] = dot;
  }
  const xz = solveLinearSystem(ZtZ, Ztb);

  const pred_red_te = new Float64Array(n_te);
  for (let i = 0; i < n_te; i++) {
    for (let j = 0; j < k_pca; j++) {
      pred_red_te[i] += Z_te[i][j] * xz[j];
    }
  }
  const red_rmse = rmse(bt_te, pred_red_te);

  const computationMs = performance.now() - startTime;

  return {
    rawN: N,
    rawD: d,
    trainN: n_tr,
    testN: n_te,
    targetName: targetCol,
    targetMean: bm,
    rank,
    nullity,
    dropped,
    basis,
    weights: Object.fromEntries(basis.map((name, i) => [name, x[i]])),
    eigenvalues: Array.from(eigVals),
    cumulativeVariance: cumsumEig,
    varianceRetained: cumsumEig[k_pca - 1],
    k: k_pca,
    train_rmse: tr_rmse,
    test_rmse: te_rmse,
    reduced_rmse: red_rmse,
    baseline_rmse: base_rmse,
    normB,
    normProj,
    normResid,
    checks,
    notes,
    computationMs,
    // Previews for UI display
    previewA,
    previewB,
    previewBlock: blk,
    rrefBlock: rrefBlk,
    luP,
    luL,
    luU,
    previewQ: Array.from({ length: Math.min(5, n_tr) }, (_, i) =>
      Array.from({ length: Math.min(5, rank) }, (_, j) => Number(Q[i][j].toFixed(3)))
    ),
    previewZ: Array.from({ length: Math.min(5, n_tr) }, (_, i) =>
      Array.from({ length: Math.min(5, k_pca) }, (_, j) => Number(Z[i][j].toFixed(3)))
    ),
    actual_test: Array.from(bt_te).map(v => v + bm),
    pred_test: Array.from(pred_te).map(v => v + bm),
    pred_red_test: Array.from(pred_red_te).map(v => v + bm)
  };
}

// 11. Worker Message Handler
self.onmessage = function(e) {
  const { type, payload } = e.data;
  if (type === 'RUN_PIPELINE') {
    try {
      const results = runLinearAlgebraPipeline(payload.rows, payload.targetCol, payload.options);
      self.postMessage({ type: 'PIPELINE_COMPLETE', payload: results });
    } catch (err) {
      self.postMessage({ type: 'PIPELINE_ERROR', payload: { message: err.message, stack: err.stack } });
    }
  }
};

