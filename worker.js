/**
 * LINEAR ALGEBRA PIPELINE WEB WORKER (v2)
 * Pure JavaScript implementation of the 10-stage linear algebra workflow:
 * Matrix -> RREF/LU -> Rank/SVD -> Basis/VIF -> Modified Gram-Schmidt (QR)
 * -> Projection/Leverage/Cook's D -> Least Squares (QR vs Normal Equations + SE/CI)
 * -> Spectral Decomposition (Jacobi) -> PCA / PCR / Ridge (SVD Shrinkage + Eckart-Young)
 * -> 5-Fold Cross-Validation Model Comparison & Correctness Ledger.
 * 
 * Zero external dependencies. Fast, deterministic, numerical precision.
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

// 2. Linear Algebra Vector & Matrix Primitives
function matMul(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = Array.from({ length: m }, () => new Float64Array(n));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) {
        C[i][j] += a * B[p][j];
      }
    }
  }
  return C;
}

function matVecMul(A, v) {
  const m = A.length, n = A[0].length;
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += A[i][j] * v[j];
    out[i] = sum;
  }
  return out;
}

function transpose(A) {
  const m = A.length, n = A[0].length;
  const T = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) T[j][i] = A[i][j];
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

function mae(actual, pred) {
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += Math.abs(actual[i] - pred[i]);
  return sum / actual.length;
}

function r2(actual, pred) {
  let mean = 0;
  for (let i = 0; i < actual.length; i++) mean += actual[i];
  mean /= actual.length;
  let ssTot = 0, ssRes = 0;
  for (let i = 0; i < actual.length; i++) {
    const dTot = actual[i] - mean;
    const dRes = actual[i] - pred[i];
    ssTot += dTot * dTot;
    ssRes += dRes * dRes;
  }
  return ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
}

// 3. Preprocessing (Data Cleaning, Dropping Constant/ID, One-Hot Encoding, Median Imputation)
function preprocess(rows, targetCol) {
  const notes = [];
  
  // A. Drop rows missing target or non-numeric target
  const validRows = [];
  for (const row of rows) {
    const val = row[targetCol];
    if (val !== undefined && val !== null && val !== '') {
      const num = parseFloat(val);
      if (!isNaN(num) && isFinite(num)) {
        validRows.push(row);
      }
    }
  }
  const droppedTargetCount = rows.length - validRows.length;
  if (droppedTargetCount > 0) {
    notes.push(`Dropped ${droppedTargetCount} row(s) with missing or non-numeric target '${targetCol}'.`);
  }

  if (validRows.length < 5) {
    throw new Error(`Insufficient valid data rows (${validRows.length}). At least 5 rows required.`);
  }

  const b = validRows.map(r => parseFloat(r[targetCol]));

  // B. Examine feature columns (drop constant and ID-like)
  const allCols = Object.keys(validRows[0]).filter(c => c !== targetCol);
  const remainingCols = [];

  for (const c of allCols) {
    const vals = validRows.map(r => r[c]);
    const uniqueVals = new Set(vals.filter(v => v !== undefined && v !== null && v !== ''));
    if (uniqueVals.size <= 1) {
      notes.push(`Dropped constant column '${c}' (unique values ≤ 1).`);
      continue;
    }
    const cLower = c.toLowerCase().trim();
    const isIdName = cLower === 'id' || cLower === 'index';
    const isUniqueInt = uniqueVals.size === validRows.length && vals.every(v => Number.isInteger(Number(v)));
    if (isIdName || isUniqueInt) {
      notes.push(`Dropped ID-like column '${c}'.`);
      continue;
    }
    remainingCols.push(c);
  }

  if (remainingCols.length === 0) {
    throw new Error("No usable feature columns remaining after dropping constant and ID columns.");
  }

  // C. Categorical vs Numeric Detection
  const numCols = [];
  const catCols = [];
  for (const c of remainingCols) {
    const nonNull = validRows.map(r => r[c]).filter(v => v !== undefined && v !== null && v !== '');
    const isNumeric = nonNull.every(v => !isNaN(parseFloat(v)) && isFinite(v));
    if (isNumeric) numCols.push(c);
    else catCols.push(c);
  }

  // D. Column Medians for Imputation
  const colMedians = {};
  for (const c of numCols) {
    const nums = validRows
      .map(r => parseFloat(r[c]))
      .filter(v => !isNaN(v) && isFinite(v))
      .sort((x, y) => x - y);
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
    const keepCats = cats.slice(1);
    catCategories[c] = keepCats;
    for (const kc of keepCats) {
      encodedFeatureNames.push(`${c}_${kc}`);
    }
    notes.push(`One-hot encoded '${c}' with categories [${cats.join(', ')}], dropping baseline '${cats[0]}'.`);
  }

  // F. Dense Matrix Construction
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
    notes.push(`Imputed ${missingImputed} missing feature value(s) using training column medians.`);
  }

  // Target leakage warning check
  for (let j = 0; j < encodedFeatureNames.length; j++) {
    let meanX = 0, meanB = 0;
    for (let i = 0; i < X.length; i++) {
      meanX += X[i][j];
      meanB += b[i];
    }
    meanX /= X.length;
    meanB /= b.length;
    let cov = 0, varX = 0, varB = 0;
    for (let i = 0; i < X.length; i++) {
      const dx = X[i][j] - meanX;
      const db = b[i] - meanB;
      cov += dx * db;
      varX += dx * dx;
      varB += db * db;
    }
    if (varX > 0 && varB > 0) {
      const rCorr = cov / Math.sqrt(varX * varB);
      if (Math.abs(rCorr) > 0.98) {
        notes.push(`WARNING: '${encodedFeatureNames[j]}' has correlation ${rCorr.toFixed(3)} with the target (potential data leakage).`);
      }
    }
  }

  return { X, b, featureNames: encodedFeatureNames, notes, validCount: validRows.length };
}

// 4. Gauss-Jordan Elimination for Full RREF
function rref(matrix, tol = 1e-9) {
  const m = matrix.length, n = matrix[0].length;
  const M = matrix.map(row => Array.from(row));
  let lead = 0;
  const pivots = [];

  for (let r = 0; r < m; r++) {
    if (lead >= n) break;
    let i = r;
    while (Math.abs(M[i][lead]) < tol) {
      i++;
      if (i === m) {
        i = r;
        lead++;
        if (lead === n) break;
      }
    }
    if (lead === n) break;

    const tmp = M[r];
    M[r] = M[i];
    M[i] = tmp;

    const val = M[r][lead];
    for (let c = 0; c < n; c++) M[r][c] /= val;

    for (let rowIdx = 0; rowIdx < m; rowIdx++) {
      if (rowIdx !== r) {
        const factor = M[rowIdx][lead];
        for (let c = 0; c < n; c++) M[rowIdx][c] -= factor * M[r][c];
      }
    }
    pivots.push(lead);
    lead++;
  }

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (Math.abs(M[i][j]) < tol) M[i][j] = 0;
    }
  }

  const free = [];
  for (let j = 0; j < n; j++) {
    if (!pivots.includes(j)) free.push(j);
  }

  const linearCombinations = {};
  for (const fCol of free) {
    const combo = {};
    for (let pIdx = 0; pIdx < pivots.length; pIdx++) {
      const pCol = pivots[pIdx];
      const coeff = M[pIdx][fCol];
      if (Math.abs(coeff) > tol) combo[pCol] = Math.round(coeff * 1000) / 1000;
    }
    linearCombinations[fCol] = combo;
  }

  return { rref: M, pivots, free, linearCombinations };
}

// 5. LU Decomposition with Partial Pivoting: P @ A = L @ U  =>  P @ L @ U = A
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

  const P = Array.from({ length: m }, () => new Float64Array(m));
  for (let k = 0; k < m; k++) P[perm[k]][k] = 1;
  return { P, L, U };
}

// 6. Direct One-Sided Jacobi (Hestenes) SVD: A = U S V^T to machine epsilon
function hestenesSVD(A, maxSweeps = 60, tol = 1e-12) {
  const m = A.length, n = A[0].length;
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const U = A.map(row => Array.from(row));

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxAngle = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) {
          alpha += U[i][p] * U[i][p];
          beta += U[i][q] * U[i][q];
          gamma += U[i][p] * U[i][q];
        }
        if (Math.abs(gamma) < 1e-15 || Math.abs(gamma) < tol * Math.sqrt(alpha * beta)) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        maxAngle = Math.max(maxAngle, Math.abs(s));

        for (let i = 0; i < m; i++) {
          const u_ip = U[i][p], u_iq = U[i][q];
          U[i][p] = c * u_ip - s * u_iq;
          U[i][q] = s * u_ip + c * u_iq;
        }
        for (let i = 0; i < n; i++) {
          const v_ip = V[i][p], v_iq = V[i][q];
          V[i][p] = c * v_ip - s * v_iq;
          V[i][q] = s * v_ip + c * v_iq;
        }
      }
    }
    if (maxAngle < tol) break;
  }

  const s = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let normSq = 0;
    for (let i = 0; i < m; i++) normSq += U[i][j] * U[i][j];
    s[j] = Math.sqrt(normSq);
    if (s[j] > 1e-14) {
      for (let i = 0; i < m; i++) U[i][j] /= s[j];
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[b] - s[a]);
  const sSorted = order.map(i => s[i]);
  const USorted = Array.from({ length: m }, (_, r) => order.map(c => U[r][c]));
  const VSorted = Array.from({ length: n }, (_, r) => order.map(c => V[r][c]));

  const eps = 2.220446049250313e-16;
  const thresh = sSorted[0] * Math.max(m, n) * eps;
  let rank = 0;
  for (let i = 0; i < n; i++) {
    if (sSorted[i] > thresh) rank++;
  }
  rank = Math.max(1, rank);
  const cond = sSorted[0] / (sSorted[rank - 1] > 0 ? sSorted[rank - 1] : 1e-12);

  return { U: USorted, s: sSorted, V: VSorted, rank, cond };
}

// 7. Greedy Gram-Schmidt Column Pivoting for Basis
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

// 8. Modified Gram-Schmidt with Upper Triangular R: M = Q @ R
function modifiedGramSchmidt(M) {
  const n = M.length, d = M[0].length;
  const Q = Array.from({ length: n }, () => new Float64Array(d));
  const R = Array.from({ length: d }, () => new Float64Array(d));
  const V = M.map(row => Array.from(row));

  for (let j = 0; j < d; j++) {
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let r = 0; r < n; r++) dot += Q[r][i] * V[r][j];
      R[i][j] = dot;
      for (let r = 0; r < n; r++) V[r][j] -= dot * Q[r][i];
    }
    let norm = 0;
    for (let r = 0; r < n; r++) norm += V[r][j] * V[r][j];
    norm = Math.sqrt(norm);
    R[j][j] = norm;
    if (norm > 1e-12) {
      for (let r = 0; r < n; r++) Q[r][j] = V[r][j] / norm;
    }
  }
  return { Q, R };
}

// 9. Solvers: Upper Triangular & General Linear System
function solveUpperTriangular(R, b) {
  const n = R.length;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) sum -= R[i][j] * x[j];
    x[i] = sum / (Math.abs(R[i][i]) > 1e-14 ? R[i][i] : 1e-14);
  }
  return x;
}

function invertUpperTriangular(R) {
  const n = R.length;
  const inv = Array.from({ length: n }, () => new Float64Array(n));
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n);
    e[j] = 1.0;
    const col = solveUpperTriangular(R, e);
    for (let i = 0; i < n; i++) inv[i][j] = col[i];
  }
  return inv;
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
        for (let j = k; j <= n; j++) M[i][j] -= factor * M[k][j];
      }
    }
  }
  return Float64Array.from(M.map(row => row[n]));
}

function invertMatrix(A) {
  const n = A.length;
  const M = A.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });

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
    if (Math.abs(pivot) < 1e-12) M[k][k] += 1e-6;
    const p = M[k][k];
    for (let j = 0; j < 2 * n; j++) M[k][j] /= p;
    for (let i = 0; i < n; i++) {
      if (i !== k) {
        const factor = M[i][k];
        for (let j = 0; j < 2 * n; j++) M[i][j] -= factor * M[k][j];
      }
    }
  }
  return M.map(row => row.slice(n));
}

// 10. Variance Inflation Factor (VIF)
function computeVIF(B) {
  const n = B.length, r = B[0].length;
  if (r <= 1) return [1.0];

  const corr = Array.from({ length: r }, () => new Float64Array(r));
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < r; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += B[k][i] * B[k][j];
      corr[i][j] = sum / (n - 1);
    }
  }

  const invCorr = invertMatrix(corr);
  const vif = [];
  for (let i = 0; i < r; i++) {
    vif.push(Math.max(1.0, invCorr[i][i]));
  }
  return vif;
}

// 11. Cyclic Jacobi Eigen-decomposition for Symmetric Matrix
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
    eig.push({ val: Math.max(0, A[i][i]), vec: V.map(row => row[i]) });
  }
  eig.sort((a, b) => b.val - a.val);

  const values = eig.map(e => e.val);
  const vectors = Array.from({ length: n }, (_, r) => eig.map(e => e.vec[r]));
  return { values, vectors };
}

// 12. PCR & Ridge Solvers via SVD
function solvePCR(svdB, y, k) {
  const { U, s, V } = svdB;
  const d = V.length;
  const x = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let c = 0; c < k; c++) {
      if (s[c] < 1e-12) continue;
      let uty = 0;
      for (let r = 0; r < U.length; r++) uty += U[r][c] * y[r];
      sum += V[i][c] * (uty / s[c]);
    }
    x[i] = sum;
  }
  return x;
}

function solveRidgeSVD(svdB, y, lambda) {
  const { U, s, V } = svdB;
  const d = V.length;
  const x = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let c = 0; c < s.length; c++) {
      const denom = s[c] * s[c] + lambda;
      if (denom < 1e-14) continue;
      let uty = 0;
      for (let r = 0; r < U.length; r++) uty += U[r][c] * y[r];
      sum += V[i][c] * ((s[c] / denom) * uty);
    }
    x[i] = sum;
  }
  return x;
}

const LAMBDAS = Array.from({ length: 30 }, (_, i) => Math.pow(10, -2 + (6 * i) / 29));

// 13. Inner Cross-Validation for Hyperparameter Selection (k or lambda)
function cvSelect(B, y, kind, grid, folds = 5, seed = 0) {
  const N = B.length;
  const rng = mulberry32(seed);
  const indices = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const foldSizes = Array.from({ length: folds }, () => Math.floor(N / folds));
  for (let i = 0; i < N % folds; i++) foldSizes[i]++;

  const err = Array.from({ length: folds }, () => new Float64Array(grid.length));
  let startIdx = 0;

  for (let f = 0; f < folds; f++) {
    const valIndices = indices.slice(startIdx, startIdx + foldSizes[f]);
    const trIndices = [...indices.slice(0, startIdx), ...indices.slice(startIdx + foldSizes[f])];
    startIdx += foldSizes[f];

    const n_tr = trIndices.length;
    const n_val = valIndices.length;
    const d = B[0].length;

    const B_tr = Array.from({ length: n_tr }, (_, i) => Array.from(B[trIndices[i]]));
    const y_tr = new Float64Array(n_tr);
    let ym = 0;
    for (let i = 0; i < n_tr; i++) {
      y_tr[i] = y[trIndices[i]];
      ym += y_tr[i];
    }
    ym /= n_tr;
    for (let i = 0; i < n_tr; i++) y_tr[i] -= ym;

    const mu_col = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let i = 0; i < n_tr; i++) sum += B_tr[i][j];
      mu_col[j] = sum / n_tr;
      for (let i = 0; i < n_tr; i++) B_tr[i][j] -= mu_col[j];
    }

    const B_val = Array.from({ length: n_val }, (_, i) => {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = B[valIndices[i]][j] - mu_col[j];
      return row;
    });
    const y_val = new Float64Array(n_val);
    for (let i = 0; i < n_val; i++) y_val[i] = y[valIndices[i]] - ym;

    const svd_tr = hestenesSVD(B_tr);

    for (let g = 0; g < grid.length; g++) {
      const p = grid[g];
      let x;
      if (kind === 'ridge') {
        x = solveRidgeSVD(svd_tr, y_tr, p);
      } else {
        x = solvePCR(svd_tr, y_tr, Math.min(Math.round(p), d));
      }
      const pred = matVecMul(B_val, x);
      err[f][g] = rmse(y_val, pred);
    }
  }

  const curve = new Float64Array(grid.length);
  for (let g = 0; g < grid.length; g++) {
    let sum = 0;
    for (let f = 0; f < folds; f++) sum += err[f][g];
    curve[g] = sum / folds;
  }

  let bestIdx = 0;
  for (let g = 1; g < grid.length; g++) {
    if (curve[g] < curve[bestIdx]) bestIdx = g;
  }

  return { best: grid[bestIdx], curve: Array.from(curve) };
}

// 14. Decision Tree Regressor for Non-Linear Baseline
function fitDecisionTree(X, y, maxDepth = 3, minSamplesSplit = 5) {
  function buildNode(indices, depth) {
    const n = indices.length;
    let mean = 0;
    for (const i of indices) mean += y[i];
    mean /= n;

    if (depth >= maxDepth || n < minSamplesSplit) {
      return { isLeaf: true, val: mean };
    }

    let bestFeat = -1, bestThresh = 0, bestVarianceGain = 0;
    let bestLeft = null, bestRight = null;

    const numFeatures = X[0].length;
    for (let f = 0; f < numFeatures; f++) {
      const vals = indices.map(i => X[i][f]).sort((a, b) => a - b);
      const step = Math.max(1, Math.floor(vals.length / 10));
      for (let s = 1; s < vals.length; s += step) {
        const thresh = (vals[s - 1] + vals[s]) / 2;
        const left = [], right = [];
        for (const i of indices) {
          if (X[i][f] <= thresh) left.push(i);
          else right.push(i);
        }
        if (left.length === 0 || right.length === 0) continue;

        let leftMean = 0, rightMean = 0;
        for (const i of left) leftMean += y[i];
        leftMean /= left.length;
        for (const i of right) rightMean += y[i];
        rightMean /= right.length;

        let totalSS = 0, childSS = 0;
        for (const i of indices) totalSS += (y[i] - mean) ** 2;
        for (const i of left) childSS += (y[i] - leftMean) ** 2;
        for (const i of right) childSS += (y[i] - rightMean) ** 2;

        const gain = totalSS - childSS;
        if (gain > bestVarianceGain) {
          bestVarianceGain = gain;
          bestFeat = f;
          bestThresh = thresh;
          bestLeft = left;
          bestRight = right;
        }
      }
    }

    if (bestFeat === -1 || !bestLeft || !bestRight) {
      return { isLeaf: true, val: mean };
    }

    return {
      isLeaf: false,
      feat: bestFeat,
      thresh: bestThresh,
      left: buildNode(bestLeft, depth + 1),
      right: buildNode(bestRight, depth + 1)
    };
  }

  const root = buildNode(Array.from({ length: X.length }, (_, i) => i), 0);
  return function predict(xRow) {
    let node = root;
    while (!node.isLeaf) {
      if (xRow[node.feat] <= node.thresh) node = node.left;
      else node = node.right;
    }
    return node.val;
  };
}

// 15. Complete 5-Fold Cross-Validation Across the Entire Dataset
function run5FoldCV(X, b, options = {}) {
  const { varTarget = 0.90, folds = 5, seed = 0 } = options;
  const N = X.length;
  const rng = mulberry32(seed);
  const indices = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const foldSizes = Array.from({ length: folds }, () => Math.floor(N / folds));
  for (let i = 0; i < N % folds; i++) foldSizes[i]++;

  const pcrRuleName = `PCR (${Math.round(varTarget * 100)}% variance rule)`;
  const modelNames = [
    "Baseline (predict mean)",
    "OLS (QR least squares)",
    pcrRuleName,
    "PCR (k by cross-val)",
    "Ridge via SVD (lambda by cross-val)",
    "Decision tree (non-linear)"
  ];

  const scores = {};
  for (const name of modelNames) {
    scores[name] = { rmse: [], mae: [], r2: [] };
  }

  let startIdx = 0;
  for (let f = 0; f < folds; f++) {
    const valIndices = indices.slice(startIdx, startIdx + foldSizes[f]);
    const trIndices = [...indices.slice(0, startIdx), ...indices.slice(startIdx + foldSizes[f])];
    startIdx += foldSizes[f];

    const n_tr = trIndices.length;
    const n_val = valIndices.length;
    const d = X[0].length;

    // Standardization parameters computed strictly inside training fold
    const mu = new Float64Array(d);
    const sd = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let i = 0; i < n_tr; i++) sum += X[trIndices[i]][j];
      mu[j] = sum / n_tr;
      let sq = 0;
      for (let i = 0; i < n_tr; i++) {
        const diff = X[trIndices[i]][j] - mu[j];
        sq += diff * diff;
      }
      const s = Math.sqrt(sq / n_tr);
      sd[j] = s === 0 ? 1.0 : s;
    }

    const A_tr = Array.from({ length: n_tr }, (_, i) => {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = (X[trIndices[i]][j] - mu[j]) / sd[j];
      return row;
    });
    const A_val = Array.from({ length: n_val }, (_, i) => {
      const row = new Float64Array(d);
      for (let j = 0; j < d; j++) row[j] = (X[valIndices[i]][j] - mu[j]) / sd[j];
      return row;
    });

    let bm = 0;
    for (let i = 0; i < n_tr; i++) bm += b[trIndices[i]];
    bm /= n_tr;
    const bt_tr = new Float64Array(n_tr);
    for (let i = 0; i < n_tr; i++) bt_tr[i] = b[trIndices[i]] - bm;
    const b_val_actual = valIndices.map(i => b[i]);

    // Basis selection on training fold
    const { piv, rank } = greedyGSPivoting(A_tr, 1e-5);
    const keep = piv.slice(0, rank).sort((x, y) => x - y);
    const B_tr = Array.from({ length: n_tr }, (_, i) => keep.map(c => A_tr[i][c]));
    const B_val = Array.from({ length: n_val }, (_, i) => keep.map(c => A_val[i][c]));

    // Model 1: Baseline
    const predBaseline = new Float64Array(n_val).fill(bm);
    scores[modelNames[0]].rmse.push(rmse(b_val_actual, predBaseline));
    scores[modelNames[0]].mae.push(mae(b_val_actual, predBaseline));
    scores[modelNames[0]].r2.push(r2(b_val_actual, predBaseline));

    // Model 2: OLS (QR)
    const { Q, R } = modifiedGramSchmidt(B_tr);
    const Qt_b = new Float64Array(rank);
    for (let j = 0; j < rank; j++) {
      let sum = 0;
      for (let i = 0; i < n_tr; i++) sum += Q[i][j] * bt_tr[i];
      Qt_b[j] = sum;
    }
    const x_qr = solveUpperTriangular(R, Qt_b);
    const predOLS = matVecMul(B_val, x_qr).map(v => v + bm);
    scores[modelNames[1]].rmse.push(rmse(b_val_actual, predOLS));
    scores[modelNames[1]].mae.push(mae(b_val_actual, predOLS));
    scores[modelNames[1]].r2.push(r2(b_val_actual, predOLS));

    // Model 3: PCR (Variance Rule)
    const svdB = hestenesSVD(B_tr);
    const sSq = svdB.s.map(v => v * v);
    const sumSSq = sSq.reduce((a, b) => a + b, 0);
    let runVar = 0, k_rule = rank;
    for (let i = 0; i < rank; i++) {
      runVar += sSq[i];
      if (runVar / sumSSq >= varTarget) {
        k_rule = i + 1;
        break;
      }
    }
    const x_pcr_rule = solvePCR(svdB, bt_tr, k_rule);
    const predPCRRule = matVecMul(B_val, x_pcr_rule).map(v => v + bm);
    scores[modelNames[2]].rmse.push(rmse(b_val_actual, predPCRRule));
    scores[modelNames[2]].mae.push(mae(b_val_actual, predPCRRule));
    scores[modelNames[2]].r2.push(r2(b_val_actual, predPCRRule));

    // Model 4: PCR (k by CV)
    const kGrid = Array.from({ length: rank }, (_, i) => i + 1);
    const { best: bestK } = cvSelect(B_tr, bt_tr, 'pcr', kGrid, 5, seed);
    const x_pcr_cv = solvePCR(svdB, bt_tr, bestK);
    const predPCRCV = matVecMul(B_val, x_pcr_cv).map(v => v + bm);
    scores[modelNames[3]].rmse.push(rmse(b_val_actual, predPCRCV));
    scores[modelNames[3]].mae.push(mae(b_val_actual, predPCRCV));
    scores[modelNames[3]].r2.push(r2(b_val_actual, predPCRCV));

    // Model 5: Ridge via SVD (lambda by CV)
    const { best: bestLambda } = cvSelect(B_tr, bt_tr, 'ridge', LAMBDAS, 5, seed);
    const x_ridge = solveRidgeSVD(svdB, bt_tr, bestLambda);
    const predRidge = matVecMul(B_val, x_ridge).map(v => v + bm);
    scores[modelNames[4]].rmse.push(rmse(b_val_actual, predRidge));
    scores[modelNames[4]].mae.push(mae(b_val_actual, predRidge));
    scores[modelNames[4]].r2.push(r2(b_val_actual, predRidge));

    // Model 6: Decision Tree
    const treePredict = fitDecisionTree(B_tr, bt_tr, 3, 5);
    const predTree = B_val.map(row => treePredict(row) + bm);
    scores[modelNames[5]].rmse.push(rmse(b_val_actual, predTree));
    scores[modelNames[5]].mae.push(mae(b_val_actual, predTree));
    scores[modelNames[5]].r2.push(r2(b_val_actual, predTree));
  }

  const table = [];
  for (const name of modelNames) {
    const sc = scores[name];
    const meanRmse = sc.rmse.reduce((a, b) => a + b, 0) / folds;
    const sdRmse = Math.sqrt(sc.rmse.reduce((a, b) => a + (b - meanRmse) ** 2, 0) / folds);
    const meanMae = sc.mae.reduce((a, b) => a + b, 0) / folds;
    const meanR2 = sc.r2.reduce((a, b) => a + b, 0) / folds;
    const sdR2 = Math.sqrt(sc.r2.reduce((a, b) => a + (b - meanR2) ** 2, 0) / folds);

    table.push({
      model: name,
      rmse: meanRmse,
      rmse_sd: sdRmse,
      mae: meanMae,
      r2: meanR2,
      r2_sd: sdR2
    });
  }

  return table;
}

// 16. Comprehensive Pipeline Execution
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

  // Optional synthetic collinear demonstration column
  if (addRedundant && featureNames.length >= 2) {
    const c1 = featureNames[0], c2 = featureNames[1];
    const redName = `${c1}+${c2}`;
    featureNames.push(redName);
    for (let i = 0; i < X.length; i++) {
      X[i].push(X[i][0] + X[i][1]);
    }
    notes.push(`Added synthetic collinear feature '${redName}' (${c1} + ${c2}) to demonstrate basis pruning.`);
  }

  const N = X.length;
  const d = featureNames.length;

  // Single train / test split for stage-by-stage walkthrough
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

  // Standardization: mean and std dev strictly from training set
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
    const s = Math.sqrt(sq / n_tr);
    sd[j] = s === 0 ? 1.0 : s;
  }

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

  // Target centering using training target mean
  let b_sum = 0;
  for (let i = 0; i < n_tr; i++) b_sum += b[trIdx[i]];
  const bm = b_sum / n_tr;

  const bt = new Float64Array(n_tr);
  for (let i = 0; i < n_tr; i++) bt[i] = b[trIdx[i]] - bm;

  const bt_te = new Float64Array(n_te);
  for (let i = 0; i < n_te; i++) bt_te[i] = b[teIdx[i]] - bm;

  const checks = {};

  // STEP 1: Matrix Representation
  const previewA = Array.from({ length: Math.min(5, n_tr) }, (_, i) =>
    Array.from({ length: Math.min(6, d) }, (_, j) => Number(A[i][j].toFixed(3)))
  );
  const previewB = Array.from({ length: Math.min(5, n_tr) }, (_, i) => Number(bt[i].toFixed(2)));

  // STEP 2: Matrix Simplification (Full RREF & Block LU)
  const rref_full = rref(A);
  const r_ = Math.min(6, n_tr);
  const c_ = Math.min(6, d);
  const blk = Array.from({ length: r_ }, (_, i) => Array.from({ length: c_ }, (_, j) => A[i][j]));
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

  const pivotColNames = rref_full.pivots.map(p => featureNames[p]);
  const freeColNames = rref_full.free.map(f => featureNames[f]);
  const freeColFormulas = {};
  for (const fIdx of rref_full.free) {
    const fName = featureNames[fIdx];
    const combo = rref_full.linearCombinations[fIdx];
    const parts = [];
    for (const [pIdx, coeff] of Object.entries(combo)) {
      const pName = featureNames[pIdx];
      parts.push(`${coeff} · ${pName}`);
    }
    freeColFormulas[fName] = parts.length > 0 ? parts.join(" + ") : "0";
  }

  // STEP 3: Space Structure (Rank, Nullity, SVD, Condition Number)
  const svdA = hestenesSVD(A);
  const rank = svdA.rank;
  const nullity = d - rank;
  const cond = svdA.cond;

  checks["RREF pivot count = SVD rank"] = {
    pass: rref_full.pivots.length === rank,
    condition: "\\#\\text{pivots} = \\text{rank}(A)",
    val: Math.abs(rref_full.pivots.length - rank)
  };

  const condClassification = cond < 30
    ? "Well-Conditioned"
    : (cond < 1000 ? "Moderately Collinear" : "Severely Collinear");

  // STEP 4: Basis Selection & Collinearity Pruning (Pivoted QR & VIF)
  const { piv } = greedyGSPivoting(A, 1e-5);
  const keep = piv.slice(0, rank).sort((x, y) => x - y);
  const basis = keep.map(i => featureNames[i]);
  const dropped = featureNames.filter((_, i) => !keep.includes(i));

  const B = Array.from({ length: n_tr }, () => new Float64Array(rank));
  for (let i = 0; i < n_tr; i++) {
    for (let j = 0; j < rank; j++) {
      B[i][j] = A[i][keep[j]];
    }
  }

  const B_te = Array.from({ length: n_te }, () => new Float64Array(rank));
  for (let i = 0; i < n_te; i++) {
    for (let j = 0; j < rank; j++) {
      B_te[i][j] = A_te[i][keep[j]];
    }
  }

  const vifList = computeVIF(B);
  const vifMap = {};
  const highVifCount = vifList.filter(v => v > 5).length;
  for (let i = 0; i < rank; i++) {
    vifMap[basis[i]] = vifList[i];
  }

  // STEP 5: Modified Gram-Schmidt Orthogonalization (B = Q R)
  const { Q, R } = modifiedGramSchmidt(B);

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

  const QR = matMul(Q, R);
  let maxQRErr = 0;
  for (let i = 0; i < n_tr; i++) {
    for (let j = 0; j < rank; j++) {
      maxQRErr = Math.max(maxQRErr, Math.abs(QR[i][j] - B[i][j]));
    }
  }
  checks["Q R = B (factorisation)"] = {
    pass: maxQRErr < 1e-4,
    condition: "\\|Q R - B\\|_{\\infty} < 10^{-4}",
    val: maxQRErr
  };

  // STEP 6: Projection, Hat Matrix, Leverage & Cook's Distance
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
  for (let j = 0; j < rank; j++) {
    let dot = 0;
    for (let r = 0; r < n_tr; r++) dot += B[r][j] * resid[r];
    maxBtResid = Math.max(maxBtResid, Math.abs(dot));
  }
  checks["Residual perpendicular to columns"] = {
    pass: maxBtResid < 1e-3 * Math.max(1, norm2(bt)),
    condition: "\\|B^T \\cdot \\text{resid}\\|_{\\infty} \\approx 0",
    val: maxBtResid
  };

  // Idempotence check: Q @ (Q^T @ proj) = proj
  const Qt_proj = new Float64Array(rank);
  for (let j = 0; j < rank; j++) {
    let dot = 0;
    for (let r = 0; r < n_tr; r++) dot += Q[r][j] * proj[r];
    Qt_proj[j] = dot;
  }
  let maxIdempErr = 0;
  for (let r = 0; r < n_tr; r++) {
    let p2 = 0;
    for (let j = 0; j < rank; j++) p2 += Q[r][j] * Qt_proj[j];
    maxIdempErr = Math.max(maxIdempErr, Math.abs(p2 - proj[r]));
  }
  checks["Projection is idempotent (H(Hb) = Hb)"] = {
    pass: maxIdempErr < 1e-4,
    condition: "\\|H (H b) - H b\\|_{\\infty} < 10^{-4}",
    val: maxIdempErr
  };

  // Leverage & Trace(H)
  const lev = new Float64Array(n_tr);
  let traceH = 0;
  for (let r = 0; r < n_tr; r++) {
    let h_i = 0;
    for (let j = 0; j < rank; j++) h_i += Q[r][j] * Q[r][j];
    lev[r] = h_i;
    traceH += h_i;
  }
  checks["trace(H) = rank"] = {
    pass: Math.abs(traceH - rank) < 1e-4,
    condition: "|\\text{tr}(H) - r| < 10^{-4}",
    val: Math.abs(traceH - rank)
  };

  const residNormSq = norm2(resid) ** 2;
  const s2 = residNormSq / Math.max(1, n_tr - rank);
  const highLevCutoff = (2 * rank) / n_tr;
  const highLevRows = [];
  const cooksDistance = new Float64Array(n_tr);

  for (let r = 0; r < n_tr; r++) {
    if (lev[r] > highLevCutoff) highLevRows.push(r);
    const denom = Math.max(1e-12, 1 - lev[r]);
    cooksDistance[r] = (resid[r] ** 2 / (rank * s2)) * (lev[r] / (denom * denom));
  }

  const sortedCooks = Array.from({ length: n_tr }, (_, i) => ({ row: i, cook: cooksDistance[i] }))
    .sort((a, b) => b.cook - a.cook)
    .slice(0, 3);

  const normB = norm2(bt);
  const normProj = norm2(proj);
  const normResid = norm2(resid);

  // STEP 7: Least Squares via QR vs Normal Equations + Confidence Intervals
  const x = solveUpperTriangular(R, Qt_bt);

  const BtB = matMul(transpose(B), B);
  const Btb = matVecMul(transpose(B), bt);
  const x_lu = solveLinearSystem(BtB, Btb);

  let maxLuDiff = 0, maxCoeff = 0;
  for (let i = 0; i < rank; i++) {
    maxLuDiff = Math.max(maxLuDiff, Math.abs(x[i] - x_lu[i]));
    maxCoeff = Math.max(maxCoeff, Math.abs(x[i]));
  }
  checks["QR solution = LU on normal equations"] = {
    pass: maxLuDiff < 1e-3 * Math.max(1, maxCoeff),
    condition: "\\|x_{\\text{QR}} - x_{\\text{LU}}\\|_{\\infty} < 10^{-3} \\|x\\|",
    val: maxLuDiff
  };

  const BtB_x = matVecMul(BtB, x);
  let maxNeErr = 0;
  for (let i = 0; i < rank; i++) maxNeErr = Math.max(maxNeErr, Math.abs(BtB_x[i] - Btb[i]));
  checks["Normal equations hold (B^T B x = B^T b)"] = {
    pass: maxNeErr < 1e-3 * Math.max(1, norm2(Btb)),
    condition: "\\|B^T B x - B^T b\\|_{\\infty} \\approx 0",
    val: maxNeErr
  };

  // Standard Errors & 95% Confidence Intervals
  const Rinv = invertUpperTriangular(R);
  const stdErrors = new Float64Array(rank);
  const ciLow = new Float64Array(rank);
  const ciHigh = new Float64Array(rank);

  for (let i = 0; i < rank; i++) {
    let rInvRowNormSq = 0;
    for (let k = 0; k < rank; k++) rInvRowNormSq += Rinv[i][k] * Rinv[i][k];
    const se = Math.sqrt(s2 * rInvRowNormSq);
    stdErrors[i] = se;
    ciLow[i] = x[i] - 1.96 * se;
    ciHigh[i] = x[i] + 1.96 * se;
  }

  const svdB = hestenesSVD(B);
  const condB = svdB.cond;
  const condBtB = condB * condB;

  // Single-split Model Predictions
  const pred_tr = matVecMul(B, x);
  const tr_rmse = rmse(bt, pred_tr);

  const pred_te = matVecMul(B_te, x);
  const te_rmse = rmse(bt_te, pred_te);
  const base_rmse = rmse(bt_te, new Float64Array(n_te));

  // STEP 8: Covariance Matrix & Eigen-Decomposition (Cyclic Jacobi)
  const C = Array.from({ length: rank }, () => new Float64Array(rank));
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) C[i][j] = BtB[i][j] / (n_tr - 1);
  }
  const { values: eigVals, vectors: eigVecs } = jacobiEigen(C);

  // C v = lambda v
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

  let trC = 0, sumEig = 0;
  for (let i = 0; i < rank; i++) {
    trC += C[i][i];
    sumEig += eigVals[i];
  }
  checks["Sum of eigenvalues = trace"] = {
    pass: Math.abs(sumEig - trC) < 1e-4,
    condition: "|\\sum \\lambda_i - \\text{tr}(C)| < 10^{-4}",
    val: Math.abs(sumEig - trC)
  };

  let maxEigSvdErr = 0;
  for (let i = 0; i < rank; i++) {
    const expected = (svdB.s[i] * svdB.s[i]) / (n_tr - 1);
    maxEigSvdErr = Math.max(maxEigSvdErr, Math.abs(eigVals[i] - expected));
  }
  checks["Eigenvalues = (singular values)^2 / (n-1)"] = {
    pass: maxEigSvdErr < 1e-4,
    condition: "|\\lambda_i - \\sigma_i^2 / (n-1)| < 10^{-4}",
    val: maxEigSvdErr
  };

  let maxVtVErr = 0;
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) {
      let dot = 0;
      for (let r = 0; r < rank; r++) dot += eigVecs[r][i] * eigVecs[r][j];
      maxVtVErr = Math.max(maxVtVErr, Math.abs(dot - (i === j ? 1 : 0)));
    }
  }
  checks["V^T V = I (eigenvectors orthonormal)"] = {
    pass: maxVtVErr < 1e-4,
    condition: "\\|V^T V - I\\|_{\\infty} < 10^{-4}",
    val: maxVtVErr
  };

  const VL = Array.from({ length: rank }, (_, i) => Array.from({ length: rank }, (_, j) => eigVecs[i][j] * eigVals[j]));
  const VLVt = matMul(VL, transpose(eigVecs));
  let maxSpecErr = 0;
  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < rank; j++) maxSpecErr = Math.max(maxSpecErr, Math.abs(VLVt[i][j] - C[i][j]));
  }
  checks["C = V diag(lambda) V^T (spectral theorem)"] = {
    pass: maxSpecErr < 1e-4,
    condition: "\\|V \\Lambda V^T - C\\|_{\\infty} < 10^{-4}",
    val: maxSpecErr
  };

  // Top Loadings for PC1 and PC2
  const pc1Loadings = basis.map((name, i) => ({ name, loading: eigVecs[i][0] }))
    .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading));
  const pc2Loadings = rank > 1
    ? basis.map((name, i) => ({ name, loading: eigVecs[i][1] })).sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
    : [];

  const cumsumEig = [];
  let runSum = 0;
  for (let i = 0; i < rank; i++) {
    runSum += eigVals[i];
    cumsumEig.push(sumEig > 0 ? runSum / sumEig : 1.0);
  }

  // STEP 9: PCA vs PCR vs Ridge Regression & Compression
  let k_pca = rank;
  for (let i = 0; i < rank; i++) {
    if (cumsumEig[i] >= varTarget) {
      k_pca = i + 1;
      break;
    }
  }
  k_pca = Math.min(k_pca, rank);

  const kGrid = Array.from({ length: rank }, (_, i) => i + 1);
  const { best: k_cv, curve: k_curve } = cvSelect(B, bt, 'pcr', kGrid, 5, seed);
  const { best: lam_cv, curve: lam_curve } = cvSelect(B, bt, 'ridge', LAMBDAS, 5, seed);

  const x_pcr90 = solvePCR(svdB, bt, k_pca);
  const x_pcrcv = solvePCR(svdB, bt, k_cv);
  const x_ridge = solveRidgeSVD(svdB, bt, lam_cv);

  // Check PCR with all components = OLS
  const x_pcr_full = solvePCR(svdB, bt, rank);
  let maxPcrOlsDiff = 0;
  for (let i = 0; i < rank; i++) maxPcrOlsDiff = Math.max(maxPcrOlsDiff, Math.abs(x_pcr_full[i] - x[i]));
  checks["PCR with all components = OLS"] = {
    pass: maxPcrOlsDiff < 1e-3 * Math.max(1, maxCoeff),
    condition: "\\|x_{\\text{PCR}}(r) - x_{\\text{OLS}}\\|_{\\infty} < 10^{-3} \\|x\\|",
    val: maxPcrOlsDiff
  };

  // Check Ridge matches regularized normal equations: (B^T B + lambda I) x_ridge = B^T b
  const BtB_lam = BtB.map((row, i) => row.map((v, j) => v + (i === j ? lam_cv : 0)));
  const BtB_lam_x = matVecMul(BtB_lam, x_ridge);
  let maxRidgeNeErr = 0;
  for (let i = 0; i < rank; i++) maxRidgeNeErr = Math.max(maxRidgeNeErr, Math.abs(BtB_lam_x[i] - Btb[i]));
  checks["Ridge matches regularised normal equations"] = {
    pass: maxRidgeNeErr < 1e-3 * Math.max(1, norm2(Btb)),
    condition: "\\|(B^T B + \\lambda I) x - B^T b\\|_{\\infty} \\approx 0",
    val: maxRidgeNeErr
  };

  // Effective degrees of freedom
  let dofRidge = 0;
  for (let i = 0; i < rank; i++) {
    const s_i2 = svdB.s[i] * svdB.s[i];
    dofRidge += s_i2 / (s_i2 + lam_cv);
  }

  // Eckart-Young Low-Rank Compression Table
  const bFrobNorm = Math.sqrt(B.reduce((acc, row) => acc + row.reduce((s, v) => s + v * v, 0), 0));
  const compressionRanks = Array.from(new Set([1, Math.max(1, Math.floor(rank / 2)), k_pca, rank])).sort((a, b) => a - b);
  const compressionTable = compressionRanks.map(k_comp => {
    let residualNormSq = 0;
    for (let c = k_comp; c < rank; c++) residualNormSq += svdB.s[c] * svdB.s[c];
    const relError = bFrobNorm > 0 ? Math.sqrt(residualNormSq) / bFrobNorm : 0;
    const stored = k_comp * (n_tr + rank + 1);
    const totalRaw = n_tr * rank;
    const pct = totalRaw > 0 ? (100 * stored) / totalRaw : 100;
    return {
      k: k_comp,
      relError,
      stored,
      pct
    };
  });

  // Single-split test predictions for PCA & Ridge
  const pred_pcr90_te = matVecMul(B_te, x_pcr90);
  const pred_ridge_te = matVecMul(B_te, x_ridge);
  const pcr90_te_rmse = rmse(bt_te, pred_pcr90_te);
  const ridge_te_rmse = rmse(bt_te, pred_ridge_te);

  // STEP 10: 5-Fold Cross-Validation Comparison Table & Synthesis
  const cvTable = run5FoldCV(X, b, { varTarget, folds: 5, seed });

  // Plain-Language Summary Logic
  const olsRow = cvTable.find(r => r.model.includes("OLS"));
  const pcrRuleRow = cvTable.find(r => r.model.includes("variance rule"));
  const ridgeRow = cvTable.find(r => r.model.includes("Ridge"));
  const treeRow = cvTable.find(r => r.model.includes("Decision tree"));

  // Find lowest RMSE model among linear models
  const linearModels = cvTable.filter(r => !r.model.includes("Baseline") && !r.model.includes("Decision tree"));
  linearModels.sort((a, b) => a.rmse - b.rmse);
  const bestLinear = linearModels[0];

  const synthesisLines = [];
  if (dropped.length > 0) {
    synthesisLines.push(`${dropped.length} column(s) were exactly redundant and pruned: [${dropped.join(", ")}].`);
  } else {
    synthesisLines.push("No exactly redundant columns detected (full column rank preserved).");
  }

  synthesisLines.push(`Data space condition number κ = ${cond.toFixed(1)} (${condClassification}); ${highVifCount} feature(s) exhibit VIF > 5.`);

  if (bestLinear.model.includes("Ridge") && ridgeRow.rmse < olsRow.rmse - 0.01 * olsRow.rmse) {
    synthesisLines.push(`Lowest cross-validated RMSE: ${bestLinear.model} (RMSE ${bestLinear.rmse.toFixed(2)} vs OLS ${olsRow.rmse.toFixed(2)}); ridge shrinkage successfully dampens multicollinear variance.`);
  } else {
    synthesisLines.push(`Lowest cross-validated RMSE among linear models: ${bestLinear.model} (RMSE ${bestLinear.rmse.toFixed(2)}); standard OLS remains competitive.`);
  }

  if (pcrRuleRow && olsRow && pcrRuleRow.rmse > 1.03 * olsRow.rmse) {
    const pctWorse = ((pcrRuleRow.rmse / olsRow.rmse - 1) * 100).toFixed(0);
    synthesisLines.push(`PCA (${Math.round(varTarget * 100)}% rule) performed ${pctWorse}% worse than OLS on holdouts: unsupervised variance selection inadvertently discarded predictive directions.`);
  }

  if (treeRow && olsRow) {
    if (treeRow.rmse < 0.95 * olsRow.rmse) {
      synthesisLines.push("Non-linear decision tree decisively outperforms linear models, demonstrating strong underlying non-linear interactions.");
    } else {
      synthesisLines.push("Non-linear decision tree provides no gain over linear models, confirming that a linear model structure is well-suited for this dataset.");
    }
  }

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
    cond,
    condClassification,
    condB,
    condBtB,
    vifMap,
    highVifCount,
    pivotColNames,
    freeColNames,
    freeColFormulas,
    weights: Object.fromEntries(basis.map((name, i) => [name, x[i]])),
    weightsTable: basis.map((name, i) => ({
      feature: name,
      weight: x[i],
      weight_lu: x_lu[i],
      std_err: stdErrors[i],
      ci_low: ciLow[i],
      ci_high: ciHigh[i],
      vif: vifMap[name] || 1.0
    })),
    eigenvalues: Array.from(eigVals),
    cumulativeVariance: cumsumEig,
    varianceRetained: cumsumEig[k_pca - 1],
    k_pca,
    k_cv,
    k_curve,
    lam_cv,
    lam_curve,
    dofRidge,
    pc1Loadings,
    pc2Loadings,
    compressionTable,
    train_rmse: tr_rmse,
    test_rmse: te_rmse,
    pcr90_test_rmse: pcr90_te_rmse,
    ridge_test_rmse: ridge_te_rmse,
    baseline_rmse: base_rmse,
    normB,
    normProj,
    normResid,
    highLevRows,
    highLevCutoff,
    sortedCooks,
    cvTable,
    synthesisLines,
    checks,
    notes,
    computationMs,
    // Previews for UI display
    previewA,
    previewB,
    previewBlock: blk,
    rrefBlock: rref_full.rref.slice(0, r_).map(r => r.slice(0, c_)),
    luP,
    luL,
    luU,
    previewQ: Array.from({ length: Math.min(5, n_tr) }, (_, i) =>
      Array.from({ length: Math.min(5, rank) }, (_, j) => Number(Q[i][j].toFixed(3)))
    ),
    previewR: Array.from({ length: Math.min(5, rank) }, (_, i) =>
      Array.from({ length: Math.min(5, rank) }, (_, j) => Number(R[i][j].toFixed(3)))
    ),
    actual_test: Array.from(bt_te).map(v => v + bm),
    pred_test: Array.from(pred_te).map(v => v + bm),
    pred_ridge_test: Array.from(pred_ridge_te).map(v => v + bm),
    pred_pcr_test: Array.from(pred_pcr90_te).map(v => v + bm),
    leverage_vals: Array.from(lev)
  };
}

// 17. Scope Export & Worker Message Listener
if (typeof window !== 'undefined') {
  window.runLinearAlgebraPipeline = runLinearAlgebraPipeline;
}

if (typeof self !== 'undefined') {
  self.onmessage = function(e) {
    const { type, payload } = e.data || {};
    if (type === 'RUN_PIPELINE' && payload) {
      try {
        const results = runLinearAlgebraPipeline(payload.rows, payload.targetCol, payload.options);
        self.postMessage({ type: 'PIPELINE_COMPLETE', payload: results });
      } catch (err) {
        self.postMessage({ type: 'PIPELINE_ERROR', payload: { message: err.message, stack: err.stack } });
      }
    }
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runLinearAlgebraPipeline };
}
