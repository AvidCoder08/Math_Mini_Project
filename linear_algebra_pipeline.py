# %% [markdown]
# # Linear Algebra Automated Data Analysis Tool (v2)
# 
# Takes **any table with a numeric target** and runs the 10-stage linear algebra workflow:
# 
# Matrix -> RREF/LU -> Rank/SVD -> Basis -> Gram-Schmidt (QR) -> Projection -> Least Squares -> Eigen -> PCA / PCR / Ridge -> Cross-validated Final Model
# 
# **What changed from v1:** the stages now *feed each other*.
# 
# - The **QR from Stage 5 is the solver** for least squares in Stage 7 (Rx = Q^T b), cross-checked against LU on the normal equations, `lstsq` and scikit-learn.
# - The **hat matrix H = QQ^T** from Stage 6 gives leverage and Cook's distance (outlier / influence detection).
# - **SVD, condition number and VIF** extend "rank" to *numerical* rank, so near-collinear data is detected, not just exact duplicates.
# - Stage 9 compares **PCA-regression (90% rule), PCR with k chosen by cross-validation, and Ridge via SVD**.
# - Stage 10 uses **k-fold cross-validation** (scaling fitted inside each fold) with RMSE, MAE, R^2 and fold spread, plus a random forest as an honest non-linear baseline.
# 
# Every stage prints its result, a **Concept -> Purpose -> Outcome** caption, and an automatic correctness check.
# 
# **How to use:** Run all cells (Runtime -> Run all). Change the dataset/target in the *Run it* section.

# %% [markdown]
# ## 1. Setup

# %%
import io, contextlib, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lu, qr, lu_factor, lu_solve, solve_triangular
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 2. Preprocessing and helpers

# %%
def preprocess(df, target):
    b_num = pd.to_numeric(df[target], errors="coerce")
    df = df.loc[b_num.notna()].copy()
    b = b_num.dropna().values.astype(float)
    X = df.drop(columns=[target])
    notes = []
    for c in list(X.columns):                       # drop constant / ID-like columns
        if X[c].nunique() <= 1:
            X = X.drop(columns=c); notes.append(f"dropped constant column '{c}'")
        elif str(c).lower() in ("id", "index") or (X[c].dtype.kind in "iu" and X[c].nunique() == len(X)):
            X = X.drop(columns=c); notes.append(f"dropped ID-like column '{c}'")
    cat = X.select_dtypes(exclude="number").columns.tolist()
    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=True, dtype=float)
        notes.append(f"one-hot encoded text columns {cat}")
    n_na = int(X.isna().sum().sum())
    if n_na:
        X = X.fillna(X.median()); notes.append(f"filled {n_na} missing values with column medians")
    X = X.astype(float)
    for c in X.columns:                              # target-leakage warning
        if X[c].std() > 0 and b.std() > 0:
            r = np.corrcoef(X[c].values, b)[0, 1]
            if abs(r) > 0.98:
                notes.append(f"WARNING: '{c}' has correlation {r:.3f} with the target (possible leakage)")
    return X, b, notes


def explain(concept, purpose, outcome):
    print(f"   Concept -> {concept}")
    print(f"   Purpose -> {purpose}")
    print(f"   Outcome -> {outcome}\n")


def rref(M, tol=1e-9):
    """Reduced row echelon form by Gauss-Jordan elimination with partial pivoting."""
    M = M.astype(float).copy(); r = 0
    for c in range(M.shape[1]):
        if r >= M.shape[0]: break
        p = r + int(np.argmax(np.abs(M[r:, c])))
        if abs(M[p, c]) < tol: continue
        M[[r, p]] = M[[p, r]]; M[r] /= M[r, c]
        for i in range(M.shape[0]):
            if i != r: M[i] -= M[i, c] * M[r]
        r += 1
    return M


def rmse(t, p): return float(np.sqrt(np.mean((t - p) ** 2)))
def mae(t, p):  return float(np.mean(np.abs(t - p)))
def r2(t, p):
    ss = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else float("nan")

# %% [markdown]
# ## 3. Linear algebra toolbox (each solver is built from the stages above it)

# %%
def gram_schmidt(M):
    """Modified Gram-Schmidt. Returns Q (orthonormal columns) and R (upper triangular) with M = QR."""
    n, d = M.shape
    Q, R, V = np.zeros((n, d)), np.zeros((d, d)), M.astype(float).copy()
    for j in range(d):
        for i in range(j):
            R[i, j] = Q[:, i] @ V[:, j]
            V[:, j] -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
    return Q, R


def numerical_rank(s, shape):
    return int((s > s[0] * max(shape) * np.finfo(float).eps).sum())


def basis_cols(A):
    """Indices of a maximal independent set of columns (pivoted QR, rank from SVD)."""
    r = numerical_rank(np.linalg.svd(A, compute_uv=False), A.shape)
    _, _, piv = qr(A, pivoting=True)
    return sorted(piv[:r])


def solve_qr(B, y):                       # Stage 5 -> Stage 7: R x = Q^T y
    Q, R = gram_schmidt(B)
    return solve_triangular(R, Q.T @ y)

def solve_lu_normal(B, y):                # LU on the normal equations (B^T B) x = B^T y
    return lu_solve(lu_factor(B.T @ B), B.T @ y)

def solve_ridge_svd(B, y, lam):           # shrink along each singular direction
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ y))

def solve_pcr(B, y, k):                   # keep only the top-k singular directions
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Vt[:k].T @ ((U[:, :k].T @ y) / s[:k])


LAMBDAS = np.logspace(-2, 4, 30)

def cv_select(B, y, kind, grid, folds=5, seed=0):
    """Pick k (kind='pcr') or lambda (kind='ridge') by inner k-fold CV. Returns (best, mean RMSE curve)."""
    err = np.zeros((folds, len(grid)))
    for f, (a, v) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(B)):
        mu = B[a].mean(0)
        Ba, Bv = B[a] - mu, B[v] - mu
        ym = y[a].mean()
        U, s, Vt = np.linalg.svd(Ba, full_matrices=False)
        Uty = U.T @ (y[a] - ym)
        for g, p in enumerate(grid):
            if kind == "ridge":
                x = Vt.T @ ((s / (s ** 2 + p)) * Uty)
            else:
                k = int(p); x = Vt[:k].T @ (Uty[:k] / s[:k])
            err[f, g] = rmse(y[v] - ym, Bv @ x)
    curve = err.mean(0)
    return grid[int(np.argmin(curve))], curve

# %% [markdown]
# ## 4. Cross-validated model comparison (used by Stage 10)

# %%
def fit_models(X_tr, b_tr, X_te, var_target=0.90):
    """Fit every model on one training fold. Scaling, basis selection and hyper-parameters use training data only."""
    mu, sd = X_tr.mean(0), X_tr.std(0); sd[sd == 0] = 1
    A, A_te = (X_tr - mu) / sd, (X_te - mu) / sd
    bm = b_tr.mean(); bt = b_tr - bm
    keep = basis_cols(A)
    B, B_te = A[:, keep], A_te[:, keep]
    r = B.shape[1]
    out = {"Baseline (predict mean)": np.full(len(B_te), bm)}

    x_qr = solve_qr(B, bt)
    out["OLS (QR least squares)"] = B_te @ x_qr + bm

    s = np.linalg.svd(B, compute_uv=False)
    k90 = int(min(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), var_target) + 1, r))
    out[f"PCR ({int(var_target*100)}% variance rule)"] = B_te @ solve_pcr(B, bt, k90) + bm

    k_cv, _ = cv_select(B, bt, "pcr", np.arange(1, r + 1))
    out["PCR (k by cross-val)"] = B_te @ solve_pcr(B, bt, k_cv) + bm

    lam, _ = cv_select(B, bt, "ridge", LAMBDAS)
    out["Ridge via SVD (lambda by cross-val)"] = B_te @ solve_ridge_svd(B, bt, lam) + bm

    rf = RandomForestRegressor(50, random_state=0, n_jobs=1).fit(B, bt)
    out["Random forest (non-linear)"] = rf.predict(B_te) + bm

    sk = LinearRegression(fit_intercept=False).fit(B, bt).coef_
    gap = float(np.max(np.abs(x_qr - sk)) / max(1.0, np.max(np.abs(sk))))
    return out, dict(k90=k90, k_cv=k_cv, lam=lam, sk_gap=gap)


def cross_validate(Xv, b, folds=5, seed=0, var_target=0.90):
    scores, hyper = {}, []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Xv):
        preds, h = fit_models(Xv[tr], b[tr], Xv[te], var_target)
        hyper.append(h)
        for name, p in preds.items():
            scores.setdefault(name, []).append((rmse(b[te], p), mae(b[te], p), r2(b[te], p)))
    rows = []
    for name, v in scores.items():
        v = np.array(v)
        rows.append(dict(model=name, RMSE=v[:, 0].mean(), RMSE_sd=v[:, 0].std(),
                         MAE=v[:, 1].mean(), R2=v[:, 2].mean(), R2_sd=v[:, 2].std()))
    return pd.DataFrame(rows), pd.DataFrame(hyper)

# %% [markdown]
# ## 5. The full pipeline

# %%
def run_pipeline(df, target, add_redundant=False, var_target=0.90,
                 test_frac=0.2, folds=5, seed=0, plots=True):
    X, b, notes = preprocess(df, target)
    if add_redundant and X.shape[1] >= 2:           # optional: exact duplicate column for demo
        c1, c2 = X.columns[:2]
        X[f"{c1}+{c2}"] = X[c1] + X[c2]
        notes.append(f"added redundant column '{c1}+{c2}' for demo")
    names = list(X.columns)
    print("DATA PREPARATION")
    for m in notes: print("  -", m)
    if not notes: print("  - no changes needed")
    print()

    # one shuffled train/test split for the stage-by-stage walkthrough (statistics from train only)
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int((1 - test_frac) * len(X))
    tr, te = idx[:cut], idx[cut:]
    Xv = X.values
    mu, sd = Xv[tr].mean(0), Xv[tr].std(0); sd[sd == 0] = 1
    A, A_te = (Xv[tr] - mu) / sd, (Xv[te] - mu) / sd
    bm = b[tr].mean()
    bt, bt_te = b[tr] - bm, b[te] - bm
    n, d = A.shape
    checks = {}

    # ---- STEP 1 ----
    print("STEP 1: MATRIX REPRESENTATION")
    print(f"   A is {n} x {d}  (rows = samples, columns = features); b has {n} target values")
    print("   First 3 rows of A:\n", A[:3, :min(d, 6)])
    explain("Data as matrix A and vector b",
            "Linear algebra only works on matrices; the model is Ax ~ b",
            f"{n} samples x {d} features ready for analysis")

    # ---- STEP 2 ----
    print("STEP 2: MATRIX SIMPLIFICATION (RREF + LU)")
    R_full = rref(A)                                  # Gauss-Jordan on the FULL training matrix
    nz = [i for i in range(R_full.shape[0]) if np.any(np.abs(R_full[i]) > 1e-9)]
    pivots = [int(np.flatnonzero(np.abs(R_full[i]) > 1e-9)[0]) for i in nz]
    free = [j for j in range(d) if j not in pivots]
    print(f"   RREF of the full {n} x {d} matrix: {len(pivots)} pivot columns, {len(free)} free columns")
    print("   Pivot columns:", [names[p] for p in pivots])
    if free:
        for j in free:
            combo = {names[p]: round(float(R_full[i, j]), 3) for i, p in zip(nz, pivots) if abs(R_full[i, j]) > 1e-9}
            print(f"   Free column '{names[j]}' = {combo}   (exact linear combination of pivot columns)")
    else:
        print("   No free columns -> every column is a pivot -> full column rank")
    r_, c_ = min(6, n), min(6, d)
    blk = A[:r_, :c_]
    P, L, U_ = lu(blk)
    checks["LU reconstructs block (P@L@U = A)"] = np.allclose(P @ L @ U_, blk)
    explain("Gauss-Jordan elimination (RREF) on the full matrix, plus LU factorisation of a block",
            "Pivot columns are the independent features; free columns are exact combinations of them",
            f"{len(pivots)} pivots, {len(free)} free column(s); LU of a {r_}x{c_} block rebuilds exactly (P@L@U = A)")

    # ---- STEP 3 ----
    print("STEP 3: STRUCTURE OF THE SPACE (RANK, NULLITY, SVD, CONDITION NUMBER)")
    s_all = np.linalg.svd(A, compute_uv=False)
    rank = numerical_rank(s_all, A.shape)
    cond = float(s_all[0] / s_all[rank - 1])
    checks["SVD rank = np.linalg.matrix_rank"] = rank == int(np.linalg.matrix_rank(A))
    checks["RREF pivot count = SVD rank"] = len(pivots) == rank
    level = "well-conditioned" if cond < 30 else ("moderately collinear" if cond < 1000 else "severely collinear")
    print(f"   numerical rank = {rank}, nullity = {d - rank}, columns = {d}")
    print("   Singular values:", s_all)
    print(f"   Condition number (on the independent part) = {cond:.1f}  -> {level}")
    explain("Rank, nullity, singular values, condition number",
            "Count independent columns AND measure how close to dependent the rest are",
            f"{rank} of {d} columns independent; condition number {cond:.0f} ({level})")

    # ---- STEP 4 ----
    print("STEP 4: REMOVE REDUNDANCY (BASIS) AND FLAG NEAR-DEPENDENCE (VIF)")
    keep = basis_cols(A)
    basis = [names[i] for i in keep]
    dropped = [nm for nm in names if nm not in basis]
    B, B_te = A[:, keep], A_te[:, keep]
    r = len(keep)
    corr = np.corrcoef(B.T) if r > 1 else np.array([[1.0]])
    vif = pd.Series(np.diag(np.linalg.pinv(corr)), index=basis)
    high = vif[vif > 5].sort_values(ascending=False)
    print("   Dropped (exactly dependent):", dropped if dropped else "none (already full rank)")
    print("   Basis columns :", basis)
    print("   Variance inflation factors > 5:", high.round(1).to_dict() if len(high) else "none")
    explain("Basis via pivoted QR; variance inflation factor = diag of inverse correlation matrix",
            "Drop exact duplicates; flag columns that are almost combinations of others",
            f"Basis of {r} columns; {len(high)} column(s) with VIF > 5 (ridge in Stage 9 handles these)")

    # ---- STEP 5 ----
    print("STEP 5: ORTHOGONALIZATION (GRAM-SCHMIDT -> QR)")
    Q, R = gram_schmidt(B)
    checks["Q^T Q = I (orthonormal)"] = np.allclose(Q.T @ Q, np.eye(r), atol=1e-6)
    checks["Q R = B (factorisation)"] = np.allclose(Q @ R, B, atol=1e-8 * max(1, np.abs(B).max()))
    print(f"   Q is {Q.shape}, R is {R.shape} (upper triangular); max |Q^T Q - I| = {np.abs(Q.T @ Q - np.eye(r)).max():.2e}")
    explain("Gram-Schmidt gives B = QR with orthonormal Q",
            "Perpendicular columns make projection stable, and R makes least squares a triangular solve",
            "Q will be reused for the hat matrix (Stage 6) and the solver (Stage 7)")

    # ---- STEP 6 ----
    print("STEP 6: PROJECTION (HAT MATRIX, LEVERAGE, INFLUENCE)")
    proj = Q @ (Q.T @ bt)                            # H b with H = Q Q^T
    resid = bt - proj
    lev = (Q ** 2).sum(1)                            # diag(H) without forming the n x n matrix
    s2 = float(resid @ resid) / max(1, n - r)
    cook = resid ** 2 / (r * s2) * lev / (np.clip(1 - lev, 1e-12, None) ** 2)
    flag = np.where(lev > 2 * r / n)[0]
    checks["Residual perpendicular to columns"] = np.allclose(B.T @ resid, 0, atol=1e-5 * max(1, abs(bt).max()))
    checks["Projection is idempotent (H(Hb) = Hb)"] = np.allclose(Q @ (Q.T @ proj), proj, atol=1e-6 * max(1, abs(bt).max()))
    checks["trace(H) = rank"] = bool(np.isclose(lev.sum(), r))
    top = np.argsort(cook)[::-1][:3]
    print(f"   |b| = {np.linalg.norm(bt):.2f}, |projection| = {np.linalg.norm(proj):.2f}, |residual| = {np.linalg.norm(resid):.2f}")
    print(f"   trace(H) = {lev.sum():.2f} (= rank {r}); {len(flag)} high-leverage rows (h > 2p/n = {2*r/n:.3f})")
    print("   Most influential rows (Cook's distance):", {int(i): round(float(cook[i]), 3) for i in top})
    explain("Orthogonal projection b -> H b with hat matrix H = Q Q^T; diag(H) = leverage",
            "Find the closest reachable point to b, and see which samples pull the fit the most",
            f"Residual norm {np.linalg.norm(resid):.2f}; {len(flag)} high-leverage samples flagged")

    # ---- STEP 7 ----
    print("STEP 7: LEAST SQUARES (SOLVED WITH THE QR FROM STEP 5)")
    x = solve_qr(B, bt)                              # R x = Q^T b
    x_lu = solve_lu_normal(B, bt)                    # LU on normal equations
    x_ls = np.linalg.lstsq(B, bt, rcond=None)[0]
    x_sk = LinearRegression(fit_intercept=False).fit(B, bt).coef_
    scale = max(1.0, np.abs(x).max())
    gaps = {k_: float(np.abs(x - v).max() / scale) for k_, v in
            {"LU normal equations": x_lu, "np.linalg.lstsq": x_ls, "scikit-learn": x_sk}.items()}
    checks["QR solution = LU on normal equations"] = gaps["LU normal equations"] < 1e-4
    checks["QR solution = np.linalg.lstsq"] = gaps["np.linalg.lstsq"] < 1e-6
    checks["QR solution = scikit-learn LinearRegression"] = gaps["scikit-learn"] < 1e-6
    checks["Normal equations hold (B^T B x = B^T b)"] = np.allclose(B.T @ B @ x, B.T @ bt, rtol=1e-4, atol=1e-6 * max(1, abs(B.T @ bt).max()))
    Rinv = solve_triangular(R, np.eye(r))
    se = np.sqrt(s2 * np.diag(Rinv @ Rinv.T))        # sigma^2 (B^T B)^-1 = sigma^2 R^-1 R^-T
    wt = pd.DataFrame({"weight": x, "std_err": se, "ci_low": x - 1.96 * se, "ci_high": x + 1.96 * se}, index=basis)
    pred_ols = B_te @ x
    tr_rmse, te_rmse, base = rmse(bt, B @ x), rmse(bt_te, pred_ols), rmse(bt_te, np.zeros_like(bt_te))
    sv = np.linalg.svd(B, compute_uv=False)
    print("   Weights with standard errors and 95% intervals:"); print(wt.round(3).to_string())
    print(f"   cond(B) = {sv[0]/sv[-1]:.1f}, cond(B^T B) = {(sv[0]/sv[-1])**2:.1f}  (normal equations square the condition number)")
    print("   Max coefficient gap vs QR:", {k_: f"{v:.1e}" for k_, v in gaps.items()})
    print(f"   Train RMSE = {tr_rmse:.3f} | Test RMSE = {te_rmse:.3f} | Baseline (predict mean) = {base:.3f}")
    explain("Least squares via QR: solve R x = Q^T b (back-substitution)",
            "Best-fit weights; QR is more stable than forming (A^T A)^-1; (B^T B)^-1 also gives standard errors",
            f"Four independent solvers agree; test RMSE {te_rmse:.2f} vs baseline {base:.2f}")

    # ---- STEP 8 ----
    print("STEP 8: EIGENVALUES & EIGENVECTORS (PATTERN DISCOVERY)")
    C = np.cov(B.T) if r > 1 else np.array([[np.var(B, ddof=1)]])
    vals, vecs = np.linalg.eigh(C)
    o = np.argsort(vals)[::-1]; vals, vecs = vals[o], vecs[:, o]
    sv_B = np.linalg.svd(B, compute_uv=False)
    checks["C v = lambda v for all pairs"] = np.allclose(C @ vecs, vecs * vals, atol=1e-6)
    checks["Sum of eigenvalues = trace"] = bool(np.isclose(vals.sum(), np.trace(C)))
    checks["Eigenvalues = (singular values)^2 / (n-1)"] = np.allclose(vals, sv_B ** 2 / (n - 1), atol=1e-8)
    checks["C = V diag(lambda) V^T (spectral theorem)"] = np.allclose(vecs @ np.diag(vals) @ vecs.T, C, atol=1e-8)
    checks["V^T V = I (eigenvectors orthonormal)"] = np.allclose(vecs.T @ vecs, np.eye(r), atol=1e-8)
    ratio = np.cumsum(vals) / vals.sum()
    print("   Eigenvalues:", vals)
    print("   Diagonalization: C = V diag(lambda) V^T, V orthogonal because C is symmetric")
    for k_ in range(min(2, r)):
        top_ = np.argsort(-np.abs(vecs[:, k_]))[:3]
        print(f"   PC{k_+1} loads mainly on:", {basis[i]: round(float(vecs[i, k_]), 2) for i in top_})
    print(f"   Largest direction explains {vals[0]/vals.sum()*100:.1f}% of variance")
    explain("Eigen-decomposition of the covariance matrix (equals SVD of the data: lambda = sigma^2/(n-1))",
            "Find the main directions of variation; big eigenvalue = important direction",
            f"Top eigenvector alone explains {vals[0]/vals.sum()*100:.1f}% of variance")

    # ---- STEP 9 ----
    print("STEP 9: PCA vs PCR vs RIDGE (SYSTEM SIMPLIFICATION)")
    k90 = int(min(np.searchsorted(ratio, var_target) + 1, len(vals)))
    k_cv, k_curve = cv_select(B, bt, "pcr", np.arange(1, r + 1))
    lam_cv, lam_curve = cv_select(B, bt, "ridge", LAMBDAS)
    x_pcr90, x_pcrcv, x_ridge = solve_pcr(B, bt, k90), solve_pcr(B, bt, k_cv), solve_ridge_svd(B, bt, lam_cv)
    x_sk_ridge = Ridge(alpha=lam_cv, fit_intercept=False).fit(B, bt).coef_
    checks["PCR with all components = OLS"] = np.allclose(solve_pcr(B, bt, r), x, rtol=1e-3, atol=1e-3 * scale)
    checks["Ridge via SVD = scikit-learn Ridge"] = np.allclose(x_ridge, x_sk_ridge, rtol=1e-4, atol=1e-6 * scale)
    dof = float(np.sum(sv_B ** 2 / (sv_B ** 2 + lam_cv)))
    rm = {"PCR 90% rule": rmse(bt_te, B_te @ x_pcr90), "PCR cross-val k": rmse(bt_te, B_te @ x_pcrcv),
          "Ridge (SVD)": rmse(bt_te, B_te @ x_ridge)}
    print(f"   Variance rule keeps k = {k90} of {r} components ({ratio[k90-1]*100:.1f}% variance)")
    print(f"   Cross-validation picks k = {k_cv} for PCR and lambda = {lam_cv:.3g} for ridge (effective dof {dof:.1f} of {r})")
    print("   Test RMSE on this split:", {k_: round(v, 3) for k_, v in rm.items()}, "| OLS:", round(te_rmse, 3))
    U_c, s_c, Vt_c = np.linalg.svd(B, full_matrices=False)
    print("   Compression (best rank-k approximation of B, Eckart-Young):")
    for k_ in sorted({1, max(1, r // 2), k90, r}):
        Bk = (U_c[:, :k_] * s_c[:k_]) @ Vt_c[:k_]
        stored = k_ * (n + r + 1)
        print(f"     rank {k_:>2}: relative error {np.linalg.norm(B - Bk) / np.linalg.norm(B):.3f}, "
              f"stores {stored} numbers vs {n * r} ({100 * stored / (n * r):.0f}%)")
    explain("PCA keeps top variance directions (ignores target); PCR chooses k by validation; ridge shrinks every singular direction smoothly",
            "Reduce noise without throwing away low-variance directions that still predict the target",
            f"k chosen by CV = {k_cv} (variance rule said {k90}); ridge lambda = {lam_cv:.3g}")

    # ---- STEP 10 ----
    print(f"STEP 10: FINAL APPLICATION OUTPUT - prediction of '{target}' ({folds}-fold cross-validation, all models refit inside every fold)")
    table, hyper = cross_validate(Xv, b, folds=folds, seed=seed, var_target=var_target)
    checks["Own QR solution = scikit-learn in every CV fold"] = bool(hyper["sk_gap"].max() < 1e-6)
    print(table.round(3).to_string(index=False))
    print(f"   Chosen per fold: PCR k = {hyper['k_cv'].tolist()}, ridge lambda = {[round(v, 2) for v in hyper['lam']]}\n")
    lin = table[~table.model.str.startswith(("Baseline", "Random"))]
    top = lin.loc[lin["RMSE"].idxmin()]
    ols_row = lin[lin.model.str.startswith("OLS")].iloc[0]
    tied = lin[lin["RMSE"] <= top["RMSE"] + top["RMSE_sd"]]            # within one fold-sd of the leader
    best = top["model"] if len(tied) == 1 else f"{top['model']} (tied within one fold-sd with {len(tied)-1} other linear model(s))"
    ols_r = float(table.loc[table.model == "OLS (QR least squares)", "RMSE"].iloc[0])
    pcr_var = table[table.model.str.contains("variance rule")]
    p90_r = float(pcr_var["RMSE"].iloc[0]) if len(pcr_var) else float(table.iloc[2]["RMSE"])
    rf_match = table[table.model.str.startswith("Random")]
    rf_r = float(rf_match["RMSE"].iloc[0]) if len(rf_match) else ols_r
    if top["RMSE"] >= ols_row["RMSE"] - ols_row["RMSE_sd"]:
        simple = "plain OLS is good enough" if ols_row["RMSE"] <= top["RMSE"] + top["RMSE_sd"] else "regularisation is worth it"
    else: simple = "regularisation is worth it"
    lines = [f"{len(dropped)} column(s) were exactly redundant and removed: {dropped}." if dropped else "No exactly redundant columns.",
             f"Condition number {cond:.0f} ({level}); {len(high)} column(s) with VIF > 5.",
             f"Lowest cross-validated RMSE among linear models: {best}; {simple}."]
    if p90_r > 1.05 * ols_r: lines.append(f"PCA ({int(var_target*100)}% rule) was {100*(p90_r/ols_r-1):.0f}% worse than OLS: it ignores the target. Cross-validated k or ridge is safer.")
    if rf_r < 0.95 * ols_r: lines.append("The random forest beats every linear model, so the relationship is not purely linear.")
    else: lines.append("No non-linear gain from a random forest, so a linear model is adequate here.")
    print("PLAIN-LANGUAGE SUMMARY")
    for l_ in lines: print("  -", l_)
    print()
    explain("Cross-validated comparison of OLS, PCR, ridge and a non-linear baseline",
            "Judge models on data they never saw, with the spread across folds",
            f"Winner: {best}")

    # ---- CHECKER ----
    print("CORRECTNESS CHECKER")
    chk = pd.DataFrame({"Check": list(checks), "Result": ["PASS" if v else "FAIL" for v in checks.values()]})
    print(chk.to_string(index=False)); print()

    # ---- PLOTS ----
    if plots:
        fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
        ax = ax.ravel()
        a_, p_ = bt_te + bm, B_te @ x_ridge + bm
        ax[0].scatter(a_, p_, alpha=.6); lo, hi = a_.min(), a_.max()
        ax[0].plot([lo, hi], [lo, hi], "r--"); ax[0].set_title("Ridge: predicted vs actual (test)")
        ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
        ax[1].semilogy(range(1, len(s_all) + 1), s_all, "o-"); ax[1].set_title(f"Singular values (condition {cond:.0f})")
        ax[1].set_xlabel("Index")
        ax[2].stem(lev, markerfmt=" "); ax[2].axhline(2 * r / n, color="r", ls="--")
        ax[2].set_title("Leverage (diag of hat matrix)"); ax[2].set_xlabel("Training row")
        ax[3].plot(range(1, r + 1), k_curve, "o-"); ax[3].axvline(k_cv, color="g", ls="--", label=f"CV k={k_cv}")
        ax[3].axvline(k90, color="r", ls=":", label=f"90% rule k={k90}"); ax[3].legend()
        ax[3].set_title("PCR: validation RMSE vs components"); ax[3].set_xlabel("Components kept")
        ax[4].semilogx(LAMBDAS, lam_curve, "o-"); ax[4].axvline(lam_cv, color="g", ls="--")
        ax[4].set_title("Ridge: validation RMSE vs lambda"); ax[4].set_xlabel("lambda")
        ax[5].barh(table["model"], table["RMSE"], xerr=table["RMSE_sd"]); ax[5].invert_yaxis()
        ax[5].set_title(f"{folds}-fold CV RMSE (mean +/- sd)")
        plt.tight_layout(); plt.show()

    return dict(rank=rank, nullity=d - rank, cond=cond, dropped=dropped, weights=dict(zip(basis, x)),
                high_vif=high.to_dict(), eigenvalues=vals, k90=k90, k_cv=k_cv, lam=lam_cv,
                table=table, best=best, checks=checks)

# %% [markdown]
# ## 6. Run it
# 
# **Option A:** built-in datasets (pick one by changing `choice`). The diabetes data is *naturally* collinear (`s1` and `s2` are strongly correlated), so no artificial duplicate is needed.
# **Option B:** upload your own CSV (set `USE_UPLOAD = True` and the `target` name).

# %% [markdown]
# ### What the data matrix represents (edit this if you change the dataset)
# 
# For the default **diabetes** dataset:
# - **Rows (samples):** one row per patient (442 patients)
# - **Columns (features):** 10 baseline measurements per patient: age, sex, BMI, blood pressure (bp) and six blood-serum measurements (s1-s6)
# - **Vector b (target):** a quantitative measure of disease progression one year after baseline
# - **Application output:** predict progression from baseline measurements, and see which directions of the data carry the signal
# 
# Note: scikit-learn ships this dataset already mean-centred and scaled.
# 

# %%
def get_dataset(choice):
    if choice == "diabetes":
        return load_diabetes(as_frame=True).frame, "target"
    if choice == "wine":
        return load_wine(as_frame=True).frame.drop(columns="target"), "proline"
    if choice == "breast_cancer":
        return load_breast_cancer(as_frame=True).frame.drop(columns="target"), "mean area"
    if choice == "california":                       # needs internet (fine on Colab)
        from sklearn.datasets import fetch_california_housing
        df = fetch_california_housing(as_frame=True).frame.sample(2000, random_state=0)
        return df, "MedHouseVal"
    raise ValueError("unknown dataset")

if __name__ == "__main__":
    # ---------- SETTINGS ----------
    USE_UPLOAD = False            # True -> upload your own CSV
    choice = "diabetes"           # diabetes | wine | breast_cancer | california
    target = None                 # only needed for uploaded CSV (column name to predict)
    CSV_PATH = "your_data.csv"    # used only when NOT on Colab and USE_UPLOAD = True
    ADD_REDUNDANT = False         # optional: add an exact duplicate column to demo rank/basis
    # ------------------------------

    if USE_UPLOAD:
        try:
            from google.colab import files
            up = files.upload()
            df = pd.read_csv(next(iter(up)))
        except ImportError:                                  # running locally / Jupyter
            df = pd.read_csv(CSV_PATH)
        print("Columns:", list(df.columns))
        assert target in df.columns, "Set `target` to one of the column names above"
    else:
        df, target = get_dataset(choice)
        print(f"Dataset: {choice} | target: {target} | shape: {df.shape}")

    results = run_pipeline(df, target, add_redundant=ADD_REDUNDANT)

    # Generalisation test: same code, many datasets
    print("\n" + "=" * 60)
    print("GENERALISATION TEST: SAME CODE, MANY DATASETS")
    print("=" * 60)
    rows = []
    for name in ["diabetes", "wine", "breast_cancer"]:
        d_, t_ = get_dataset(name)
        with contextlib.redirect_stdout(io.StringIO()):          # silence step output
            r_ = run_pipeline(d_, t_, plots=False)
        tb = r_["table"].set_index("model")["RMSE"]
        pcr90_val = tb.loc[tb.index.str.contains("variance rule")].iloc[0] if any(tb.index.str.contains("variance rule")) else tb.iloc[2]
        rows.append(dict(dataset=name, rank=r_["rank"], cond=round(r_["cond"]), dropped=r_["dropped"],
                         k_vs_90rule=f"{r_['k_cv']} vs {r_['k90']}", OLS=round(tb.iloc[1], 2),
                         PCR90=round(pcr90_val, 2), PCR_cv=round(tb.iloc[3], 2), Ridge=round(tb.iloc[4], 2),
                         RandomForest=round(tb.iloc[5], 2), Baseline=round(tb.iloc[0], 2),
                         best=r_["best"], all_checks_pass=all(r_["checks"].values())))
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

# %% [markdown]
# ## 8. Viva cheat sheet (Concept -> Purpose -> Outcome)
# 
# | Stage | Concept | Why we needed it | How it feeds the next stage |
# |---|---|---|---|
# | 1 | Matrix A, vector b | Linear algebra works on matrices | Input to everything |
# | 2 | RREF (full matrix) / LU | Pivot columns = independent features, free columns = exact dependencies | Pivot columns become the basis in Stage 4; LU reused on the normal equations |
# | 3 | Rank, SVD, condition number | Count independent info and measure near-dependence | Tells us whether least squares will be stable |
# | 4 | Basis (pivoted QR), VIF | Remove exact duplicates, flag near-duplicates | Gives the full-rank matrix B that QR needs |
# | 5 | Gram-Schmidt, B = QR | Perpendicular columns | Q gives the hat matrix, R gives the solver |
# | 6 | Projection, hat matrix H = QQ^T | Closest reachable point; leverage and Cook's distance | Residual is perpendicular to columns, which is the normal equations |
# | 7 | Least squares: R x = Q^T b | Best weights, with standard errors from (B^T B)^-1 | Cross-checked against LU, lstsq and scikit-learn |
# | 8 | Eigen-decomposition / diagonalization C = V L V^T | Main directions of variation (symmetric C, so V orthogonal) | Equals SVD: eigenvalue = sigma^2/(n-1) |
# | 9 | PCA / PCR / ridge (SVD), low-rank compression | Reduce noise and storage without ignoring the target | k and lambda chosen by cross-validation |
# | 10 | k-fold cross-validation | Honest comparison on unseen data | RMSE, MAE, R^2 with spread across folds |
# 
# **Key talking points**
# - Normal equations square the condition number (cond(B^T B) = cond(B)^2); QR avoids that.
# - `matrix_rank` only detects *exact* dependence; the condition number and VIF detect *near* dependence.
# - PCA ignores the target, so keeping 90% of variance can discard predictive directions. Ridge shrinks smoothly instead of cutting.
# - Ridge via SVD shrinks each singular direction by sigma^2/(sigma^2 + lambda).
# 
# **Limitations to state:** linear model only (the random forest shows when that is not enough); numeric target needed; basic preprocessing (no date handling, no outlier treatment); a dense SVD will be slow on very large tables.


