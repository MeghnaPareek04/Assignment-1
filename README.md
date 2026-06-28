# Assignment-1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import StringIO

plt.rcParams['figure.figsize'] = (9, 4)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
print('All imports OK ✓')

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Python Fundamentals
# ═══════════════════════════════════════════════════════════════════════════

# ── 1.1 Data Types & Control Flow ──────────────────────────────────────────
def classify_number(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n <= 10:
        return "small positive"
    else:
        return "large positive"

assert classify_number(-5) == "negative"
assert classify_number(0)  == "zero"
assert classify_number(7)  == "small positive"
assert classify_number(42) == "large positive"
print("1.1 passed ✓")


# ── 1.2 Data Structures ────────────────────────────────────────────────────
words = ['apple', 'banana', 'cherry', 'apple', 'date', 'banana', 'apple']

# 1. word_count dict (using only a loop)
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# 2. unique_words set
unique_words = set(words)

# 3. long_words list comprehension
long_words = [w for w in words if len(w) > 5]

assert word_count == {'apple': 3, 'banana': 2, 'cherry': 1, 'date': 1}
assert unique_words == {'apple', 'banana', 'cherry', 'date'}
assert set(long_words) == {'banana', 'cherry'}
print("1.2 passed ✓")


# ── 1.3 Exceptions ─────────────────────────────────────────────────────────
def safe_divide(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numeric")
    try:
        return a / b
    except ZeroDivisionError:
        return None

assert safe_divide(10, 2) == 5.0
assert safe_divide(5, 0) is None
try:
    safe_divide('x', 2)
    assert False, "Should have raised TypeError"
except TypeError as e:
    assert str(e) == "Inputs must be numeric"
print("1.3 passed ✓")


# ── 1.4 Functions & Lambdas ────────────────────────────────────────────────
def apply_twice(f, x):
    return f(f(x))

triple = lambda x: x * 3

result = apply_twice(triple, 4)
assert result == 36, f"Expected 36, got {result}"
print("1.4 passed ✓")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — NumPy
# ═══════════════════════════════════════════════════════════════════════════

# ── 2.1 Array Creation & Shapes ────────────────────────────────────────────
arr1d = np.arange(12)
arr2d = arr1d.reshape(3, 4)
arr3d = arr1d.reshape(2, 2, 3)

assert arr1d.shape == (12,)
assert arr2d.shape == (3, 4)
assert arr3d.shape == (2, 2, 3)
print("2.1 passed ✓")
print(f"arr2d:\n{arr2d}")
print(f"arr3d:\n{arr3d}")


# ── 2.2 Indexing & Slicing ─────────────────────────────────────────────────
row2 = arr2d[1]           # second row (0-indexed)
col3 = arr2d[:, 2]        # third column (0-indexed)
sub  = arr2d[1:, 2:]      # bottom-right 2×2
gt7  = arr2d[arr2d > 7]   # boolean indexing

assert list(row2) == [4, 5, 6, 7]
assert list(col3) == [2, 6, 10]
assert sub.shape == (2, 2)
assert list(gt7) == [8, 9, 10, 11]
print("2.2 passed ✓")


# ── 2.3 Operations & Dot Product ───────────────────────────────────────────
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C  = A * B                        # element-wise
D  = A @ B                        # matrix multiplication
dp = np.dot([1, 2, 3], [4, 5, 6]) # dot product
E  = A * 3                        # scalar multiplication

assert np.array_equal(C, [[5, 12], [21, 32]])
assert np.array_equal(D, [[19, 22], [43, 50]])
assert dp == 32
assert np.array_equal(E, [[3, 6], [9, 12]])
print("2.3 passed ✓")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Pandas
# ═══════════════════════════════════════════════════════════════════════════

CSV_DATA = """employee_id,name,department,salary,age,years_exp,performance
1,Alice,Engineering,95000,30,5,4.2
2,Bob,Marketing,72000,35,8,3.8
3,Carol,Engineering,88000,28,3,4.5
4,Dave,HR,61000,42,15,3.1
5,Eve,Engineering,102000,38,12,4.8
6,Frank,Marketing,68000,29,4,3.5
7,Grace,HR,,31,6,3.9
8,Hank,Engineering,91000,45,20,4.1
9,Iris,Marketing,75000,,7,4.0
10,Jack,Engineering,85000,33,8,
"""
df = pd.read_csv(StringIO(CSV_DATA))
print(f"Shape: {df.shape}")


# ── 3.1 DataFrames vs Series ───────────────────────────────────────────────
salary_series = df['salary']
name_dept_df  = df[['name', 'department']]

assert isinstance(salary_series, pd.Series)
assert isinstance(name_dept_df, pd.DataFrame)
print("3.1 passed ✓")


# ── 3.2 iloc & loc ─────────────────────────────────────────────────────────
iloc_result = df.iloc[2:5, 0:3]           # rows 2-4 inclusive, cols 0-2
loc_result  = df.loc[5:6, ['name', 'salary']]  # index labels 5,6; name & salary cols

assert iloc_result.shape == (3, 3)
assert list(loc_result.columns) == ['name', 'salary']
print("3.2 passed ✓")


# ── 3.3 Filtering & Group By ───────────────────────────────────────────────
senior_eng = df[(df['department'] == 'Engineering') & (df['salary'] > 90000)]

dept_stats = (
    df.groupby('department')
      .agg(mean_salary=('salary', 'mean'), mean_performance=('performance', 'mean'))
      .sort_values('mean_salary', ascending=False)
)

print(f"Senior engineers: {len(senior_eng)}")
print(senior_eng[['name', 'salary']])
print("\nDept stats:")
print(dept_stats)


# ── 3.4 Handling Missing Data ──────────────────────────────────────────────
print("Missing values before:")
print(df.isnull().sum())

df_filled = df.copy()

# Fill missing salary with median
median_salary = df_filled['salary'].median()
df_filled['salary'] = df_filled['salary'].fillna(median_salary)

# Fill missing age with rounded mean
mean_age = round(df_filled['age'].mean())
df_filled['age'] = df_filled['age'].fillna(mean_age)

# Drop rows where performance is missing
df_filled = df_filled.dropna(subset=['performance'])

print("\nMissing values after:")
print(df_filled.isnull().sum())
assert df_filled.isnull().sum().sum() == 0, "Still has nulls!"
print("3.4 passed ✓")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Linear Algebra
# ═══════════════════════════════════════════════════════════════════════════

# ── 4.1 Vectors & Matrices ─────────────────────────────────────────────────
v = np.array([3, 4])
norm_v = np.linalg.norm(v)

M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

fig, ax = plt.subplots()
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='steelblue')
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Vector Visualization')
ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='k', lw=0.8)
plt.tight_layout(); plt.show()

assert abs(norm_v - 5.0) < 1e-9
assert M.shape == (3, 3)
print(f"4.1 passed ✓  |v| = {norm_v}")


# ── 4.2 Matrix Operations ──────────────────────────────────────────────────
P = np.array([[2, 1], [0, 3]])
Q = np.array([[1, 4], [2, 0]])

PplusQ   = P + Q
scalar3P = 3 * P
PQ       = P @ Q
QP       = Q @ P

print(f"P + Q =\n{PplusQ}")
print(f"3*P =\n{scalar3P}")
print(f"P @ Q =\n{PQ}")
print(f"Q @ P =\n{QP}")
print(f"PQ == QP? {np.array_equal(PQ, QP)}")
assert not np.array_equal(PQ, QP)
print("4.2 passed ✓")


# ── 4.3 Eigenvalues & Eigenvectors ─────────────────────────────────────────
A_eigen = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A_eigen)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify Av = λv for each eigenpair
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    vec = eigenvectors[:, i]
    lhs = A_eigen @ vec
    rhs = lam * vec
    assert np.allclose(lhs, rhs), f"Eigenpair {i} failed verification"
    print(f"Eigenpair {i}: Av = {lhs}, λv = {rhs} ✓")

# Plot eigenvectors scaled by their eigenvalues
fig, ax = plt.subplots()
colors = ['steelblue', 'tomato']
for i in range(len(eigenvalues)):
    vec = eigenvectors[:, i] * eigenvalues[i]
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
              color=colors[i], label=f'λ={eigenvalues[i]:.1f}')
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_title('Eigenvectors scaled by Eigenvalues')
ax.legend(); plt.tight_layout(); plt.show()
print("4.3 passed ✓")

# Geometric explanation:
# An eigenvector of a matrix A is a special direction that does NOT rotate under
# the linear transformation — it only gets stretched or squished. The eigenvalue λ
# tells you the scale factor: if λ > 1 the vector is stretched, 0 < λ < 1 it is
# squished, λ < 0 it is flipped, and λ = 1 leaves it unchanged. All other vectors
# change direction when multiplied by A; eigenvectors are the invariant axes.


# ── 4.4 SVD & Dimensionality Reduction ────────────────────────────────────
np.random.seed(42)
X = np.random.randn(4, 3)

U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Reconstruct: X = U @ diag(S) @ Vt
X_reconstructed = U @ np.diag(S) @ Vt

# Rank-1 approximation: use only the first singular value/vector
X_approx = S[0] * np.outer(U[:, 0], Vt[0, :])

print(f"U shape: {U.shape}, S shape: {S.shape}, Vt shape: {Vt.shape}")
print(f"Singular values: {S}")
print(f"Reconstruction error: {np.linalg.norm(X - X_reconstructed):.2e}")
print(f"Rank-1 approximation:\n{X_approx}")

assert np.allclose(X, X_reconstructed, atol=1e-10)
print("4.4 passed ✓")

# SVD → PCA connection:
# In SVD, U contains the left singular vectors (directions in sample space),
# S contains the singular values (square roots of variance explained), and
# Vt contains the right singular vectors — these ARE the principal component
# directions (the rows of Vt / columns of V). PCA on a mean-centered matrix X
# is equivalent to its SVD: the principal components are the rows of Vt, and
# the variance explained by each component is S[i]² / (n-1). The matrix Vt
# holds the directions of maximum variance.


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Statistics
# ═══════════════════════════════════════════════════════════════════════════

# ── 5.1 Descriptive Statistics ─────────────────────────────────────────────
salary = df_filled['salary']

mean_s   = salary.mean()
median_s = salary.median()
std_s    = salary.std()
min_s    = salary.min()
max_s    = salary.max()
iqr_s    = np.percentile(salary, 75) - np.percentile(salary, 25)

print(f"Mean:   {mean_s:.0f}")
print(f"Median: {median_s:.0f}")
print(f"Std:    {std_s:.0f}")
print(f"Range:  {min_s:.0f} – {max_s:.0f}")
print(f"IQR:    {iqr_s:.0f}")

# Histogram + KDE overlay
fig, ax = plt.subplots()
ax.hist(salary, bins=6, density=True, alpha=0.5, color='steelblue', label='Histogram')
kde_x = np.linspace(salary.min() - 5000, salary.max() + 5000, 300)
kde = stats.gaussian_kde(salary)
ax.plot(kde_x, kde(kde_x), color='tomato', lw=2, label='KDE')
ax.set_xlabel('Salary'); ax.set_title('Salary Distribution')
ax.legend(); plt.tight_layout(); plt.show()

# Definitions:
# Population: The complete set of all individuals or observations of interest
#             (e.g. every employee at the company).
# Sample: A subset drawn from the population used to make inferences
#         (e.g. the 10 employees in our CSV).
# Descriptive statistic: A number that summarises or describes features of a
#                        dataset (e.g. mean, std, IQR).
# Inferential statistic: A technique that uses sample data to draw conclusions
#                        or make predictions about a broader population
#                        (e.g. t-test, confidence interval).


# ── 5.2 Hypothesis Testing ─────────────────────────────────────────────────
eng_salaries = df_filled[df_filled['department'] == 'Engineering']['salary']
overall_mean = df_filled['salary'].mean()

# H₀: Engineering mean salary = overall mean salary
# H₁: Engineering mean salary > overall mean salary (one-tailed)
t_stat, p_value = stats.ttest_1samp(eng_salaries, overall_mean)

print(f"Overall mean salary: {overall_mean:.0f}")
print(f"Engineering mean salary: {eng_salaries.mean():.0f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H0 at α=0.05? {p_value < 0.05}")

# Pearson correlation: salary vs years_exp
r, r_pval = stats.pearsonr(df_filled['salary'], df_filled['years_exp'])
print(f"\nPearson r (salary vs years_exp): {r:.4f}, p={r_pval:.4f}")

# H₀: The mean salary of Engineering employees equals the overall mean salary.
# H₁: The mean salary of Engineering employees is greater than the overall mean.
# Conclusion: If p < 0.05, we reject H₀ and conclude Engineering salaries are
# significantly higher than the company average.


# ── 5.3 Error Metrics ──────────────────────────────────────────────────────
y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5, 6.0, 1.5, 8.0])
y_pred = np.array([2.8, 5.2, 2.1, 7.5, 4.0, 6.3, 2.0, 7.8])
n, p = len(y_true), 2

mae    = np.mean(np.abs(y_true - y_pred))
mse    = np.mean((y_true - y_pred) ** 2)
rmse   = np.sqrt(mse)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2     = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE:        {mae:.4f}")
print(f"MSE:        {mse:.4f}")
print(f"RMSE:       {rmse:.4f}")
print(f"R²:         {r2:.4f}")
print(f"Adj. R²:    {adj_r2:.4f}")


# ── 5.4 Distribution Testing & Stationarity ────────────────────────────────
from statsmodels.tsa.stattools import adfuller

np.random.seed(0)
s1 = np.random.normal(0, 1, 200)
s2 = np.random.exponential(1, 200)

ks_s1 = stats.kstest(s1, 'norm')
ks_s2 = stats.kstest(s2, 'norm')
print(f"KS test s1 (normal):      stat={ks_s1.statistic:.4f}, p={ks_s1.pvalue:.4f}")
print(f"KS test s2 (exponential): stat={ks_s2.statistic:.4f}, p={ks_s2.pvalue:.4f}")

t = np.arange(200)
ts = 0.05 * t + np.random.normal(0, 1, 200)

adf_result = adfuller(ts)
print(f"\nADF statistic: {adf_result[0]:.4f}")
print(f"ADF p-value:   {adf_result[1]:.4f}")
print(f"Stationary?    {adf_result[1] < 0.05}")

ts_diff  = np.diff(ts)
adf_diff = adfuller(ts_diff)
print(f"\nAfter differencing — p-value: {adf_diff[1]:.4f}, Stationary? {adf_diff[1] < 0.05}")


# ── 5.5 Model Monitoring / PSI ─────────────────────────────────────────────
def compute_psi(expected, actual, bins=10):
    """PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))"""
    epsilon = 1e-10
    # Create bin edges from the expected distribution
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), bins + 1)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts,   _ = np.histogram(actual,   bins=breakpoints)

    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct   = actual_counts   / len(actual)   + epsilon

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

np.random.seed(1)
train_dist = np.random.normal(50, 10, 1000)
drift_dist = np.random.normal(65, 12, 1000)

psi_value = compute_psi(train_dist, drift_dist)
print(f"PSI: {psi_value:.4f}")
print(f"Shift severity: {'Major' if psi_value > 0.2 else 'Minor' if psi_value > 0.1 else 'Stable'}")

fig, ax = plt.subplots()
ax.hist(train_dist, bins=30, alpha=0.5, density=True, label='Training', color='steelblue')
ax.hist(drift_dist, bins=30, alpha=0.5, density=True, label='Production (drifted)', color='tomato')
ax.legend(); ax.set_title('Distribution Shift Visualization')
plt.tight_layout(); plt.show()

# Concept drift: the statistical relationship between inputs and the target
#   output changes over time (the model's learned mapping becomes incorrect).
# Covariate drift: only the input feature distribution P(X) changes, while
#   the conditional P(Y|X) remains the same.
# PSI < 0.1:   No significant shift — model is stable.
# PSI 0.1–0.2: Minor shift — monitor more closely.
# PSI > 0.2:   Major shift — investigate and consider retraining.
# Retraining trigger: Schedule an automatic retraining job whenever PSI > 0.2
#   on any top-3 most-important feature over a rolling 7-day window.


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Probability Theory
# ═══════════════════════════════════════════════════════════════════════════

# ── 6.1 Core Probability ───────────────────────────────────────────────────
total = 10
red, blue, green = 4, 3, 3

p_red   = red   / total          # 0.4
p_blue  = blue  / total          # 0.3
p_green = green / total          # 0.3

# Joint: P(red first AND blue second) — without replacement
p_red_then_blue = (red / total) * (blue / (total - 1))

# Conditional: P(blue second | red first)
p_blue_given_red = blue / (total - 1)   # 3/9

# Independence: draws are NOT independent (sampling w/o replacement changes probs)
# P(blue|red) = 3/9 ≠ P(blue) = 3/10
independent = (p_blue_given_red == p_blue)

print(f"P(red)={p_red:.3f}, P(blue)={p_blue:.3f}, P(green)={p_green:.3f}")
print(f"P(red,blue): {p_red_then_blue:.4f}")
print(f"P(blue|red): {p_blue_given_red:.4f}")
print(f"P(blue):     {p_blue:.4f}")
print(f"Independent? {independent}")


# ── 6.2 Distributions in the Wild ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Normal
x = np.linspace(-4, 6, 300)
axes[0].plot(x, stats.norm.pdf(x, 0, 1),   label='N(0,1)',     color='steelblue')
axes[0].plot(x, stats.norm.pdf(x, 2, 0.5), label='N(2,0.5)',   color='tomato')
axes[0].legend(); axes[0].set_title('Normal Distribution')
# ML use case: modelling residuals/errors in linear regression

# Binomial
k = np.arange(0, 21)
axes[1].bar(k - 0.2, stats.binom.pmf(k, 20, 0.5), width=0.4, label='p=0.5', color='steelblue', alpha=0.8)
axes[1].bar(k + 0.2, stats.binom.pmf(k, 20, 0.7), width=0.4, label='p=0.7', color='tomato',    alpha=0.8)
axes[1].legend(); axes[1].set_title('Binomial Distribution')
# ML use case: number of correctly classified items in a batch (binary classification)

# Poisson
k2 = np.arange(0, 16)
axes[2].bar(k2, stats.poisson.pmf(k2, mu=3), color='mediumseagreen', alpha=0.8)
axes[2].set_title('Poisson Distribution')
# ML use case: modelling rare event counts (e.g. fraud transactions per hour)

plt.tight_layout(); plt.show()


# ── 6.3 Bayes' Theorem ─────────────────────────────────────────────────────
p_spam            = 0.30
p_free_given_spam = 0.80
p_free_given_ham  = 0.05
p_ham             = 1 - p_spam

# P("free") — law of total probability
p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham

# Bayes' theorem: P(Spam | "free") = P("free" | Spam) * P(Spam) / P("free")
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

print(f"P('free'): {p_free:.4f}")
print(f"P(Spam | 'free'): {p_spam_given_free:.4f}")

def naive_bayes_predict(prior_spam, p_word_given_spam, p_word_given_ham):
    p_ham_local = 1 - prior_spam
    p_word = p_word_given_spam * prior_spam + p_word_given_ham * p_ham_local
    return (p_word_given_spam * prior_spam) / p_word

pred = naive_bayes_predict(0.30, 0.80, 0.05)
assert abs(pred - p_spam_given_free) < 1e-9
print(f"naive_bayes_predict: {pred:.4f}")
print("6.3 passed ✓")

# Bayes term mapping:
# Prior     P(Spam)          = 0.30  — baseline belief that any email is spam
# Likelihood P(word|Spam)    = 0.80  — how often spam emails contain "free"
# Evidence  P(word)          = 0.29  — overall probability of seeing "free"
# Posterior P(Spam|word)     ≈ 0.828 — updated belief after seeing "free"


# ── 6.4 Central Limit Theorem ─────────────────────────────────────────────
np.random.seed(7)
population = np.random.exponential(scale=1.0, size=100_000)

n_samples   = 5000
sample_size = 30
sample_means = np.array([
    np.random.choice(population, size=sample_size, replace=False).mean()
    for _ in range(n_samples)
])

pop_mean = population.mean()
pop_std  = population.std()
clt_std  = pop_std / np.sqrt(sample_size)

print(f"Population mean: {pop_mean:.4f}, Population std: {pop_std:.4f}")
print(f"Sample means mean: {sample_means.mean():.4f}, std: {sample_means.std():.4f}")
print(f"CLT predicted std: {clt_std:.4f}")

fig, ax = plt.subplots()
ax.hist(sample_means, bins=50, density=True, alpha=0.6, color='steelblue', label='Sample means')
x_clt = np.linspace(sample_means.min(), sample_means.max(), 300)
ax.plot(x_clt, stats.norm.pdf(x_clt, pop_mean, clt_std), 'r-', lw=2, label='CLT Normal')
ax.legend(); ax.set_title('CLT: Distribution of Sample Means')
plt.tight_layout(); plt.show()

ks_result = stats.kstest(sample_means, 'norm', args=(pop_mean, clt_std))
print(f"KS test p-value: {ks_result.pvalue:.4f} → Approximately normal? {ks_result.pvalue > 0.05}")

# CLT Reflection:
# The CLT guarantees that regardless of the population's shape (here, exponential),
# the distribution of sample means converges to a Normal distribution as sample
# size grows. This is foundational for statistics because it justifies using
# z-tests and t-tests (which assume normality) even when the underlying data is
# not normal — as long as the sample is large enough (typically n ≥ 30).
# In ML, it underpins confidence intervals for model metrics and the validity of
# many classical statistical tests used in feature selection and model evaluation.
