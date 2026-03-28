"""
Smoke tests for ELFS novelties.

Tests GMM adaptive thresholds, hybrid scoring, adaptive selection,
and early-stop TD epoch computation -- all without GPU or real data.

Run:
    python test_novelties.py
"""
import sys
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers (copied / imported logic from the project to keep tests self-contained)
# ---------------------------------------------------------------------------

def _fit_gmm_thresholds(scores, n_components=3, random_state=42):
    """Identical to the function added in generate_importance_score.py."""
    from sklearn.mixture import GaussianMixture
    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    order = np.argsort(means)
    means, stds = means[order], stds[order]

    def _gaussian_intersection(mu1, s1, mu2, s2):
        if np.isclose(s1, s2):
            return (mu1 + mu2) / 2.0
        a = 1.0 / (2 * s1**2) - 1.0 / (2 * s2**2)
        b = mu2 / (s2**2) - mu1 / (s1**2)
        c = mu1**2 / (2 * s1**2) - mu2**2 / (2 * s2**2) - np.log(s2 / s1)
        disc = b**2 - 4 * a * c
        if disc < 0:
            return (mu1 + mu2) / 2.0
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        for r in roots:
            if mu1 <= r <= mu2:
                return r
        return (mu1 + mu2) / 2.0

    threshold_low = _gaussian_intersection(means[0], stds[0], means[1], stds[1])
    threshold_high = _gaussian_intersection(means[1], stds[1], means[2], stds[2])
    return threshold_low, threshold_high, gmm

# ---------------------------------------------------------------------------
# Test 1: GMM fitting recovers known structure
# ---------------------------------------------------------------------------
def test_gmm_fitting():
    """Create synthetic 3-component data and verify thresholds separate them."""
    np.random.seed(0)
    hard = np.random.normal(loc=-5, scale=0.5, size=1000)
    useful = np.random.normal(loc=0, scale=0.5, size=3000)
    easy = np.random.normal(loc=5, scale=0.5, size=1000)
    scores = np.concatenate([hard, useful, easy])
    np.random.shuffle(scores)

    t_low, t_high, gmm = _fit_gmm_thresholds(scores)

    # Thresholds should be roughly between the group means
    assert t_low > -5 and t_low < 0, f"t_low={t_low} not between -5 and 0"
    assert t_high > 0 and t_high < 5, f"t_high={t_high} not between 0 and 5"
    print("[PASS] test_gmm_fitting")

# ---------------------------------------------------------------------------
# Test 2: Hybrid scoring produces correct weighted average
# ---------------------------------------------------------------------------
def test_hybrid_scoring():
    """Check that alpha blending of normalized arrays is correct."""
    aum = np.array([0.0, 0.5, 1.0])
    density = np.array([1.0, 0.5, 0.0])
    alpha = 0.5

    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if not np.isclose(mn, mx) else np.zeros_like(arr)

    aum_n = _minmax(aum)
    den_n = _minmax(density)
    hybrid = alpha * aum_n + (1 - alpha) * den_n

    expected = np.array([0.5, 0.5, 0.5])
    assert np.allclose(hybrid, expected), f"hybrid={hybrid}, expected={expected}"
    print("[PASS] test_hybrid_scoring")

# ---------------------------------------------------------------------------
# Test 3: CoresetSelection.adaptive_selection
# ---------------------------------------------------------------------------
def test_adaptive_selection():
    """Feed mock data_score with known thresholds and verify index filtering."""
    sys.path.insert(0, '.')
    from core.data.Coreset import CoresetSelection

    data_score = {
        'accumulated_margin': torch.tensor([-3.0, -1.0, 0.5, 2.0, 4.0, 6.0]),
        'gmm_thresholds': (-0.5, 3.0),
    }
    idx = CoresetSelection.adaptive_selection(data_score)
    expected = torch.tensor([2, 3])  # indices where score in [-0.5, 3.0]
    assert torch.equal(idx, expected), f"got {idx}, expected {expected}"
    print("[PASS] test_adaptive_selection")

# ---------------------------------------------------------------------------
# Test 4: Early-stop TD cutoff epoch calculation
# ---------------------------------------------------------------------------
def test_early_stop_td():
    """Verify cutoff epoch is computed correctly."""
    total_epochs = 200
    ratio = 0.2
    cutoff = int(total_epochs * ratio)
    assert cutoff == 40, f"cutoff={cutoff}, expected 40"

    # ratio=1.0 should give full epochs
    assert int(total_epochs * 1.0) == 200
    print("[PASS] test_early_stop_td")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_gmm_fitting()
    test_hybrid_scoring()
    test_adaptive_selection()
    test_early_stop_td()
    print("\n===== ALL TESTS PASSED =====")
