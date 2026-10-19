"""Tests for face_verifier service."""

import math
import pytest
from services.face_verifier import verify_embedding, cosine_similarity, FaceVerificationError

# ── helpers ───────────────────────────────────────────────────────────────────

def unit_vec(dim: int, index: int = 0) -> list[float]:
    """Return a unit vector in `dim` dimensions with 1.0 at `index`."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def norm_vec(values: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in values))
    return [x / mag for x in values]


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_identical_embeddings_give_similarity_1():
    import numpy as np
    v = np.array([0.5, 0.3, -0.2, 0.7] * 128, dtype=float)
    arr = list(v)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_orthogonal_embeddings_give_similarity_0():
    import numpy as np
    a = np.array(unit_vec(512, 0), dtype=float)
    b = np.array(unit_vec(512, 1), dtype=float)
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_zero_vector_returns_0():
    import numpy as np
    a = np.zeros(512)
    b = np.ones(512)
    assert cosine_similarity(a, b) == 0.0


# ── verify_embedding ──────────────────────────────────────────────────────────

def _make_similar_embeddings(dim=512, noise=0.05):
    """Return (base, noisy) where cosine similarity > 0.9."""
    import numpy as np
    base = np.random.default_rng(42).standard_normal(dim)
    base /= np.linalg.norm(base)
    noisy = base + np.random.default_rng(7).standard_normal(dim) * noise
    noisy /= np.linalg.norm(noisy)
    return base.tolist(), noisy.tolist()


def test_verified_when_embeddings_are_similar():
    base, noisy = _make_similar_embeddings(noise=0.01)
    result = verify_embedding(noisy, base, threshold=0.72)
    assert result["verified"] is True
    assert result["similarity"] >= 0.72


def test_not_verified_when_embeddings_are_different():
    import numpy as np
    rng = np.random.default_rng(99)
    a = rng.standard_normal(512)
    b = rng.standard_normal(512)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    result = verify_embedding(a.tolist(), b.tolist(), threshold=0.72)
    assert result["verified"] is False


def test_invalid_embedding_raises():
    with pytest.raises(FaceVerificationError) as exc_info:
        verify_embedding([1.0] * 10, [1.0] * 512)  # too short
    assert exc_info.value.code == "INVALID_EMBEDDING"


def test_dimension_mismatch_raises():
    with pytest.raises(FaceVerificationError) as exc_info:
        verify_embedding([1.0] * 128, [1.0] * 512)
    assert exc_info.value.code == "EMBEDDING_DIMENSION_MISMATCH"


def test_nan_in_embedding_raises():
    import math
    bad = [float("nan")] * 512
    good = [1.0] * 512
    with pytest.raises(FaceVerificationError):
        verify_embedding(bad, good)


def test_threshold_is_respected():
    base, noisy = _make_similar_embeddings(noise=0.01)
    # Force threshold above the actual similarity
    result_strict = verify_embedding(noisy, base, threshold=0.9999)
    result_loose = verify_embedding(noisy, base, threshold=0.50)
    assert result_strict["verified"] is False
    assert result_loose["verified"] is True


@pytest.mark.parametrize('bad', [
    [[1.0]] * 512, ['not-a-number'] * 512, [True] * 512,
    [0.0] * 512, [101.0] * 512, [1.0] * 2049,
])
def test_malformed_vectors_raise_domain_error(bad):
    with pytest.raises(FaceVerificationError) as exc:
        verify_embedding(bad, [1.0] * 512)
    assert exc.value.code == 'INVALID_EMBEDDING'

@pytest.mark.parametrize('threshold', [0, -1, 1.1, float('nan')])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(FaceVerificationError):
        verify_embedding([1.0] * 512, [1.0] * 512, threshold=threshold)
