import numpy as np
import numpy.typing as npt

from govec_bench.types import NeighborIndices


def exact_cosine_neighbors(base: npt.ArrayLike, queries: npt.ArrayLike, k: int) -> list[NeighborIndices]:
    """Each query's true top-k base vectors by cosine similarity, found by brute force.

    Recall must be graded against the metric the databases search with. Every database here runs
    cosine, but SIFT ships ground truth computed with Euclidean distance; the two disagree on
    near-ties, so grading against SIFT's file marked correct cosine answers wrong and capped even
    a perfect search below 100% (98.0% at recall@1 on SIFT10K).
    """
    base_arr = np.asarray(base, dtype=np.float64)
    query_arr = np.asarray(queries, dtype=np.float64)

    base_unit = base_arr / np.linalg.norm(base_arr, axis=1, keepdims=True)
    query_unit = query_arr / np.linalg.norm(query_arr, axis=1, keepdims=True)
    similarity = query_unit @ base_unit.T

    # A stable sort on the negated similarity keeps exact ties in index order, so the key is
    # deterministic across runs.
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    return [row.tolist() for row in order]
