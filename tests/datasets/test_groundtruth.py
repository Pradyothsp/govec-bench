from govec_bench.datasets.groundtruth import exact_cosine_neighbors


def test_exact_cosine_neighbors__euclidean_and_cosine_disagree__ranks_by_cosine() -> None:
    # Arrange
    # Index 0 points exactly along the query but is long, so it is far away in straight-line
    # distance; index 1 is close by Euclidean distance but points elsewhere. Cosine must pick 0.
    query = [1.0, 0.0]
    base = [[10.0, 0.0], [0.9, 0.4]]

    # Act
    neighbors = exact_cosine_neighbors(base, [query], k=1)

    # Assert
    assert neighbors == [[0]]


def test_exact_cosine_neighbors__several_queries__returns_k_indices_per_query_nearest_first() -> None:
    # Arrange
    base = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    queries = [[0.0, 2.0], [3.0, 0.1]]

    # Act
    neighbors = exact_cosine_neighbors(base, queries, k=2)

    # Assert
    assert neighbors == [[1, 2], [0, 2]]


def test_exact_cosine_neighbors__vectors_tie_exactly__keeps_index_order() -> None:
    # Arrange
    # Same direction, different lengths: identical cosine similarity to the query.
    base = [[2.0, 2.0], [1.0, 1.0], [5.0, 5.0]]

    # Act
    neighbors = exact_cosine_neighbors(base, [[1.0, 1.0]], k=3)

    # Assert
    assert neighbors == [[0, 1, 2]]
