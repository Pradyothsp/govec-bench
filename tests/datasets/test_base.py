import numpy as np
import pytest

from govec_bench.adapters.base import InsertItem
from govec_bench.datasets.base import ArrayItems


@pytest.fixture
def items() -> ArrayItems:
    return ArrayItems("ds", np.array([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]], dtype=np.float32))


def test_getitem__index__returns_item_with_positional_id(items: ArrayItems) -> None:
    # Arrange

    # Act
    item = items[1]

    # Assert
    assert item == InsertItem(id="ds_1", vector=[1.5, 2.0])


def test_getitem__negative_index__counts_from_the_end(items: ArrayItems) -> None:
    # Arrange

    # Act
    item = items[-1]

    # Assert
    assert item.id == "ds_2"


def test_getitem__slice__returns_list_of_items(items: ArrayItems) -> None:
    # Arrange

    # Act
    batch = items[1:3]

    # Assert
    assert batch == [InsertItem(id="ds_1", vector=[1.5, 2.0]), InsertItem(id="ds_2", vector=[2.5, 3.0])]


def test_getitem__index_out_of_range__raises_index_error(items: ArrayItems) -> None:
    # Arrange

    # Act / Assert
    with pytest.raises(IndexError):
        items[3]


def test_iter__all_items__yields_every_item_in_order(items: ArrayItems) -> None:
    # Arrange

    # Act
    ids = [item.id for item in items]

    # Assert
    assert ids == ["ds_0", "ds_1", "ds_2"]
