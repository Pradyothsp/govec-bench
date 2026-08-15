from govec_bench.adapters.base import InsertItem, VectorDBAdapter

SETUP_BATCH_SIZE = 100


def load_dataset(adapter: VectorDBAdapter, items: list[InsertItem], batch_size: int = SETUP_BATCH_SIZE) -> None:
    # Chunked rather than one giant batch_insert call -- some backends (e.g.
    # Chroma) enforce a max batch size well below typical dataset sizes.
    for i in range(0, len(items), batch_size):
        adapter.batch_insert(items[i : i + batch_size])
