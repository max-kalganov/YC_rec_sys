from typing import Optional, List, Iterable

import numpy as np


class BatchesComposer:
    def __init__(self, batch_size: Optional[int] = None):
        assert batch_size is None or batch_size >= 1, f"incorrect batch size = {batch_size}"
        self.batch_size = batch_size

    def get_batches(self, samples_generator: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        pass
