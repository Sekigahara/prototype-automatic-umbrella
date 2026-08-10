from pathlib import Path
from datetime import datetime

import logging
from tqdm.auto import tqdm

def setup_logging(output_dir="hmm_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = (
        output_dir /
        f"hmm_run_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )

    return output_dir, log_file