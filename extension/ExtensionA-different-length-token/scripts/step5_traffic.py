
from pathlib import Path
from track_llm_apis.config import config
from track_llm_apis.sampling.bi_phase_1 import _load_traffic_prompts

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config.data_dir = root / "data"
    config.datasets_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing LMSYS background traffic...", flush=True)
    traffic = _load_traffic_prompts()
    if not traffic:
        raise RuntimeError("The repository loader returned no background requests.")

    print("Background requests:", len(traffic))
    print("Processed dataset directory:", config.datasets_dir)
    print("STEP 5 COMPLETE")
