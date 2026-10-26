import json
from pathlib import Path

import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
store_path = project_root / "data" / "raw" / "store.csv"
manifest_path = project_root / "config" / "store_sample.json"

stores = pd.read_csv(store_path)
store_ids = np.sort(stores["Store"].unique())

if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_ids = np.asarray(manifest["store_ids"], dtype=int)
    print("Reusing the saved store sample.")
else:
    seed = 42
    generator = np.random.default_rng(seed)
    selected_ids = np.sort(generator.choice(store_ids, size=50, replace=False))
    manifest = {
        "sampling_method": "uniform random sample without replacement from sorted Store IDs",
        "seed": seed,
        "n_stores": len(selected_ids),
        "store_ids": selected_ids.tolist(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

sample_metadata = stores.loc[stores["Store"].isin(selected_ids)]
print(f"Saved {len(selected_ids)} store IDs to {manifest_path.relative_to(project_root)}")
print("Store type counts:\n", sample_metadata["StoreType"].value_counts().sort_index())
print("Store IDs:", selected_ids.tolist())
