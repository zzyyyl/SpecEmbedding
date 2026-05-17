import argparse
import hashlib
import logging
import pickle
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem
from tqdm import tqdm

from SpecEmbedding.config import config
from src.data import MassBankProvider, MassSpecGymProvider, MoNAProvider
from train import setup_logging


class RateLimiter:
    """Ensure requests don't exceed PubChem's rate limits across threads."""
    def __init__(self, requests_per_second=4):
        self.delay = 1.0 / requests_per_second
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()

class PubChemFetcher:
    """Parallel fetcher for PubChem similarity candidates."""
    def __init__(self, cache_dir=None, threshold=90, max_workers=10):
        if cache_dir is None:
            cache_dir = config.data.pubchem_cache_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/smiles/{}/JSON?Threshold={}"
        self.rate_limiter = RateLimiter(requests_per_second=4)

    def _prepare_query_smiles(self, smiles):
        """Remove stereochemistry from SMILES for a cleaner, broader PubChem search."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Remove @, @@, and /, \ markers
                Chem.RemoveStereochemistry(mol)
                return Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            pass
        return smiles

    def fetch_candidates(self, smiles):
        """Worker function to fetch candidates with 5xx and exception retry mechanism."""
        if not smiles:
            return []

        query_smiles = self._prepare_query_smiles(smiles)
        safe_name = hashlib.md5(smiles.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{safe_name}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass

        encoded_smiles = urllib.parse.quote(query_smiles)
        url = self.base_url.format(encoded_smiles, self.threshold)
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()
                response = requests.get(url, timeout=30)
                
                # Handle 202 Accepted (Async)
                if response.status_code == 202:
                    data = response.json()
                    listkey = data.get('Waiting', {}).get('ListKey')
                    if listkey:
                        return self._poll_listkey(listkey, smiles, cache_file)
                    return []

                # Handle 200 OK
                if response.status_code == 200:
                    return self._parse_and_cache(response.json(), cache_file)
                
                # Handle 404
                if response.status_code == 404:
                    logging.warning(f"No results for {smiles}")
                    return []

                # Handle 5xx errors or other transient issues with retry
                if 500 <= response.status_code < 600:
                    logging.warning(f"PubChem server error {response.status_code} for {smiles}. Attempt {attempt+1}/{max_retries}")
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue
                
                # Other non-retryable errors
                logging.error(f"PubChem error {response.status_code} for {smiles}")
                break

            except (requests.exceptions.RequestException, Exception) as e:
                logging.warning(f"Attempt {attempt+1}/{max_retries} failed for {smiles}: {e}")
                time.sleep(2 ** attempt)
                continue
        
        return []

    def _poll_listkey(self, listkey, smiles, cache_file, max_retries=15):
        """Independent polling logic for a thread."""
        poll_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{listkey}/property/IsomericSMILES/JSON"
        
        for i in range(max_retries):
            # Wait with exponential backoff (this thread is "parked")
            time.sleep(5 * (i + 1)) 
            
            try:
                # Polling also counts towards rate limit but usually more lenient
                self.rate_limiter.wait()
                resp = requests.get(poll_url, timeout=30)
                
                if resp.status_code == 200:
                    return self._parse_and_cache(resp.json(), cache_file)
                elif resp.status_code == 202:
                    continue # Still working
                else:
                    break
            except Exception:
                break
        return []

    def _parse_and_cache(self, data, cache_file):
        props = data.get('PropertyTable', {}).get('Properties', [])
        smiles_list = [p['SMILES'] for p in props if 'SMILES' in p]
        unique_smiles = list(set(smiles_list))
        
        if unique_smiles:
            with open(cache_file, 'wb') as f:
                pickle.dump(unique_smiles, f)
        return unique_smiles

def main():
    parser = argparse.ArgumentParser(description="Prepare PubChem structural similarity candidate sets.")
    parser.add_argument("--data_path", type=str, default=config.data.data_path, help="Base directory containing processed dataset folders")
    parser.add_argument("--dataset", type=str, choices=["massbank", "massspecgym", "mona"], default="massbank", help="Dataset type")
    parser.add_argument("--output", type=str, required=True, help="Path to save the candidates mapping (.pkl)")
    parser.add_argument("--max_cands", type=int, default=-1, help="Maximum number of candidates to keep per SMILES")
    parser.add_argument("--cache_dir", type=str, default=config.data.pubchem_cache_path, help="Directory for API cache")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "val", "test"], help="Which data split to process")
    parser.add_argument("--threshold", type=int, default=90, help="PubChem Tanimoto similarity threshold (0-100), default: 90")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads")

    args = parser.parse_args()
    setup_logging("prepare.log")
    # 1. Load Data
    if args.dataset == "massspecgym":
        provider = MassSpecGymProvider(data_dir=args.data_path)
    elif args.dataset == "massbank":
        provider = MassBankProvider(data_dir=args.data_path)
    elif args.dataset == "mona":
        provider = MoNAProvider(data_dir=args.data_path)
    else:
        raise ValueError("--dataset is invalid")

    data = provider.load_data(mode=args.mode)

    if not data:
        logging.error("No data loaded.")
        return

    unique_smiles = sorted(list(set(item.get("smiles") for item in data)))
    logging.info(f"Unique SMILES: {len(unique_smiles)}. Using {args.workers} workers.")

    # 2. Parallel Processing
    fetcher = PubChemFetcher(cache_dir=args.cache_dir, threshold=args.threshold, max_workers=args.workers)
    final_mapping = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Map SMILES to future objects
        future_to_smiles = {executor.submit(fetcher.fetch_candidates, s): s for s in unique_smiles}
        
        for future in tqdm(as_completed(future_to_smiles), total=len(unique_smiles), desc="Parallel Fetching", ascii=True):
            s = future_to_smiles[future]
            try:
                cands = future.result()
                
                if s in cands:
                    cands.remove(s)
                cands = [s] + cands

                if args.max_cands > 0 and len(cands) > args.max_cands:
                    cands = cands[:args.max_cands]

                final_mapping[s] = cands
            except Exception as e:
                logging.error(f"Worker generated exception for {s}: {e}")
                final_mapping[s] = [s]

    # 3. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(final_mapping, f)
    
    logging.info(f"Saved {len(final_mapping)} sets to {args.output}")
    sizes = [len(v) for v in final_mapping.values()]
    if sizes:
        logging.info(f"Avg: {np.mean(sizes):.1f}, Max: {np.max(sizes)}, Min: {np.min(sizes)}")

if __name__ == "__main__":
    main()
