# The transparency paradox in collaborative intrusion detection

Code to reproduce the experiments in

> A. Venkata Nagarjun and Sujatha Rajkumar. *The Transparency Paradox in Collaborative Intrusion Detection: Adaptive Attacks on
> Smart-Contract-Governed Federated Learning with Post-Quantum Signatures in Cloud Networks.* Manuscript under review, 2026.

When a federated intrusion detector puts its acceptance rule in a smart contract, every tenant can read that rule, including the
malicious ones. The code here simulates such a detector with twelve acceptance rules and six attackers, one of which runs a private
copy of the published rule and optimises its update against it. It also contains the Solidity contract (commit-reveal, block-hash
beacon, committee attestation, slashing) and the post-quantum layer (ML-DSA signatures, ML-KEM envelopes).

## Contents

| Path | What is there |
|---|---|
| `qbfl/` | Python package: data loading and tenant construction (`data.py`), the MLP (`model.py`), acceptance rules (`defenses.py`), attackers (`attacks.py`), the federated training loop (`fl.py`), slashing rules (`accountability.py`), the post-quantum layer (`pqc.py`), experiment grids (`grid.py`) and statistics helpers (`analysis.py`) |
| `notebooks/` | The four Kaggle notebooks, R1 to R4, that run every experiment in the paper |
| `chain/` | The contract (`contracts/CIDSContract.sol`), Hardhat setup, contract tests, gas measurement and a 40-round end-to-end run |
| `results/best_params.json` | The acceptance-rule settings selected by tuning in R1, used by the end-to-end chain run |
| `tools/` | `build_replication.py` regenerates the notebooks from the package, `mock_run_notebook.py` dry-runs a notebook on fake data, `test_eprocess.py` checks the slashing rules on synthetic tenants |
| `tests/` | Unit tests |

## Setup

```bash
git clone https://github.com/VenkataNagarjun05/transparency-paradox-cids.git
cd transparency-paradox-cids
pip install -r requirements.txt
python3 -m pytest -q tests
```

## Running the experiments

The experiments run on Kaggle (GPU T4 x2, internet on). Import `notebooks/quantum_bc_fl_cids_R1.ipynb` and run all cells. It downloads
the four NetFlow-v2 datasets with `kagglehub`, tunes every rule on half A of the data, evaluates on half B, and writes a results bundle
with one JSON line per run. R1 takes about 7 GPU hours. Before running R2, add the Kubernetes dataset `redamorsli/kube-ids0` as a
notebook input (about 5 GPU hours). R3 and R4 run the same way. If a session is interrupted, rerunning the notebook continues from the
last finished run.

To check a notebook end to end on fake data before spending GPU time:

```bash
python3 tools/mock_run_notebook.py notebooks/quantum_bc_fl_cids_R1.ipynb
```

## Smart contract

```bash
cd chain && npm install && npx hardhat compile
python3 -m pytest -q test_contract.py     # 11 contract tests on a local EVM
npx hardhat node &                        # then, in a second shell:
python3 measure_gas.py && python3 integration.py
```

## Data

All datasets are public and none are included here: the NetFlow-v2 versions of UNSW-NB15, CSE-CIC-IDS2018, ToN-IoT and BoT-IoT
(Kaggle `dhoogla/nfunswnb15v2`, `dhoogla/nfcsecicids2018v2`, `dhoogla/nftoniotv2`, `dhoogla/nfbotiotv2`; CC BY-NC-SA 4.0) and the
Kubernetes intrusion-detection dataset `redamorsli/kube-ids0`.

## Citation

```bibtex
@article{nagarjun2026transparency,
  author = {Nagarjun, A. Venkata and Rajkumar, Sujatha},
  title  = {The Transparency Paradox in Collaborative Intrusion Detection: Adaptive Attacks on Smart-Contract-Governed
            Federated Learning with Post-Quantum Signatures in Cloud Networks},
  note   = {Manuscript under review},
  year   = {2026}
}
```

Contact: Sujatha Rajkumar (corresponding author), Sujatha.r@vit.ac.in, School of Electronics Engineering, Vellore Institute of
Technology, Vellore, India.
