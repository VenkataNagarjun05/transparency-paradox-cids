"""Measured gas for every contract operation, per round and per tenant, on a local Hardhat EVM (Cancun).

Real ML-DSA signatures (liboqs) are posted as calldata, so calldata cost is real, not estimated.
Variants: ML-DSA-44 / 65 / 87, Ed25519 (64 bytes, classical reference), and "hash only" (no signature on chain).
Usage: python3 measure_gas.py [--rpc http://127.0.0.1:8545] [--out ../results/chain_gas.json]
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
from web3 import Web3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qbfl import pqc  # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts/contracts/CIDSContract.sol/CIDSContract.json")


def deploy(w3, coordinator, bond=10**18, window=5):
    a = json.load(open(ART))
    C = w3.eth.contract(abi=a["abi"], bytecode=a["bytecode"])
    tx = C.constructor(bond, b"\x11" * 32, b"\x22" * 32, window).transact({"from": coordinator})
    rc = w3.eth.wait_for_transaction_receipt(tx)
    return w3.eth.contract(address=rc.contractAddress, abi=a["abi"]), rc.gasUsed


def send(fn, frm, value=0):
    tx = fn.transact({"from": frm, "value": value})
    rc = fn.w3.eth.wait_for_transaction_receipt(tx)
    assert rc.status == 1, "transaction reverted"
    return rc.gasUsed


def one_round(c, accts, coord, r, sig_kind, n_params, rng):
    """Run one full round for len(accts) tenants and return gas per operation (lists)."""
    g = {"openRound": [], "commit": [], "closeCommit": [], "reveal": [], "closeReveal": [], "finalizeBeacon": [], "attest": []}
    signers = {}
    g["openRound"].append(send(c.functions.openRound(r), coord))
    secrets = []
    for i, a in enumerate(accts):
        delta = rng.normal(size=n_params).astype(np.float32)
        salt = os.urandom(16)
        com = pqc.commitment(r, i, delta, salt)
        if sig_kind.startswith("ML-DSA"):
            s = signers.setdefault(i, pqc.MLDSA(sig_kind)); sig = s.sign(com)
        elif sig_kind == "Ed25519":
            sig = pqc.Ed25519().sign(com)
        else:
            sig = b""
        share = os.urandom(32)
        g["commit"].append(send(c.functions.commit(r, com, Web3.keccak(share), sig), a))
        secrets.append((hashlib.sha256(pqc.update_bytes(delta)).digest(), salt, share))
    g["closeCommit"].append(send(c.functions.closeCommit(r), coord))
    for i, a in enumerate(accts):
        uh, salt, share = secrets[i]
        g["reveal"].append(send(c.functions.reveal(r, uh, salt, share), a))
    g["closeReveal"].append(send(c.functions.closeReveal(r), coord))
    c.w3.provider.make_request("hardhat_mine", ["0x2"])                      # the beacon block (closing block + 1) must exist first
    g["finalizeBeacon"].append(send(c.functions.finalizeBeacon(r), coord))
    g["attest"].append(send(c.functions.attest(r, os.urandom(32), os.urandom(32), (1 << len(accts)) - 1), coord))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default="http://127.0.0.1:8545")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/chain_gas.json"))
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    w3 = Web3(Web3.HTTPProvider(args.rpc))
    assert w3.is_connected(), "start the node: npx hardhat node"
    accs = w3.eth.accounts
    coord = accs[0]
    rng = np.random.default_rng(0)
    out = {"chain": "Hardhat local EVM, hardfork cancun, solc 0.8.24 (optimizer 200 runs)", "block_gas_limit": 30_000_000,
           "n_params": 45_000, "runs": []}
    for n_ten in (20, 100):
        for kind in ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "Ed25519", "hash-only"):
            c, g_deploy = deploy(w3, coord)
            ten = accs[1:1 + n_ten]
            t0 = time.time()
            g_reg = [send(c.functions.register(Web3.keccak(os.urandom(32))), a, value=10**18) for a in ten]
            rounds = [one_round(c, ten, coord, r, kind, out["n_params"], rng) for r in range(1, args.rounds + 1)]
            # accountability paths (measured once each)
            r = args.rounds + 1
            c.functions.openRound(r).transact({"from": coord})
            share = os.urandom(32); salt = os.urandom(16); delta = rng.normal(size=10).astype(np.float32)
            com = pqc.commitment(r, 0, delta, salt)
            send(c.functions.commit(r, com, Web3.keccak(share), b"\x00" * 10), ten[0])
            com1 = pqc.commitment(r, 1, delta, salt)
            send(c.functions.commit(r, com1, Web3.keccak(share), b"\x00" * 10), ten[1])
            send(c.functions.closeCommit(r), coord)
            send(c.functions.reveal(r, hashlib.sha256(pqc.update_bytes(delta)).digest(), salt, share), ten[1])
            # a mismatching reveal must revert (equivocation cannot be posted)
            try:
                c.functions.reveal(r, b"\x00" * 32, salt, share).transact({"from": ten[0]})
                mismatch_reverts = False
            except Exception:
                mismatch_reverts = True
            send(c.functions.closeReveal(r), coord); c.w3.provider.make_request("hardhat_mine", ["0x2"]); send(c.functions.finalizeBeacon(r), coord)
            g_noreveal = send(c.functions.slashNoReveal(r, 0), accs[-1])
            g_propose = send(c.functions.proposeSlash(r, 1, os.urandom(32)), coord)
            for _ in range(6):
                w3.provider.make_request("evm_mine", [])
            g_exec = send(c.functions.executeSlash(r, 1), accs[-1])
            per = {k: float(np.mean(sum((rd[k] for rd in rounds), []))) for k in rounds[0]}
            round_total = float(np.mean([sum(sum(v) for v in rd.values()) for rd in rounds]))
            row = dict(n_tenants=n_ten, signature=kind, deploy=g_deploy, register=float(np.mean(g_reg)),
                       per_op_mean=per, round_total_gas=round_total, round_gas_per_tenant=round_total / n_ten,
                       round_share_of_one_block=round_total / out["block_gas_limit"],
                       slash_no_reveal=g_noreveal, propose_slash=g_propose, execute_slash=g_exec,
                       mismatched_reveal_reverts=mismatch_reverts, wall_s=time.time() - t0)
            out["runs"].append(row)
            print(f"N={n_ten:3d} {kind:9s} commit {per['commit']:9.0f}  reveal {per['reveal']:7.0f}  round {round_total:11.0f}"
                  f"  ({row['round_share_of_one_block']:.1%} of a 30M block)  mismatch reverts={mismatch_reverts}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
