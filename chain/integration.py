"""End-to-end run: a federated IDS training loop that drives the contract every round.

Each round, for every tenant: the real model update is committed (SHA-256 commitment, ML-DSA-65 signature via liboqs,
posted as calldata), then revealed against that commitment on chain. The on-chain beacon (XOR of the tenants'
revealed shares, mixed with the hash of a block produced after the reveal phase closed) seeds the sealed probe draw used by the validation rule. The committee then attests the new global
model hash, the reputation-vector root and the accepted bitmap, and proposes slashes for updates that failed validation
by more than the slash margin. Every slash proposal is backed by off-chain evidence that anyone can re-check: the
ML-DSA-signed commitment plus the revealed update (qbfl.pqc.Ledger.challenge).

Data here are SYNTHETIC on purpose: this run checks that the protocol works end to end and measures chain cost and
latency per round; gas does not depend on the data. Detection results are reported only from the Kaggle studies.
Usage: python3 integration.py [--rounds 40] [--defense sealed_fam] [--attack probe_memorize]
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
from web3 import Web3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qbfl import pqc  # noqa: E402
from qbfl.data import build_world, split_pools, synthetic_pools  # noqa: E402
from qbfl.fl import Config, run_one  # noqa: E402
from measure_gas import deploy, send  # noqa: E402


class ChainHooks:
    def __init__(self, w3, c, tenants, coord, sig_alg="ML-DSA-65"):
        self.w3, self.c, self.ten, self.coord = w3, c, tenants, coord
        self.signers = {i: pqc.MLDSA(sig_alg) for i in range(len(tenants))}
        self.ledger = pqc.Ledger(sig_alg)
        self.gas, self.wall, self.salts, self.log = [], [], {}, []
        self.reg_gas = []
        for i, a in enumerate(tenants):
            self.reg_gas.append(send(c.functions.register(Web3.keccak(self.signers[i].pk)), a, value=10**18))
            self.ledger.register(i, self.signers[i].pk)

    def before(self, r, U_sub, mal):
        t0 = time.time(); g = 0; rr = r + 1
        g += send(self.c.functions.openRound(rr), self.coord)
        secrets = {}
        for i, a in enumerate(self.ten):
            delta = U_sub[i].detach().cpu().numpy().astype(np.float32)
            salt = os.urandom(16); share = os.urandom(32)
            com = pqc.commitment(rr, i, delta, salt)
            sig = self.signers[i].sign(com)
            g += send(self.c.functions.commit(rr, com, Web3.keccak(share), sig), a)
            self.ledger.post(rr, i, com, sig, salt)
            secrets[i] = (hashlib.sha256(pqc.update_bytes(delta)).digest(), salt, share, delta)
        g += send(self.c.functions.closeCommit(rr), self.coord)
        for i, a in enumerate(self.ten):
            uh, salt, share, _ = secrets[i]
            g += send(self.c.functions.reveal(rr, uh, salt, share), a)
        g += send(self.c.functions.closeReveal(rr), self.coord)
        self.w3.provider.make_request("hardhat_mine", ["0x2"])                # beacon block = closing block + 1
        g += send(self.c.functions.finalizeBeacon(rr), self.coord)
        self.secrets, self._g, self._t0 = secrets, g, t0
        return bytes(self.c.functions.rounds(rr).call()[1])          # on-chain beacon

    def after(self, r, theta, defense, mal):
        rr = r + 1; g = self._g
        agg = hashlib.sha256(theta.detach().cpu().numpy().astype(np.float32).tobytes()).digest()
        rep = getattr(defense, "rep", None)
        rep_root = hashlib.sha256(rep.detach().cpu().numpy().astype(np.float32).tobytes()).digest() if rep is not None else b"\x00" * 32
        last = getattr(defense, "last", {}) or {}
        ok = last.get("ok")
        accepted = int(sum(1 << i for i in range(len(self.ten)) if ok is None or ok[i] > 0))
        g += send(self.c.functions.attest(rr, agg, rep_root, accepted), self.coord)
        proposed = []
        if "dmg" in last and np.isfinite(last.get("thr", np.inf)):
            margin = defense.slash_margin * (last["thr"] if defense.mode == "rel_fam" else 1.0)
            for i in np.flatnonzero(last["dmg"] > last["thr"] + margin):
                i = int(i)
                ev = self.ledger.challenge(rr, i, self.secrets[i][3])          # anyone can re-check this
                assert ev["signature_valid"] and ev["reveal_matches_commitment"]
                evidence = hashlib.sha256(self.secrets[i][0] + np.asarray(defense.last_idx).tobytes()
                                          + np.float32(last["dmg"][i]).tobytes()).digest()
                g += send(self.c.functions.proposeSlash(rr, i, evidence), self.coord)
                proposed.append(i)
        self.gas.append(g); self.wall.append(time.time() - self._t0)
        self.log.append(dict(round=rr, gas=g, wall_s=self.wall[-1], proposed=proposed, malicious=list(mal)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default="http://127.0.0.1:8545")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--defense", default="sealed_fam")
    ap.add_argument("--attack", default="probe_memorize")
    ap.add_argument("--out", default=os.path.join(HERE, "../results/chain_integration.json"))
    args = ap.parse_args()
    w3 = Web3(Web3.HTTPProvider(args.rpc)); assert w3.is_connected(), "start the node: npx hardhat node"
    accs = w3.eth.accounts; coord = accs[0]
    best = json.load(open(os.path.join(HERE, "../results/best_params.json")))
    tp, fp_ = split_pools(synthetic_pools(n_per=16000))
    world = build_world(fp_, n_tenants=20, n_train=600, n_test=1500, n_audit=4000)
    c, g_dep = deploy(w3, coord)
    hooks = ChainHooks(w3, c, accs[1:21], coord)
    cfg = Config(defense=args.defense, attack=args.attack, mal_frac=0.3, rounds=args.rounds, eval_every=10,
                 local_epochs=1, dparams=best.get(args.defense, {}), seed=7)
    t0 = time.time()
    row = run_one(world, cfg, "cpu", hooks=hooks)
    mal = set(row and hooks.log[-1]["malicious"])
    props = [i for l in hooks.log for i in l["proposed"]]
    attack_rounds = [l for l in hooks.log if l["round"] - 1 >= cfg.attack_start]
    # execute one unchallenged slash after the window to close the loop on chain
    executed = None
    for l in hooks.log:
        if l["proposed"]:
            for _ in range(6):
                w3.provider.make_request("hardhat_mine", ["0x2"])
            executed = dict(round=l["round"], tenant=l["proposed"][0],
                            gas=send(c.functions.executeSlash(l["round"], l["proposed"][0]), accs[-1]),
                            bond_after_wei=c.functions.tenants(l["proposed"][0]).call()[2])
            break
    out = dict(note="synthetic data; checks the protocol end to end and measures chain cost per round",
               defense=args.defense, attack=args.attack, rounds=args.rounds, n_tenants=20, signature="ML-DSA-65",
               deploy_gas=g_dep, register_gas_mean=float(np.mean(hooks.reg_gas)),
               round_gas_mean=float(np.mean(hooks.gas)), round_gas_max=int(np.max(hooks.gas)),
               chain_wall_s_per_round_mean=float(np.mean(hooks.wall)), total_wall_s=time.time() - t0,
               slash_proposals=len(props),
               slash_proposals_on_malicious=sum(1 for i in props if i in mal),
               slash_proposals_on_honest=sum(1 for i in props if i not in mal),
               attack_rounds=len(attack_rounds), executed_slash=executed,
               onchain_commitments_verified=20 * args.rounds, final_metrics=row["final"], per_round=hooks.log)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_round", "final_metrics")}, indent=1, default=float))
    print("final (synthetic) metrics:", {k: round(v, 3) for k, v in row["final"].items() if k in ("macro_f1", "rec_target", "fpr")})


if __name__ == "__main__":
    main()
