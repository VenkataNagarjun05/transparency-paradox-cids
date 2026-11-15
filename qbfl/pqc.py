"""Post-quantum layer: ML-DSA signed update commitments and ML-KEM protected transport.

What is real here (measured, not estimated):
  * ML-DSA-{44,65,87} sign / verify and ML-KEM-{512,768,1024} keygen / encapsulate / decapsulate
    through liboqs, plus sizes.
  * A hybrid envelope: ML-KEM shared secret -> HKDF-SHA256 -> AES-256-GCM over the update bytes.
  * Calldata gas for posting a signature on an EVM chain, computed exactly from EIP-2028
    (4 gas per zero byte, 16 per non-zero byte) on the real signature bytes.
What is NOT done here: verifying ML-DSA inside the EVM. That cost is not measured by this module.
The contract design verifies off-chain and records a hash on-chain with a challenge path.
"""
from __future__ import annotations

import hashlib
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

try:
    import oqs
    HAVE_OQS = True
except Exception:  # pragma: no cover
    HAVE_OQS = False

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.exceptions import InvalidSignature


def calldata_gas(b: bytes) -> int:
    z = b.count(0)
    return 4 * z + 16 * (len(b) - z)


def update_bytes(delta: np.ndarray) -> bytes:
    return np.asarray(delta, dtype=np.float32).tobytes()


def commitment(round_: int, tenant: int, delta: np.ndarray, salt: bytes) -> bytes:
    """What gets signed: binds the update to a round, a tenant and a fresh salt (replay protection)."""
    return hashlib.sha256(struct.pack(">II", round_, tenant) + salt + hashlib.sha256(update_bytes(delta)).digest()).digest()


class MLDSA:
    def __init__(self, alg="ML-DSA-65"):
        self.alg = alg
        self.sig = oqs.Signature(alg)
        self.pk = self.sig.generate_keypair()

    def sign(self, msg: bytes) -> bytes:
        return self.sig.sign(msg)

    @staticmethod
    def verify(alg, msg, sig, pk) -> bool:
        with oqs.Signature(alg) as v:
            return bool(v.verify(msg, sig, pk))


class Ed25519:
    def __init__(self):
        self.sk = Ed25519PrivateKey.generate()
        self.pk = self.sk.public_key()

    def sign(self, msg):
        return self.sk.sign(msg)

    def verify(self, msg, sig) -> bool:
        try:
            self.pk.verify(sig, msg)
            return True
        except InvalidSignature:
            return False


def kem_envelope(kem_alg: str, recipient_pk: bytes, payload: bytes):
    """Sender side: encapsulate to the recipient's ML-KEM public key, encrypt payload with AES-GCM."""
    with oqs.KeyEncapsulation(kem_alg) as k:
        ct, ss = k.encap_secret(recipient_pk)
    key = HKDF(hashes.SHA256(), 32, None, b"qbfl-update-v1").derive(ss)
    nonce = os.urandom(12)
    return ct, nonce, AESGCM(key).encrypt(nonce, payload, b"")


def kem_open(kem: "oqs.KeyEncapsulation", ct, nonce, blob) -> bytes:
    ss = kem.decap_secret(ct)
    key = HKDF(hashes.SHA256(), 32, None, b"qbfl-update-v1").derive(ss)
    return AESGCM(key).decrypt(nonce, blob, b"")


def bench(n_params: int, n_tenants: int, sig_alg="ML-DSA-65", kem_alg="ML-KEM-768", reps=50, seed=0) -> dict:
    """Per-round cost of signing + verifying + sealing K tenant updates, measured."""
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=n_params).astype(np.float32)
    out = {"n_params": n_params, "update_bytes": n_params * 4, "n_tenants": n_tenants}
    s = MLDSA(sig_alg)
    ed = Ed25519()
    msg = commitment(1, 0, delta, os.urandom(16))
    ts = {k: [] for k in ("pq_sign", "pq_verify", "ed_sign", "ed_verify", "kem_keygen", "kem_encap", "kem_decap")}
    for _ in range(reps):
        t = time.perf_counter(); sg = s.sign(msg); ts["pq_sign"].append(time.perf_counter() - t)
        t = time.perf_counter(); ok = MLDSA.verify(sig_alg, msg, sg, s.pk); ts["pq_verify"].append(time.perf_counter() - t)
        assert ok
        t = time.perf_counter(); eg = ed.sign(msg); ts["ed_sign"].append(time.perf_counter() - t)
        t = time.perf_counter(); assert ed.verify(msg, eg); ts["ed_verify"].append(time.perf_counter() - t)
    with oqs.KeyEncapsulation(kem_alg) as kem:
        t = time.perf_counter()
        for _ in range(reps):
            pk = kem.generate_keypair()
        ts["kem_keygen"] = [(time.perf_counter() - t) / reps]
        pt = update_bytes(delta)
        for _ in range(reps):
            t = time.perf_counter(); ct, nonce, blob = kem_envelope(kem_alg, pk, pt); e = time.perf_counter() - t
            ts["kem_encap"].append(e)
            t = time.perf_counter(); back = kem_open(kem, ct, nonce, blob); ts["kem_decap"].append(time.perf_counter() - t)
            assert back == pt
    for k, v in ts.items():
        out[f"{k}_ms"] = float(np.median(v) * 1e3)
    out.update(pq_pk_bytes=len(s.pk), pq_sig_bytes=len(sg), ed_sig_bytes=len(eg), kem_pk_bytes=len(pk), kem_ct_bytes=len(ct),
               envelope_overhead_bytes=len(ct) + 12 + 16,
               pq_sig_calldata_gas=calldata_gas(sg), ed_sig_calldata_gas=calldata_gas(eg),
               round_sign_verify_ms=n_tenants * (out["pq_sign_ms"] + out["pq_verify_ms"]),
               round_seal_open_ms=n_tenants * (out["kem_encap_ms"] + out["kem_decap_ms"]),
               round_calldata_gas=n_tenants * (calldata_gas(sg) + 32))
    return out


def security_checks(sig_alg="ML-DSA-65", kem_alg="ML-KEM-768", n_params=2000) -> dict:
    """Properties the protocol relies on, each tested by an attempted attack. Returns pass/fail."""
    rng = np.random.default_rng(1)
    delta = rng.normal(size=n_params).astype(np.float32)
    a, b = MLDSA(sig_alg), MLDSA(sig_alg)
    salt = os.urandom(16)
    msg = commitment(7, 3, delta, salt)
    sg = a.sign(msg)
    res = {}
    res["valid_signature_accepts"] = MLDSA.verify(sig_alg, msg, sg, a.pk)
    tam = delta.copy(); tam[0] += 1e-3
    res["tampered_update_rejected"] = not MLDSA.verify(sig_alg, commitment(7, 3, tam, salt), sg, a.pk)
    res["wrong_round_rejected"] = not MLDSA.verify(sig_alg, commitment(8, 3, delta, salt), sg, a.pk)
    res["wrong_tenant_rejected"] = not MLDSA.verify(sig_alg, commitment(7, 4, delta, salt), sg, a.pk)
    res["other_key_rejected"] = not MLDSA.verify(sig_alg, msg, sg, b.pk)
    bad = bytearray(sg); bad[10] ^= 1
    res["bitflipped_signature_rejected"] = not MLDSA.verify(sig_alg, msg, bytes(bad), a.pk)
    with oqs.KeyEncapsulation(kem_alg) as kem:
        pk = kem.generate_keypair()
        ct, nonce, blob = kem_envelope(kem_alg, pk, update_bytes(delta))
        res["envelope_roundtrip"] = kem_open(kem, ct, nonce, blob) == update_bytes(delta)
        tb = bytearray(blob); tb[5] ^= 1
        try:
            kem_open(kem, ct, nonce, bytes(tb)); res["tampered_ciphertext_rejected"] = False
        except Exception:
            res["tampered_ciphertext_rejected"] = True
    return res


class Ledger:
    """Minimal accountable ledger used by the experiments: records signed commitments and lets anyone
    check a dispute (challenge) by re-verifying the signature over the revealed update."""

    def __init__(self, sig_alg="ML-DSA-65"):
        self.alg = sig_alg
        self.keys, self.entries, self.bond = {}, {}, {}

    def register(self, tenant, pk, bond=1.0):
        self.keys[tenant] = pk
        self.bond[tenant] = bond

    def post(self, round_, tenant, commit, sig, salt):
        key = (round_, tenant)
        if key in self.entries:
            return False  # one commitment per tenant per round
        self.entries[key] = (commit, sig, salt)
        return True

    def challenge(self, round_, tenant, revealed_delta) -> dict:
        """Dispute path: re-derive the commitment from the revealed update and re-verify the signature."""
        commit, sig, salt = self.entries[(round_, tenant)]
        ok_hash = commitment(round_, tenant, revealed_delta, salt) == commit
        ok_sig = MLDSA.verify(self.alg, commit, sig, self.keys[tenant])
        return {"reveal_matches_commitment": ok_hash, "signature_valid": ok_sig,
                # a valid signature over a commitment that the revealed update does not match is
                # publicly checkable proof that this tenant equivocated
                "provable_equivocation": ok_sig and not ok_hash}

    def slash(self, tenant, frac=0.5):
        self.bond[tenant] = max(0.0, self.bond[tenant] - frac)
        return self.bond[tenant]
