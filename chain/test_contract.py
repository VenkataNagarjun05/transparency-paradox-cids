"""Contract test suite for CIDSContract.sol (run: cd chain && python3 -m pytest -q test_contract.py).

Starts a local Hardhat EVM (Cancun) on port 8546, deploys a fresh contract for every test, and checks each property the paper
attributes to the contract (Table "trust model"): bonded registration, phase ordering, binding commit-reveal (including the exact
commitment layout shared with qbfl/pqc.py), the beacon, committee-only actions, attestation, both slashing paths with the dispute
window, banning, and Merkle verification of sealed probe samples. ML-DSA signatures are checked off chain against the stored hash.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from web3 import Web3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qbfl import pqc  # noqa: E402

PORT = 8546
ART = os.path.join(HERE, "artifacts/contracts/CIDSContract.sol/CIDSContract.json")
BOND = 10**18
WINDOW = 5


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def w3():
    node = subprocess.Popen(["npx", "hardhat", "node", "--port", str(PORT)], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    w = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{PORT}"))
    for _ in range(120):
        try:
            if w.is_connected():
                break
        except Exception:
            pass
        time.sleep(0.5)
    assert w.is_connected(), "hardhat node did not start"
    yield w
    node.terminate()


def merkle(leaves):
    """sha256 Merkle tree matching verifyProbe: parent = sha256(left || right). Returns root and proofs."""
    level = list(leaves); proofs = [[] for _ in leaves]; idx = list(range(len(leaves)))
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            l, r = level[i], level[i + 1]
            nxt.append(hashlib.sha256(l + r).digest())
        for k, pos in enumerate(idx):
            proofs[k].append(level[pos ^ 1]); idx[k] = pos // 2
        level = nxt
    return level[0], proofs


LEAVES = [hashlib.sha256(f"probe-{i}".encode()).digest() for i in range(8)]
ROOT, PROOFS = merkle(LEAVES)


@pytest.fixture
def env(w3):
    a = json.load(open(ART))
    accs = w3.eth.accounts
    coord, ten, outsider = accs[0], accs[1:5], accs[10]
    C = w3.eth.contract(abi=a["abi"], bytecode=a["bytecode"])
    rc = w3.eth.wait_for_transaction_receipt(C.constructor(BOND, ROOT, b"\x22" * 32, WINDOW).transact({"from": coord}))
    c = w3.eth.contract(address=rc.contractAddress, abi=a["abi"])
    return w3, c, coord, ten, outsider


def ok(w3, fn, frm, value=0):
    rc = w3.eth.wait_for_transaction_receipt(fn.transact({"from": frm, "value": value}))
    assert rc.status == 1
    return rc


def reverts(fn, frm, value=0):
    with pytest.raises(Exception):
        fn.transact({"from": frm, "value": value})


def register_all(w3, c, ten):
    for t in ten:
        ok(w3, c.functions.register(Web3.keccak(os.urandom(32))), t, value=BOND)


def committed_round(w3, c, coord, ten, r, n=None, rng=np.random.default_rng(0)):
    """Opens round r and commits for the first n tenants; returns their secrets."""
    ok(w3, c.functions.openRound(r), coord)
    sec = []
    for i, t in enumerate(ten[: n or len(ten)]):
        delta = rng.normal(size=64).astype(np.float32); salt = os.urandom(16); share = os.urandom(32)
        com = pqc.commitment(r, i, delta, salt)
        ok(w3, c.functions.commit(r, com, Web3.keccak(share), b"\x01" * 32), t)
        sec.append((hashlib.sha256(pqc.update_bytes(delta)).digest(), salt, share, delta))
    return sec


# ------------------------------------------------------------------ registry
def test_register_requires_exact_bond_and_is_once(env):
    w3, c, coord, ten, _ = env
    reverts(c.functions.register(b"\x01" * 32), ten[0], value=BOND - 1)
    ok(w3, c.functions.register(b"\x01" * 32), ten[0], value=BOND)
    reverts(c.functions.register(b"\x01" * 32), ten[0], value=BOND)
    assert c.functions.nTenants().call() == 1
    assert c.functions.tenants(0).call()[2] == BOND


# ------------------------------------------------------------------ phases and committee-only actions
def test_only_coordinator_drives_the_round(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    reverts(c.functions.openRound(1), outsider)
    ok(w3, c.functions.openRound(1), coord)
    reverts(c.functions.openRound(1), coord)                                  # a round number is used once
    reverts(c.functions.closeCommit(1), outsider)
    ok(w3, c.functions.closeCommit(1), coord)
    reverts(c.functions.closeReveal(1), outsider)
    ok(w3, c.functions.closeReveal(1), coord)
    reverts(c.functions.attest(1, b"\x01" * 32, b"\x02" * 32, 0), outsider)


def test_commit_only_in_commit_phase_once_and_by_registered(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    reverts(c.functions.commit(1, b"\x01" * 32, b"\x02" * 32, b""), ten[0])  # round not open
    ok(w3, c.functions.openRound(1), coord)
    reverts(c.functions.commit(1, b"\x01" * 32, b"\x02" * 32, b""), outsider)  # not registered
    ok(w3, c.functions.commit(1, b"\x01" * 32, b"\x02" * 32, b"sig"), ten[0])
    reverts(c.functions.commit(1, b"\x03" * 32, b"\x02" * 32, b"sig"), ten[0])  # no second commitment
    assert bytes(c.functions.sigHashOf(1, 0).call()) == Web3.keccak(b"sig")
    ok(w3, c.functions.closeCommit(1), coord)
    reverts(c.functions.commit(1, b"\x04" * 32, b"\x02" * 32, b""), ten[1])  # commit phase closed


# ------------------------------------------------------------------ binding commit-reveal
def test_reveal_binds_to_the_python_commitment_layout(env):
    w3, c, coord, ten, _ = env
    register_all(w3, c, ten)
    sec = committed_round(w3, c, coord, ten, 1)
    reverts(c.functions.reveal(1, sec[0][0], sec[0][1], sec[0][2]), ten[0])  # still in commit phase
    ok(w3, c.functions.closeCommit(1), coord)
    uh, salt, share, _ = sec[0]
    reverts(c.functions.reveal(1, b"\x00" * 32, salt, share), ten[0])        # different update
    reverts(c.functions.reveal(1, uh, os.urandom(16), share), ten[0])          # different salt
    reverts(c.functions.reveal(1, uh, salt, os.urandom(32)), ten[0])           # different beacon share
    reverts(c.functions.reveal(1, sec[1][0], sec[1][1], sec[1][2]), ten[0])  # another tenant's opening
    ok(w3, c.functions.reveal(1, uh, salt, share), ten[0])                     # exact opening of the SHA-256 layout
    reverts(c.functions.reveal(1, uh, salt, share), ten[0])                    # once only
    assert bytes(c.functions.updateHashOf(1, 0).call()) == uh


def finalize(w3, c, coord, r):
    """Close the reveal phase, let the beacon block be produced, then fix the beacon (anyone may call it)."""
    ok(w3, c.functions.closeReveal(r), coord)
    w3.provider.make_request("hardhat_mine", ["0x2"])
    ok(w3, c.functions.finalizeBeacon(r), coord)


def test_beacon_uses_a_block_produced_after_reveals_close(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    sec = committed_round(w3, c, coord, ten, 1)
    ok(w3, c.functions.closeCommit(1), coord)
    x = bytes(32)
    for i, t in enumerate(ten[:-1]):
        ok(w3, c.functions.reveal(1, sec[i][0], sec[i][1], sec[i][2]), t)
        x = bytes(a ^ b for a, b in zip(x, sec[i][2]))
    rc = ok(w3, c.functions.closeReveal(1), coord)
    target = rc.blockNumber + 1
    assert c.functions.rounds(1).call()[5] == target
    last = len(ten) - 1
    reverts(c.functions.reveal(1, sec[last][0], sec[last][1], sec[last][2]), ten[last])  # no reveal once closed
    reverts(c.functions.finalizeBeacon(1), outsider)                          # lands in the target block itself: too early
    w3.provider.make_request("hardhat_mine", ["0x2"])
    ok(w3, c.functions.finalizeBeacon(1), outsider)                           # anyone may fix the beacon
    expected = Web3.keccak(x + bytes(w3.eth.get_block(target).hash) + (1).to_bytes(4, "big"))
    assert bytes(c.functions.rounds(1).call()[1]) == expected
    assert c.functions.rounds(1).call()[0] == 4                              # Closed
    reverts(c.functions.finalizeBeacon(1), outsider)                          # once only


def test_round_is_void_if_beacon_not_fixed_within_256_blocks(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    committed_round(w3, c, coord, ten, 1)
    ok(w3, c.functions.closeCommit(1), coord); ok(w3, c.functions.closeReveal(1), coord)
    w3.provider.make_request("hardhat_mine", [hex(260)])
    ok(w3, c.functions.finalizeBeacon(1), outsider)
    rd = c.functions.rounds(1).call()
    assert rd[0] == 5 and bytes(rd[1]) == bytes(32)                          # Void, no beacon: cannot be re-drawn later
    reverts(c.functions.attest(1, b"\x01" * 32, b"\x02" * 32, 15), coord)


# ------------------------------------------------------------------ attestation
def test_attest_only_after_close_and_only_once(env):
    w3, c, coord, ten, _ = env
    register_all(w3, c, ten)
    committed_round(w3, c, coord, ten, 1)
    reverts(c.functions.attest(1, b"\x01" * 32, b"\x02" * 32, 15), coord)    # not closed
    ok(w3, c.functions.closeCommit(1), coord); ok(w3, c.functions.closeReveal(1), coord)
    reverts(c.functions.attest(1, b"\x01" * 32, b"\x02" * 32, 15), coord)    # beacon not fixed yet
    w3.provider.make_request("hardhat_mine", ["0x2"]); ok(w3, c.functions.finalizeBeacon(1), coord)
    ok(w3, c.functions.attest(1, b"\x01" * 32, b"\x02" * 32, 15), coord)
    reverts(c.functions.attest(1, b"\x03" * 32, b"\x02" * 32, 15), coord)    # cannot be overwritten
    rd = c.functions.rounds(1).call()
    assert bytes(rd[2]) == b"\x01" * 32 and rd[4] == 15


# ------------------------------------------------------------------ accountability
def test_no_reveal_slash_by_anyone_once(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    sec = committed_round(w3, c, coord, ten, 1, n=2)
    ok(w3, c.functions.closeCommit(1), coord)
    ok(w3, c.functions.reveal(1, sec[1][0], sec[1][1], sec[1][2]), ten[1])
    reverts(c.functions.slashNoReveal(1, 0), outsider)                        # round not closed yet
    ok(w3, c.functions.closeReveal(1), coord)
    reverts(c.functions.slashNoReveal(1, 1), outsider)                        # tenant 1 revealed
    reverts(c.functions.slashNoReveal(1, 2), outsider)                        # tenant 2 never committed
    ok(w3, c.functions.slashNoReveal(1, 0), outsider)
    assert c.functions.tenants(0).call()[2] == BOND // 2
    reverts(c.functions.slashNoReveal(1, 0), outsider)                        # one slash per offence


def test_validation_slash_needs_reveal_and_respects_dispute_window(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    sec = committed_round(w3, c, coord, ten, 1, n=2)
    ok(w3, c.functions.closeCommit(1), coord)
    for i in (0, 1):
        ok(w3, c.functions.reveal(1, sec[i][0], sec[i][1], sec[i][2]), ten[i])
    ok(w3, c.functions.closeReveal(1), coord)
    reverts(c.functions.proposeSlash(1, 2, b"\x09" * 32), coord)              # nothing revealed by tenant 2
    reverts(c.functions.proposeSlash(1, 0, b"\x09" * 32), outsider)           # committee only
    ok(w3, c.functions.proposeSlash(1, 0, b"\x09" * 32), coord)
    ok(w3, c.functions.proposeSlash(1, 1, b"\x09" * 32), coord)
    reverts(c.functions.proposeSlash(1, 0, b"\x09" * 32), coord)              # already pending
    reverts(c.functions.executeSlash(1, 0), outsider)                         # inside the dispute window
    ok(w3, c.functions.challenge(1), ten[1])                                  # tenant 1 disputes in time
    for _ in range(WINDOW + 1):
        w3.provider.make_request("hardhat_mine", ["0x2"])
    reverts(c.functions.challenge(1), ten[0])                                 # too late to dispute
    reverts(c.functions.executeSlash(1, 1), outsider)                         # disputed: not executable
    ok(w3, c.functions.executeSlash(1, 0), outsider)                          # unchallenged: executes
    reverts(c.functions.executeSlash(1, 0), outsider)                         # once
    assert c.functions.tenants(0).call()[2] == BOND // 2 and c.functions.tenants(1).call()[2] == BOND


def test_two_slashes_ban(env):
    w3, c, coord, ten, outsider = env
    register_all(w3, c, ten)
    for r in (1, 2):
        committed_round(w3, c, coord, ten, r, n=1)
        ok(w3, c.functions.closeCommit(r), coord); ok(w3, c.functions.closeReveal(r), coord)
        ok(w3, c.functions.slashNoReveal(r, 0), outsider)
        if r == 1:
            assert c.functions.tenants(0).call()[3] is False                  # one slash does not ban
    t = c.functions.tenants(0).call()
    assert t[2] == BOND // 4 and t[3] is True                                 # 1/2, 1/4 -> banned at the second slash
    ok(w3, c.functions.openRound(3), coord)
    reverts(c.functions.commit(3, b"\x01" * 32, b"\x02" * 32, b""), ten[0])   # banned tenants cannot commit


# ------------------------------------------------------------------ sealed probes and signatures
def test_merkle_verification_of_probe_samples(env):
    _, c, *_ = env
    for i, (leaf, proof) in enumerate(zip(LEAVES, PROOFS)):
        assert c.functions.verifyProbe(leaf, proof, i).call() is True
    assert c.functions.verifyProbe(hashlib.sha256(b"not in pool").digest(), PROOFS[0], 0).call() is False
    assert c.functions.verifyProbe(LEAVES[0], PROOFS[0], 1).call() is False   # wrong position


@pytest.mark.skipif(not pqc.HAVE_OQS, reason="liboqs not installed")
def test_mldsa_signature_recorded_and_checkable_off_chain(env):
    w3, c, coord, ten, _ = env
    signer = pqc.MLDSA("ML-DSA-65")
    ok(w3, c.functions.register(Web3.keccak(signer.pk)), ten[0], value=BOND)
    ok(w3, c.functions.openRound(1), coord)
    delta = np.arange(32, dtype=np.float32); salt = os.urandom(16); share = os.urandom(32)
    com = pqc.commitment(1, 0, delta, salt); sig = signer.sign(com)
    ok(w3, c.functions.commit(1, com, Web3.keccak(share), sig), ten[0])
    assert bytes(c.functions.sigHashOf(1, 0).call()) == Web3.keccak(sig)
    assert bytes(c.functions.tenants(0).call()[1]) == Web3.keccak(signer.pk)
    led = pqc.Ledger("ML-DSA-65"); led.register(0, signer.pk); led.post(1, 0, com, sig, salt)
    honest = led.challenge(1, 0, delta)
    assert honest["signature_valid"] and honest["reveal_matches_commitment"] and not honest["provable_equivocation"]
    lie = led.challenge(1, 0, delta + 1)
    assert lie["provable_equivocation"]
