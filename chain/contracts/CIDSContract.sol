// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Contract layer for contract-governed collaborative intrusion detection with post-quantum signatures
/// @notice Consortium setting: a validator committee (the coordinator address, a multisig in deployment)
///         drives the round phases and posts attestations. Model updates and ML-DSA signatures never sit
///         in storage: the update stays off chain, its SHA-256 commitment and the hash of its signature are
///         stored, and the signature bytes remain retrievable from the transaction input for disputes.
///         The commitment layout is identical to the Python layer (qbfl/pqc.py):
///             sha256( uint32 round || uint32 tenant || bytes16 salt || sha256(update) )
contract CIDSContract {
    address public coordinator;
    uint96 public bondAmount;
    bytes32 public probeRoot;      // Merkle root (sha256) of the sealed probe pool, fixed at setup
    bytes32 public paramsHash;     // hash of the public rule constants (kappa, s_min, ...)
    uint64 public disputeWindow;   // blocks a tenant has to challenge a validation slash

    struct Tenant { address addr; bytes32 pkHash; uint96 bond; bool banned; }
    Tenant[] public tenants;
    mapping(address => uint32) public idPlusOne;

    /// Round schedule: Commit -> Reveal -> Drawing (reveals closed, shares fixed, beacon block not yet produced)
    ///                 -> Closed (beacon fixed from a block produced after the reveals closed) or Void.
    enum Phase { None, Commit, Reveal, Drawing, Closed, Void }
    struct Round { Phase phase; bytes32 beacon; bytes32 aggHash; bytes32 repRoot; uint256 accepted; uint64 target; }
    uint64 public constant BEACON_DELAY = 1;     // beacon block = closing block + BEACON_DELAY
    mapping(uint32 => Round) public rounds;

    mapping(uint32 => mapping(uint32 => bytes32)) public commitOf;      // round => tenant => commitment
    mapping(uint32 => mapping(uint32 => bytes32)) public sigHashOf;     // keccak256(ML-DSA signature)
    mapping(uint32 => mapping(uint32 => bytes32)) public rCommitOf;     // beacon share commitment
    mapping(uint32 => mapping(uint32 => bytes32)) public updateHashOf;  // revealed sha256(update)
    mapping(uint32 => bytes32) public beaconAcc;

    struct PendingSlash { bytes32 evidence; uint64 deadline; bool challenged; bool done; }
    mapping(uint32 => mapping(uint32 => PendingSlash)) public pending;

    event Registered(uint32 indexed id, address addr, bytes32 pkHash);
    event Committed(uint32 indexed round, uint32 indexed id, bytes32 commitment, bytes32 sigHash);
    event Revealed(uint32 indexed round, uint32 indexed id, bytes32 updateHash);
    event RevealClosed(uint32 indexed round, bytes32 shareXor, uint64 beaconBlock);
    event BeaconFinal(uint32 indexed round, bytes32 beacon);
    event RoundVoided(uint32 indexed round);
    event Attested(uint32 indexed round, bytes32 aggHash, bytes32 repRoot, uint256 accepted);
    event SlashProposed(uint32 indexed round, uint32 indexed id, bytes32 evidence, uint64 deadline);
    event Challenged(uint32 indexed round, uint32 indexed id);
    event Slashed(uint32 indexed round, uint32 indexed id, uint96 amount, uint8 reason);

    modifier onlyCoordinator() { require(msg.sender == coordinator, "not coordinator"); _; }

    constructor(uint96 _bond, bytes32 _probeRoot, bytes32 _paramsHash, uint64 _disputeWindow) {
        coordinator = msg.sender;
        bondAmount = _bond;
        probeRoot = _probeRoot;
        paramsHash = _paramsHash;
        disputeWindow = _disputeWindow;
    }

    // ------------------------------------------------------------------ registry
    function register(bytes32 pkHash) external payable returns (uint32 id) {
        require(idPlusOne[msg.sender] == 0, "registered");
        require(msg.value == bondAmount, "bond");
        id = uint32(tenants.length);
        tenants.push(Tenant(msg.sender, pkHash, uint96(msg.value), false));
        idPlusOne[msg.sender] = id + 1;
        emit Registered(id, msg.sender, pkHash);
    }

    function _id() internal view returns (uint32) {
        uint32 v = idPlusOne[msg.sender];
        require(v != 0, "unknown tenant");
        require(!tenants[v - 1].banned, "banned");
        return v - 1;
    }

    // ------------------------------------------------------------------ round phases
    function openRound(uint32 r) external onlyCoordinator {
        require(rounds[r].phase == Phase.None, "round used");
        rounds[r].phase = Phase.Commit;
    }

    /// @param c commitment (sha256 layout above); @param rCommit keccak256(beacon share); @param sig ML-DSA signature over c
    function commit(uint32 r, bytes32 c, bytes32 rCommit, bytes calldata sig) external {
        require(rounds[r].phase == Phase.Commit, "not commit phase");
        uint32 id = _id();
        require(commitOf[r][id] == bytes32(0), "already committed");
        bytes32 sh = keccak256(sig);
        commitOf[r][id] = c;
        sigHashOf[r][id] = sh;
        rCommitOf[r][id] = rCommit;
        emit Committed(r, id, c, sh);
    }

    function closeCommit(uint32 r) external onlyCoordinator {
        require(rounds[r].phase == Phase.Commit, "not commit phase");
        rounds[r].phase = Phase.Reveal;
    }

    /// @notice Reveal binds the off-chain update (by its hash) to the signed commitment. A mismatch reverts,
    ///         so equivocation cannot be posted; failing to reveal is slashable by anyone.
    function reveal(uint32 r, bytes32 updateHash, bytes16 salt, bytes32 share) external {
        require(rounds[r].phase == Phase.Reveal, "not reveal phase");
        uint32 id = _id();
        require(updateHashOf[r][id] == bytes32(0), "already revealed");
        require(sha256(abi.encodePacked(r, id, salt, updateHash)) == commitOf[r][id], "commitment mismatch");
        require(keccak256(abi.encodePacked(share)) == rCommitOf[r][id], "beacon share mismatch");
        updateHashOf[r][id] = updateHash;
        beaconAcc[r] ^= share;
        emit Revealed(r, id, updateHash);
    }

    /// @notice Ends the reveal phase. From here on the XOR of the revealed shares is fixed, and the beacon will use the
    ///         hash of a block that does not exist yet, so no tenant can see it when deciding whether to reveal.
    function closeReveal(uint32 r) external onlyCoordinator {
        require(rounds[r].phase == Phase.Reveal, "not reveal phase");
        rounds[r].phase = Phase.Drawing;
        rounds[r].target = uint64(block.number) + BEACON_DELAY;
        emit RevealClosed(r, beaconAcc[r], rounds[r].target);
    }

    /// @notice Anyone can fix the beacon once the target block exists. blockhash is available for 256 blocks; a round whose
    ///         beacon is not fixed in time is void (it cannot be re-drawn from a later block, which would allow re-rolling).
    function finalizeBeacon(uint32 r) external {
        Round storage rd = rounds[r];
        require(rd.phase == Phase.Drawing, "not drawing");
        require(block.number > rd.target, "beacon block not produced yet");
        if (block.number > uint256(rd.target) + 256) {
            rd.phase = Phase.Void;
            emit RoundVoided(r);
            return;
        }
        bytes32 b = keccak256(abi.encodePacked(beaconAcc[r], blockhash(rd.target), r));
        rd.beacon = b;
        rd.phase = Phase.Closed;
        emit BeaconFinal(r, b);
    }

    /// @notice Committee posts the result of off-chain validation and aggregation: hash of the new global model,
    ///         root of the reputation vector (the vector itself is in the event-free calldata), accepted bitmap.
    function attest(uint32 r, bytes32 aggHash, bytes32 repRoot, uint256 accepted) external onlyCoordinator {
        require(rounds[r].phase == Phase.Closed, "not closed");
        require(rounds[r].aggHash == bytes32(0), "attested");
        rounds[r].aggHash = aggHash;
        rounds[r].repRoot = repRoot;
        rounds[r].accepted = accepted;
        emit Attested(r, aggHash, repRoot, accepted);
    }

    // ------------------------------------------------------------------ accountability
    function _slash(uint32 r, uint32 id, uint8 reason) internal {
        Tenant storage t = tenants[id];
        uint96 amt = t.bond / 2;
        t.bond -= amt;
        if (t.bond <= bondAmount / 4) t.banned = true;   // two slashes ban, as in the simulated rules
        (bool ok, ) = coordinator.call{value: amt}("");
        require(ok, "transfer");
        emit Slashed(r, id, amt, reason);
    }

    /// @notice On-chain provable: committed but never revealed. Anyone can call once the reveal phase has closed.
    function slashNoReveal(uint32 r, uint32 id) external {
        require(uint8(rounds[r].phase) >= uint8(Phase.Drawing), "reveals not closed");
        require(commitOf[r][id] != bytes32(0) && updateHashOf[r][id] == bytes32(0), "not slashable");
        commitOf[r][id] = bytes32(0); // one slash per offence
        _slash(r, id, 1);
    }

    /// @notice Off-chain provable: the revealed update failed validation on the revealed probes. The evidence
    ///         hash binds (revealed update hash, probe indices with Merkle proofs, computed scores). The tenant
    ///         may challenge inside the dispute window; an unchallenged proposal executes.
    function proposeSlash(uint32 r, uint32 id, bytes32 evidence) external onlyCoordinator {
        require(updateHashOf[r][id] != bytes32(0), "nothing revealed");
        PendingSlash storage p = pending[r][id];
        require(p.deadline == 0, "pending");
        p.evidence = evidence;
        p.deadline = uint64(block.number) + disputeWindow;
        emit SlashProposed(r, id, evidence, p.deadline);
    }

    function challenge(uint32 r) external {
        uint32 id = _id();
        PendingSlash storage p = pending[r][id];
        require(p.deadline != 0 && block.number <= p.deadline && !p.challenged, "cannot challenge");
        p.challenged = true;          // resolution: committee re-runs the public, deterministic check
        emit Challenged(r, id);
    }

    function executeSlash(uint32 r, uint32 id) external {
        PendingSlash storage p = pending[r][id];
        require(p.deadline != 0 && block.number > p.deadline && !p.challenged && !p.done, "not executable");
        p.done = true;
        _slash(r, id, 2);
    }

    // ------------------------------------------------------------------ sealed probes
    /// @notice Checks that a revealed probe sample belongs to the pool committed at setup (sha256 Merkle tree).
    function verifyProbe(bytes32 leaf, bytes32[] calldata proof, uint256 index) external view returns (bool) {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            h = (index & 1) == 0 ? sha256(abi.encodePacked(h, proof[i])) : sha256(abi.encodePacked(proof[i], h));
            index >>= 1;
        }
        return h == probeRoot;
    }

    function nTenants() external view returns (uint256) { return tenants.length; }
}
