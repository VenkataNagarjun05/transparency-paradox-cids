/** Local EVM for gas and latency measurements. 120 funded accounts so 100 tenants + committee fit. */
module.exports = {
  solidity: { version: "0.8.24", settings: { optimizer: { enabled: true, runs: 200 }, evmVersion: "cancun" } },
  networks: {
    hardhat: { hardfork: "cancun", chainId: 31337, blockGasLimit: 30_000_000,
               accounts: { count: 120, accountsBalance: "1000000000000000000000000" } },
  },
};
