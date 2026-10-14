#!/usr/bin/env node

/**
 * Custom test runner that skips Babel transformation for node_modules
 * This helps avoid issues with private class methods in dependencies
 */

const jest = require("jest");
const path = require("path");

// Pass all arguments to Jest except for our custom ones
const jestArgs = process.argv.slice(2);

// Add --no-transform to skip transformation of node_modules
if (!jestArgs.includes("--no-transform")) {
  jestArgs.push("--no-transform");
}

// Optional: Add other helpful default flags
if (!jestArgs.includes("--colors")) {
  jestArgs.push("--colors");
}

// Run Jest synchronously to avoid async issues with environment teardown
jest.runCLI(
  {
    _: jestArgs.filter((arg) => !arg.startsWith("--")),
  },
  [process.cwd()],
);
