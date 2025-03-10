#!/usr/bin/env node

/**
 * Custom test runner that skips Babel transformation for node_modules
 * This helps avoid issues with private class methods in dependencies
 */

const jest = require('jest');
const path = require('path');

// Pass all arguments to Jest except for our custom ones
const jestArgs = process.argv.slice(2);

// Add --no-transform to skip transformation of node_modules
if (!jestArgs.includes('--no-transform')) {
  jestArgs.push('--no-transform');
}

// Optional: Add other helpful default flags
if (!jestArgs.includes('--colors')) {
  jestArgs.push('--colors');
}

// Add --forceExit to ensure Jest exits properly
if (!jestArgs.includes('--forceExit')) {
  jestArgs.push('--forceExit');
}

// Run Jest synchronously to avoid async issues with environment teardown
const runResult = jest.runCLI({
  // Add any additional config options here
  forceExit: true,
  // Pass other arguments through
  _: jestArgs.filter(arg => !arg.startsWith('--')),
}, [process.cwd()]);

// Jest will handle the process.exit internally with forceExit: true 