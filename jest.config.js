module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testPathIgnorePatterns: [
    '<rootDir>/node_modules/', 
    '<rootDir>/.next/',
    '<rootDir>/src/__tests__/utils/test-utils.tsx',
    '<rootDir>/scripts/test.js'
  ],
  testRegex: '(/__tests__/.*|\\.(test|spec))\\.[jt]sx?$',
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
    '^.+\\.node_modules\\.(js|jsx|ts|tsx)$': '<rootDir>/src/__mocks__/transformers/babelTransformer.js',
  },
  transformIgnorePatterns: [
    // Transform ESM modules that need to be processed
    '/node_modules/(?!react-markdown|micromark|decode-named-character-reference|character-entities|property-information|space-separated-tokens|comma-separated-tokens|hast-util-whitespace|remark-parse|remark-rehype|unified|bail|is-plain-obj|trough|vfile|vfile-message|unist-util-stringify-position|mdast-util-from-markdown|mdast-util-to-string|micromark-util-.*|mdast-util-.*|unist-util-.*|hast-util-.*|hastscript|web-namespaces)/',
  ],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    // Mock SVG files
    '\\.svg$': '<rootDir>/src/__mocks__/svg.js',
  },
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/_*.{js,jsx,ts,tsx}',
    '!src/**/index.{js,jsx,ts,tsx}',
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname',
  ],
  // Increase timeouts and isolate tests for more reliable test runs
  testTimeout: 20000,
  maxWorkers: '50%',
  bail: 1,
  // Improve error reporting
  verbose: true,
  forceExit: true,
}; 