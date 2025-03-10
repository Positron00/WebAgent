# Testing Guide

This document provides an overview of the testing framework used in the WebAgent project, including setup, best practices, and troubleshooting tips.

## Testing Setup

The project uses Jest for testing React components and utilities, with the following key configurations:

- **Jest**: Test runner and assertion library
- **React Testing Library**: For testing React components
- **babel-jest**: For transpiling modern JavaScript features during testing
- **jest-environment-jsdom**: Provides a DOM-like environment for tests

## Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run a specific test file
npm run test:single -- path/to/test.tsx

# Run the ApiClient tests specifically
npm run test:apiClient
```

## Test Structure

### Unit Tests

Unit tests are located in the `src/__tests__` directory and follow this structure:

```
src/
├── __tests__/
│   ├── utils/              # Tests for utility functions
│   │   ├── apiClient.test.ts
│   │   └── ...
│   ├── components/         # Tests for React components
│   │   ├── Button.test.tsx
│   │   └── ...
│   └── ...
```

### Component Tests

Component tests typically verify:

1. Rendering
2. User interactions
3. State changes
4. Prop handling

Example:

```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '../components/Button';

describe('Button Component', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
```

## Mocking

### Browser APIs

We mock various browser APIs to ensure tests run consistently:

- **localStorage/sessionStorage**: For testing persistent data
- **matchMedia**: For testing responsive components
- **SpeechRecognition**: For voice input features
- **IntersectionObserver**: For lazy loading and scroll detection
- **ResizeObserver**: For size change detection
- **URL.createObjectURL**: For file upload previews

### React Contexts

Context providers are mocked in `src/__tests__/utils/test-utils.tsx` to provide consistent test environments:

```tsx
import { render } from '@testing-library/react';
import { AppProvider } from '@/contexts/AppContext';
import { ChatProvider } from '@/contexts/ChatContext';

const customRender = (ui, options) => 
  render(ui, { wrapper: ({ children }) => (
    <AppProvider>
      <ChatProvider>
        {children}
      </ChatProvider>
    </AppProvider>
  ), ...options });

export * from '@testing-library/react';
export { customRender as render };
```

## Testing Best Practices

1. **Test functionality, not implementation**: Focus on user behavior rather than implementation details.
2. **Keep tests simple**: Each test should verify one specific behavior.
3. **Use meaningful assertions**: Make it clear what's being tested and why.
4. **Mock external dependencies**: Isolate the code being tested from external systems.
5. **Use data-testid attributes**: When there's no accessible text to select elements.
6. **Test edge cases**: Including error states and boundary conditions.
7. **Avoid testing library internals**: Focus on your own code.

## Troubleshooting

### Common Issues

1. **Test times out**: Increase the timeout in Jest configuration or check for unresolved promises.
2. **DOM-related errors**: Make sure you're using `jest-environment-jsdom` and mocking browser APIs.
3. **Context errors**: Ensure components are wrapped with the necessary providers.
4. **Module import errors**: Check transformIgnorePatterns and moduleNameMapper in Jest config.
5. **Babel transpilation issues**: Verify babel.config.js includes the necessary plugins.

### ESM Module Issues

Some libraries like `react-markdown` use ESM modules which can cause issues with Jest. We handle this by:

1. Using appropriate transformIgnorePatterns to process these modules
2. Providing a custom Babel transformer for node_modules
3. Using a custom test runner script to skip transformation for problematic modules

## Continuous Integration

The test suite runs in CI on every pull request and commit to main. The pipeline will fail if:

1. Any tests fail
2. Test coverage drops below the thresholds
3. There are linting errors

## Further Reading

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro)
- [Testing React Applications](https://reactjs.org/docs/testing.html) 