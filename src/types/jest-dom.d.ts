import '@testing-library/jest-dom';

declare global {
  namespace jest {
    interface Matchers<R> {
      toBeInTheDocument(): R;
      toHaveValue(value: string | number | string[]): R;
      toBeDisabled(): R;
      toHaveTextContent(text: string): R;
    }
  }
}

// Empty export to make this a module
export {}; 