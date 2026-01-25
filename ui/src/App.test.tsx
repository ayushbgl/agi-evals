import { test, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

test('renders game arena home page', () => {
  const { getByText } = render(<App />);
  const linkElement = getByText(/Game Arena/i);
  expect(linkElement).toBeInTheDocument();
});
