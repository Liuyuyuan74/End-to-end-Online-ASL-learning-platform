import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import InputFileUpload from './InputFileUpload'; 

// Mock fetch globally
beforeAll(() => {
  global.fetch = jest.fn();
  window.alert = jest.fn(); // Mock window.alert
  global.URL.createObjectURL = jest.fn(() => "https://example.com/object-url"); // Mock URL.createObjectURL
});

beforeEach(() => {
  fetch.mockClear();
});

test('file upload triggers fetch and updates state', async () => {
  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({ url: 'https://example.com/pre-signed-url' }),
  }).mockResolvedValueOnce({
    ok: true,
  });

  const file = new File(['dummy content'], 'testfile.png', { type: 'image/png' });
  render(<InputFileUpload />);
  const input = screen.getByLabelText(/upload file/i);

  fireEvent.change(input, { target: { files: [file] } });

  await waitFor(() => {
    expect(screen.getByText(/file uploaded: testfile.png/i)).toBeInTheDocument();
  });
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(window.alert).toHaveBeenCalledWith('Upload successful');
});
