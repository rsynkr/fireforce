import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// jsdom does not implement canvas rendering; the grid effect handles a null context.
Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: vi.fn(() => null),
  writable: true,
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
