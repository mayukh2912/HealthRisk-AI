import "@testing-library/jest-dom/vitest";

// Recharts' ResponsiveContainer needs ResizeObserver in a DOM test environment.
class RO { observe() {} unobserve() {} disconnect() {} }
(globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver = RO;
