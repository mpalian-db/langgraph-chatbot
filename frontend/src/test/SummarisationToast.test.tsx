import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import SummarisationToast from "../components/SummarisationToast";

describe("SummarisationToast", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders nothing when triggerKey is 0 (initial state)", () => {
    const { container } = render(<SummarisationToast triggerKey={0} />);
    expect(container.firstChild).toBeNull();
  });

  it("appears when triggerKey becomes non-zero", () => {
    const { rerender } = render(<SummarisationToast triggerKey={0} />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    rerender(<SummarisationToast triggerKey={1} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText(/conversation compressed/i)).toBeInTheDocument();
  });

  it("auto-dismisses after the configured duration", () => {
    render(<SummarisationToast triggerKey={1} durationMs={3000} />);
    expect(screen.getByRole("status")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(3001);
    });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("re-shows on a subsequent triggerKey bump after auto-dismissal", () => {
    const { rerender } = render(
      <SummarisationToast triggerKey={1} durationMs={1000} />,
    );
    act(() => {
      vi.advanceTimersByTime(1500);
    });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    rerender(<SummarisationToast triggerKey={2} durationMs={1000} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("can be manually dismissed before the auto-dismiss fires", async () => {
    // userEvent needs a real timer; switch back for this test.
    vi.useRealTimers();
    const user = userEvent.setup();

    render(<SummarisationToast triggerKey={1} durationMs={60_000} />);
    expect(screen.getByRole("status")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
