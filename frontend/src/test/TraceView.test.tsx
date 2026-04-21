import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import TraceView from "../components/TraceView";
import type { TraceEntryOut } from "../api/types";

const entry = (
  node: string,
  duration_ms: number,
  data: Record<string, unknown> = {},
): TraceEntryOut => ({ node, duration_ms, data });

describe("TraceView", () => {
  it("renders nothing when entries is empty", () => {
    const { container } = render(<TraceView entries={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows collapsed toggle with node count and total duration", () => {
    render(
      <TraceView
        entries={[entry("router", 12), entry("chat_agent", 345)]}
      />,
    );
    expect(screen.getByText(/Show trace/)).toBeInTheDocument();
    expect(screen.getByText(/2 nodes/)).toBeInTheDocument();
    expect(screen.getByText(/357ms total/)).toBeInTheDocument();
  });

  it("uses singular 'node' for a single entry", () => {
    render(<TraceView entries={[entry("router", 10)]} />);
    expect(screen.getByText(/1 node,/)).toBeInTheDocument();
  });

  it("expands to show node names on click", async () => {
    const user = userEvent.setup();
    render(
      <TraceView
        entries={[entry("router", 5), entry("retrieval", 120)]}
      />,
    );

    await user.click(screen.getByRole("button"));
    expect(screen.getByText("router")).toBeInTheDocument();
    expect(screen.getByText("retrieval")).toBeInTheDocument();
  });

  it("collapses again on second click", async () => {
    const user = userEvent.setup();
    render(<TraceView entries={[entry("router", 5)]} />);

    const btn = screen.getByRole("button");
    await user.click(btn);
    expect(screen.getByText("router")).toBeInTheDocument();

    await user.click(btn);
    expect(screen.queryByText("router")).not.toBeInTheDocument();
  });

  it("formats sub-second durations as ms", async () => {
    const user = userEvent.setup();
    render(<TraceView entries={[entry("router", 42)]} />);
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("42ms")).toBeInTheDocument();
  });

  it("formats durations >= 1s in seconds", async () => {
    const user = userEvent.setup();
    render(<TraceView entries={[entry("answer_generation", 2340)]} />);
    await user.click(screen.getByRole("button"));
    // Total duration label shows seconds
    expect(screen.getByText(/2\.34s total/)).toBeInTheDocument();
  });

  it("renders data fields when expanded", async () => {
    const user = userEvent.setup();
    render(
      <TraceView
        entries={[entry("verifier", 80, { outcome: "accept", score: 0.92 })]}
      />,
    );
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("outcome")).toBeInTheDocument();
    expect(screen.getByText("accept")).toBeInTheDocument();
    expect(screen.getByText("score")).toBeInTheDocument();
    expect(screen.getByText("0.92")).toBeInTheDocument();
  });

  it("renders boolean data values as yes/no", async () => {
    const user = userEvent.setup();
    render(
      <TraceView entries={[entry("router", 5, { reranked: true, cached: false })]} />,
    );
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("yes")).toBeInTheDocument();
    expect(screen.getByText("no")).toBeInTheDocument();
  });

  it("renders null/undefined data values as em dash", async () => {
    const user = userEvent.setup();
    render(
      <TraceView entries={[entry("router", 5, { missing: null })]} />,
    );
    await user.click(screen.getByRole("button"));
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
