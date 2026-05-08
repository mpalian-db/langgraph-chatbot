import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import ConversationHistoryPanel from "../components/ConversationHistoryPanel";
import type { ConversationDetailOut } from "../api/types";

function detail(
  overrides: Partial<ConversationDetailOut> = {},
): ConversationDetailOut {
  return {
    conversation_id: "conv-1",
    title: null,
    summary: null,
    turns: [],
    ...overrides,
  };
}

describe("ConversationHistoryPanel", () => {
  it("renders the persisted-history header", () => {
    render(<ConversationHistoryPanel detail={detail()} />);
    expect(screen.getByText(/persisted history/i)).toBeInTheDocument();
  });

  it("shows summary section when summary is present", () => {
    render(
      <ConversationHistoryPanel
        detail={detail({ summary: "User asked about LangGraph.\nAssistant explained." })}
      />,
    );
    expect(screen.getByText(/rolling summary/i)).toBeInTheDocument();
    expect(
      screen.getByText(/User asked about LangGraph/i),
    ).toBeInTheDocument();
  });

  it("hides summary section when summary is null", () => {
    render(<ConversationHistoryPanel detail={detail({ summary: null })} />);
    expect(screen.queryByText(/rolling summary/i)).not.toBeInTheDocument();
  });

  it("renders all verbatim turns with their role labels", () => {
    render(
      <ConversationHistoryPanel
        detail={detail({
          turns: [
            { role: "user", content: "what is langgraph?" },
            { role: "assistant", content: "a stateful agent framework" },
          ],
        })}
      />,
    );
    expect(screen.getByText("what is langgraph?")).toBeInTheDocument();
    expect(screen.getByText("a stateful agent framework")).toBeInTheDocument();
    expect(screen.getByText("user")).toBeInTheDocument();
    expect(screen.getByText("assistant")).toBeInTheDocument();
  });

  it("shows a placeholder when no verbatim turns are persisted", () => {
    /** Summary-only conversation (all turns folded in) -- the panel should
     * still render usefully rather than show an empty list. */
    render(
      <ConversationHistoryPanel
        detail={detail({ summary: "all folded in", turns: [] })}
      />,
    );
    expect(
      screen.getByText(/no verbatim turns past the summary boundary/i),
    ).toBeInTheDocument();
  });

  it("renders both summary and turns when both are present", () => {
    render(
      <ConversationHistoryPanel
        detail={detail({
          summary: "Earlier context here.",
          turns: [{ role: "user", content: "follow-up question" }],
        })}
      />,
    );
    expect(screen.getByText(/Earlier context here/i)).toBeInTheDocument();
    expect(screen.getByText("follow-up question")).toBeInTheDocument();
  });

  it("hides the delete button when onDelete is not provided", () => {
    render(<ConversationHistoryPanel detail={detail()} />);
    expect(
      screen.queryByRole("button", { name: /delete this conversation/i }),
    ).not.toBeInTheDocument();
  });

  it("shows the delete button and fires onDelete on click", async () => {
    const onDelete = vi.fn();
    const user = userEvent.setup();
    render(<ConversationHistoryPanel detail={detail()} onDelete={onDelete} />);

    const btn = screen.getByRole("button", { name: /delete this conversation/i });
    await user.click(btn);

    expect(onDelete).toHaveBeenCalledTimes(1);
  });
});
