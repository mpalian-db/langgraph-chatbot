import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import ConversationListSidebar from "../components/ConversationListSidebar";
import type { ConversationOverviewOut } from "../api/types";

function ov(
  overrides: Partial<ConversationOverviewOut> = {},
): ConversationOverviewOut {
  return {
    conversation_id: "conv-1",
    turn_count: 2,
    has_summary: false,
    last_updated_at: null,
    ...overrides,
  };
}

describe("ConversationListSidebar", () => {
  it("shows empty-state copy when there are no conversations", () => {
    render(
      <ConversationListSidebar
        overviews={[]}
        activeConversationId={null}
        onSelect={vi.fn()}
      />,
    );
    expect(
      screen.getByText(/no conversations yet/i),
    ).toBeInTheDocument();
  });

  it("renders one row per conversation with truncated id and turn count", () => {
    render(
      <ConversationListSidebar
        overviews={[
          ov({ conversation_id: "abcd1234-5678-90ab-cdef-1234567890ab", turn_count: 5 }),
          ov({ conversation_id: "ffff0000-aaaa-bbbb-cccc-dddddddddddd", turn_count: 1 }),
        ]}
        activeConversationId={null}
        onSelect={vi.fn()}
      />,
    );
    // First 8 chars of each id appear.
    expect(screen.getByText("abcd1234")).toBeInTheDocument();
    expect(screen.getByText("ffff0000")).toBeInTheDocument();
    expect(screen.getByText("5 turns")).toBeInTheDocument();
    expect(screen.getByText("1 turn")).toBeInTheDocument();
  });

  it("highlights the active conversation row via aria-current", () => {
    render(
      <ConversationListSidebar
        overviews={[
          ov({ conversation_id: "active-id" }),
          ov({ conversation_id: "other-id" }),
        ]}
        activeConversationId="active-id"
        onSelect={vi.fn()}
      />,
    );

    // Active row is marked aria-current; other isn't.
    const rows = screen.getAllByRole("button");
    const active = rows.find(
      (el) => el.getAttribute("aria-current") === "true",
    );
    expect(active).toBeDefined();
    expect(active?.textContent).toContain("active-i");
  });

  it("calls onSelect with the conversation id when a row is clicked", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "click-me" })]}
        activeConversationId={null}
        onSelect={onSelect}
      />,
    );

    await user.click(screen.getByRole("button", { current: undefined }));
    expect(onSelect).toHaveBeenCalledWith("click-me");
  });

  it("renders a 'summarised' badge only when has_summary is true", () => {
    render(
      <ConversationListSidebar
        overviews={[
          ov({ conversation_id: "with-sum", has_summary: true }),
          ov({ conversation_id: "without-sum", has_summary: false }),
        ]}
        activeConversationId={null}
        onSelect={vi.fn()}
      />,
    );

    const badges = screen.getAllByText(/summarised/i);
    expect(badges).toHaveLength(1);
  });
});
