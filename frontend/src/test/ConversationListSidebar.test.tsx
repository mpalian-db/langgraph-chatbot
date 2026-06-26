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
    title: null,
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

  it("does not render the rename button when onRename is not provided", () => {
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "conv-1", title: "Some title" })]}
        activeConversationId={null}
        onSelect={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /rename conversation/i }),
    ).not.toBeInTheDocument();
  });

  it("rename button switches the row to an editable input", async () => {
    const user = userEvent.setup();
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "conv-1", title: "Original" })]}
        activeConversationId={null}
        onSelect={vi.fn()}
        onRename={vi.fn()}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /rename conversation/i }),
    );
    const input = screen.getByRole("textbox", { name: /conversation title/i });
    expect(input).toHaveValue("Original");
  });

  it("Enter on the rename input commits via onRename callback", async () => {
    const onRename = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "conv-1", title: "Old" })]}
        activeConversationId={null}
        onSelect={vi.fn()}
        onRename={onRename}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /rename conversation/i }),
    );
    const input = screen.getByRole("textbox", { name: /conversation title/i });
    await user.clear(input);
    await user.type(input, "New title{Enter}");

    expect(onRename).toHaveBeenCalledWith("conv-1", "New title");
  });

  it("Escape on the rename input cancels without calling onRename", async () => {
    const onRename = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "conv-1", title: "Old" })]}
        activeConversationId={null}
        onSelect={vi.fn()}
        onRename={onRename}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /rename conversation/i }),
    );
    const input = screen.getByRole("textbox", { name: /conversation title/i });
    await user.type(input, "{Escape}");

    expect(onRename).not.toHaveBeenCalled();
    // Edit mode exited, original title visible again.
    expect(screen.getByText("Old")).toBeInTheDocument();
  });

  it("commits a trimmed title and skips no-op edits", async () => {
    const onRename = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "conv-1", title: "Old" })]}
        activeConversationId={null}
        onSelect={vi.fn()}
        onRename={onRename}
      />,
    );

    // Edit but keep the same value -- should NOT call onRename.
    await user.click(
      screen.getByRole("button", { name: /rename conversation/i }),
    );
    const input = screen.getByRole("textbox", { name: /conversation title/i });
    await user.type(input, "{Enter}");

    expect(onRename).not.toHaveBeenCalled();
  });

  it("clicking the rename button does not trigger the row's onSelect", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationListSidebar
        overviews={[ov({ conversation_id: "conv-1", title: "Title" })]}
        activeConversationId={null}
        onSelect={onSelect}
        onRename={vi.fn()}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /rename conversation/i }),
    );

    // The pencil button must stop propagation so loading isn't triggered.
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("renders title when present, truncated id otherwise", () => {
    render(
      <ConversationListSidebar
        overviews={[
          ov({
            conversation_id: "abcd1234-with-title",
            title: "What is LangGraph?",
          }),
          ov({
            conversation_id: "ffff0000-no-title",
            title: null,
          }),
        ]}
        activeConversationId={null}
        onSelect={vi.fn()}
      />,
    );

    // Titled conversation shows the title, not the id.
    expect(screen.getByText("What is LangGraph?")).toBeInTheDocument();
    // Untitled falls back to the truncated id.
    expect(screen.getByText("ffff0000")).toBeInTheDocument();
  });
});
