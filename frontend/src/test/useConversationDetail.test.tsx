import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useConversationDetail } from "../hooks/useConversationDetail";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("useConversationDetail", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns null detail before any conversation_id is set", () => {
    vi.stubGlobal("fetch", vi.fn());
    const { result } = renderHook(() => useConversationDetail(null, 0));
    expect(result.current.detail).toBeNull();
  });

  it("fetches detail when a conversation_id is provided", async () => {
    const data = {
      conversation_id: "abc",
      summary: "the rolling summary",
      turns: [{ role: "user", content: "hi" }],
    };
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(data)));

    const { result } = renderHook(() => useConversationDetail("abc", 0));

    await waitFor(() => expect(result.current.detail).not.toBeNull());
    expect(result.current.detail).toEqual(data);
  });

  it("refetches when refetchKey changes", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      // Echo back the abort signal presence so we can verify it was passed.
      void init?.signal;
      return jsonResponse({ conversation_id: "abc", summary: null, turns: [] });
    });
    vi.stubGlobal("fetch", fetchMock);

    const { rerender } = renderHook(
      ({ key }: { key: number }) => useConversationDetail("abc", key),
      { initialProps: { key: 0 } },
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    rerender({ key: 1 });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("clears detail when conversation_id becomes null", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({ conversation_id: "abc", summary: "x", turns: [] }),
      ),
    );

    const { result, rerender } = renderHook(
      ({ id }: { id: string | null }) => useConversationDetail(id, 0),
      { initialProps: { id: "abc" as string | null } },
    );
    await waitFor(() => expect(result.current.detail).not.toBeNull());

    rerender({ id: null });
    expect(result.current.detail).toBeNull();
  });

  it("silently degrades on fetch failure (no thrown error)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("err", { status: 500 })),
    );

    const { result } = renderHook(() => useConversationDetail("abc", 0));

    // Wait a tick for the rejected promise to settle. The hook must not
    // throw and must leave detail as null.
    await new Promise((r) => setTimeout(r, 10));
    expect(result.current.detail).toBeNull();
  });

  it("forwards an AbortSignal to the fetch call", async () => {
    let receivedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        receivedSignal = init?.signal ?? undefined;
        return jsonResponse({ conversation_id: "abc", summary: null, turns: [] });
      }),
    );

    renderHook(() => useConversationDetail("abc", 0));
    await waitFor(() => expect(receivedSignal).toBeDefined());
    expect(receivedSignal).toBeInstanceOf(AbortSignal);
  });
});
