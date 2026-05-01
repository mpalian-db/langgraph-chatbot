import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useConversationList } from "../hooks/useConversationList";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("useConversationList", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns an empty list initially and fetches the overview on mount", async () => {
    const data = [
      {
        conversation_id: "conv-A",
        turn_count: 4,
        has_summary: false,
        last_updated_at: 1234567890,
      },
    ];
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(data)));

    const { result } = renderHook(() => useConversationList(0));

    await waitFor(() => expect(result.current.overviews.length).toBe(1));
    expect(result.current.overviews[0].conversation_id).toBe("conv-A");
  });

  it("refetches when refetchKey changes", async () => {
    const fetchMock = vi.fn(async () => jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    const { rerender } = renderHook(
      ({ key }: { key: number }) => useConversationList(key),
      { initialProps: { key: 0 } },
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    rerender({ key: 1 });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("imperative refetch() bumps the fetch count without external key change", async () => {
    const fetchMock = vi.fn(async () => jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useConversationList("static"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    act(() => result.current.refetch());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("silently degrades on fetch failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("err", { status: 500 })),
    );

    const { result } = renderHook(() => useConversationList(0));

    await new Promise((r) => setTimeout(r, 10));
    expect(result.current.overviews).toEqual([]);
  });

  it("forwards an AbortSignal so a stale fetch can be cancelled", async () => {
    let receivedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        receivedSignal = init?.signal ?? undefined;
        return jsonResponse([]);
      }),
    );

    renderHook(() => useConversationList(0));

    await waitFor(() => expect(receivedSignal).toBeDefined());
    expect(receivedSignal).toBeInstanceOf(AbortSignal);
  });
});
