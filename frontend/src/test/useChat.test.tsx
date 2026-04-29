/**
 * Tests for the useChat hook -- focusing on the conversation memory
 * round-trip: capturing the server-supplied id on first send, echoing it
 * on subsequent sends, and resetting on clear().
 */

import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useChat } from "../hooks/useChat";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a Response whose body is an NDJSON stream of pre-canned events. */
function ndjsonResponse(events: object[]): Response {
  const body = events.map((e) => JSON.stringify(e)).join("\n") + "\n";
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson" },
  });
}

/** Capture every fetch invocation and return the raw RequestInit body. */
type FetchCall = { url: string; body: unknown };

function setupFetchCapture(events: object[]): {
  calls: FetchCall[];
  fetchMock: ReturnType<typeof vi.fn>;
} {
  const calls: FetchCall[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({
      url,
      body: init?.body ? JSON.parse(init.body as string) : undefined,
    });
    return ndjsonResponse(events);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

function resultEvent(answer: string, conversationId: string) {
  return {
    event: "result",
    data: {
      answer,
      conversation_id: conversationId,
      route: "chat",
      citations: [],
      trace: [],
    },
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useChat conversation memory", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("captures server-supplied conversation_id on first send", async () => {
    setupFetchCapture([resultEvent("hi there", "conv-abc")]);

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("hello");
    });

    await waitFor(() => {
      expect(result.current.conversationId).toBe("conv-abc");
    });
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].content).toBe("hi there");
  });

  it("echoes the conversation_id on subsequent sends", async () => {
    const { calls } = setupFetchCapture([resultEvent("first", "conv-xyz")]);
    const { result } = renderHook(() => useChat());

    // First turn: no id sent, server returns conv-xyz.
    await act(async () => {
      await result.current.send("first message");
    });
    await waitFor(() =>
      expect(result.current.conversationId).toBe("conv-xyz"),
    );

    // Second turn: hook should pass conv-xyz back to the server.
    await act(async () => {
      await result.current.send("second message");
    });

    expect(calls).toHaveLength(2);
    expect((calls[0].body as { conversation_id?: string }).conversation_id).toBeUndefined();
    expect((calls[1].body as { conversation_id?: string }).conversation_id).toBe("conv-xyz");
  });

  it("clear() resets conversation_id and messages", async () => {
    setupFetchCapture([resultEvent("hi", "conv-1")]);
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.send("hello");
    });
    await waitFor(() => expect(result.current.conversationId).toBe("conv-1"));

    act(() => {
      result.current.clear();
    });

    expect(result.current.conversationId).toBeNull();
    expect(result.current.messages).toEqual([]);
  });

  it("clear() aborts a stalled reader (post-response hang)", async () => {
    // Backend returns 200 + a streaming body, but the body never produces
    // any chunks. Without abort propagation, useChat would hang in the
    // for-await loop indefinitely. The test pins that abort unwedges the
    // reader by erroring the stream from inside pull().
    let pullCalled = false;

    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init?: RequestInit) => {
        const stream = new ReadableStream<Uint8Array>({
          pull(controller) {
            pullCalled = true;
            // Hold the stream open until the signal aborts; on abort,
            // surface the error to the consumer's reader.
            return new Promise<void>((_resolve, reject) => {
              init?.signal?.addEventListener("abort", () => {
                const err = new DOMException("aborted", "AbortError");
                controller.error(err);
                reject(err);
              });
            });
          },
        });
        return Promise.resolve(
          new Response(stream, {
            status: 200,
            headers: { "Content-Type": "application/x-ndjson" },
          }),
        );
      }),
    );

    const { result } = renderHook(() => useChat());

    let sendPromise: Promise<void> | undefined;
    act(() => {
      sendPromise = result.current.send("never produces chunks");
    });
    await waitFor(() => expect(pullCalled).toBe(true));
    expect(result.current.loading).toBe(true);

    act(() => result.current.clear());

    await sendPromise;

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("clear() aborts an in-flight stream so the user is never trapped", async () => {
    // Build a stream that never resolves -- simulates a hung backend.
    const aborts: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, init?: RequestInit) => {
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            aborts.push(url);
            reject(new DOMException("aborted", "AbortError"));
          });
          // Never resolves on its own.
        });
      }),
    );

    const { result } = renderHook(() => useChat());

    // Fire send -- it will hang on fetch.
    let sendPromise: Promise<void> | undefined;
    act(() => {
      sendPromise = result.current.send("hangs forever");
    });
    await waitFor(() => expect(result.current.loading).toBe(true));

    // User clicks "New conversation" while loading. The abort must fire and
    // loading must drop to false; the send promise resolves cleanly.
    act(() => {
      result.current.clear();
    });

    await sendPromise;

    expect(aborts).toEqual(["/api/chat/stream"]);
    expect(result.current.loading).toBe(false);
    expect(result.current.conversationId).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("after clear(), the next send starts a new conversation", async () => {
    // First call returns conv-A; after clear, second call returns conv-B.
    const responses = [
      ndjsonResponse([resultEvent("a", "conv-A")]),
      ndjsonResponse([resultEvent("b", "conv-B")]),
    ];
    let callIndex = 0;
    const calls: FetchCall[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({
          url,
          body: init?.body ? JSON.parse(init.body as string) : undefined,
        });
        return responses[callIndex++];
      }),
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.send("first");
    });
    await waitFor(() => expect(result.current.conversationId).toBe("conv-A"));

    act(() => result.current.clear());
    await act(async () => {
      await result.current.send("second");
    });

    // Second call must NOT include conv-A -- clear() reset the id.
    expect(calls).toHaveLength(2);
    expect((calls[1].body as { conversation_id?: string }).conversation_id).toBeUndefined();
    await waitFor(() => expect(result.current.conversationId).toBe("conv-B"));
  });
});
