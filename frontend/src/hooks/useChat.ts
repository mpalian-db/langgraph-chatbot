/**
 * Hook for managing chat state -- message history, sending queries,
 * live streaming progress via /api/chat/stream, and conversation memory.
 *
 * Conversation memory: on the first send, the backend generates a uuid
 * and returns it on the response. We capture it and pass it back on
 * every subsequent send, so the backend can load prior turns and feed
 * them to the chat agent. `clear()` resets both the visible message
 * list and the conversation id, starting a genuinely fresh conversation
 * on the next send.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { sendChatStream } from "../api/client";
import type { Message } from "../api/types";

let nextId = 0;
function uid(): string {
  nextId += 1;
  return `msg-${nextId}-${Date.now()}`;
}

export interface UseChatReturn {
  messages: Message[];
  loading: boolean;
  activeNode: string | null;
  error: string | null;
  conversationId: string | null;
  send: (query: string, collection?: string) => Promise<void>;
  clear: () => void;
  /**
   * Switch the chat to an existing server-side conversation by id.
   *
   * Aborts any in-flight stream, resets the visible messages list, and
   * sets `conversationId` so the next send() carries history from that
   * conversation (loaded server-side via the memory port). The visible
   * thread starts empty -- the persisted history is available via the
   * introspection endpoint and the "View history" panel; the in-memory
   * `messages` array is the in-session view, not a hydrated mirror of
   * stored turns.
   */
  loadConversation: (id: string) => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  // Holds the AbortController for the in-flight stream so clear() can cancel
  // a hung request. Without this, a backend hang would leave loading=true
  // forever and the user would have no UI escape from the conversation.
  const inFlightRef = useRef<AbortController | null>(null);

  const send = useCallback(
    async (query: string, collection?: string) => {
      const userMsg: Message = {
        id: uid(),
        role: "user",
        content: query,
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      setActiveNode(null);
      setError(null);

      // Bind the controller for this send so clear() can abort it.
      const controller = new AbortController();
      inFlightRef.current = controller;

      try {
        for await (const event of sendChatStream(
          {
            query,
            collection,
            // Pass the current conversation id (if any) so the backend can
            // load history. On the first turn this is null and the backend
            // generates a fresh id, which we capture below.
            conversation_id: conversationId ?? undefined,
          },
          controller.signal,
        )) {
          if (event.event === "node_start") {
            setActiveNode(event.node);
          } else if (event.event === "node_end") {
            setActiveNode(null);
          } else if (event.event === "result") {
            const res = event.data;
            // Capture (or update) the server-supplied conversation id.
            // On follow-up turns this should equal the existing value.
            setConversationId(res.conversation_id);
            const assistantMsg: Message = {
              id: uid(),
              role: "assistant",
              content: res.answer,
              route: res.route,
              citations: res.citations,
              trace: res.trace,
            };
            setMessages((prev) => [...prev, assistantMsg]);
          }
        }
      } catch (err) {
        // AbortError from clear() is intentional cancellation -- swallow it
        // silently rather than surfacing as a user-visible error.
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        const msg =
          err instanceof Error ? err.message : "An unexpected error occurred";
        setError(msg);
      } finally {
        // Identity guard: if a clear() or a newer send() has already replaced
        // this controller, do nothing -- the new request owns loading state
        // now. Without this, an aborted older request could flip loading
        // back to false while a fresh send is mid-flight, leaving the user
        // with no spinner during a real request.
        if (inFlightRef.current === controller) {
          inFlightRef.current = null;
          setLoading(false);
          setActiveNode(null);
        }
      }
    },
    [conversationId],
  );

  // On unmount, abort any in-flight stream so a navigation away from the
  // chat view doesn't leave a zombie fetch reading bytes into nowhere.
  useEffect(() => {
    return () => {
      inFlightRef.current?.abort();
      inFlightRef.current = null;
    };
  }, []);

  const clear = useCallback(() => {
    // Abort any in-flight stream so the user is never trapped in a hung
    // request. Once aborted, the send() catch swallows AbortError silently;
    // no late state writes will land from this controller.
    inFlightRef.current?.abort();
    inFlightRef.current = null;
    // Reset both the visible thread and the conversation id so the next
    // send starts a genuinely new conversation server-side. Without
    // resetting the id, the server would keep accumulating history under
    // it, and the new "first" message would inherit prior context.
    setMessages([]);
    setConversationId(null);
    setError(null);
    setLoading(false);
    setActiveNode(null);
  }, []);

  const loadConversation = useCallback((id: string) => {
    // Same abort + reset shape as clear(), but instead of nulling the
    // conversation id we set it to the supplied one. The visible message
    // list still starts empty -- the persisted thread surfaces via the
    // history panel; this hook is the in-session view only.
    inFlightRef.current?.abort();
    inFlightRef.current = null;
    setMessages([]);
    setConversationId(id);
    setError(null);
    setLoading(false);
    setActiveNode(null);
  }, []);

  return {
    messages,
    loading,
    activeNode,
    error,
    conversationId,
    send,
    clear,
    loadConversation,
  };
}
