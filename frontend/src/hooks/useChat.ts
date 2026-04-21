/**
 * Hook for managing chat state -- message history, sending queries,
 * and live streaming progress via /api/chat/stream.
 */

import { useCallback, useState } from "react";
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
  send: (query: string, collection?: string) => Promise<void>;
  clear: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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

      try {
        for await (const event of sendChatStream({ query, collection })) {
          if (event.event === "node_start") {
            setActiveNode(event.node);
          } else if (event.event === "node_end") {
            setActiveNode(null);
          } else if (event.event === "result") {
            const res = event.data;
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
        const msg =
          err instanceof Error ? err.message : "An unexpected error occurred";
        setError(msg);
      } finally {
        setLoading(false);
        setActiveNode(null);
      }
    },
    [],
  );

  const clear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, loading, activeNode, error, send, clear };
}
