/**
 * Hook for managing chat state -- message history, sending queries,
 * and loading indicators.
 */

import { useCallback, useState } from "react";
import { sendChatMessage } from "../api/client";
import type { Message } from "../api/types";

let nextId = 0;
function uid(): string {
  nextId += 1;
  return `msg-${nextId}-${Date.now()}`;
}

export interface UseChatReturn {
  messages: Message[];
  loading: boolean;
  error: string | null;
  send: (query: string, collection?: string) => Promise<void>;
  clear: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
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
      setError(null);

      try {
        const res = await sendChatMessage({ query, collection });

        const assistantMsg: Message = {
          id: uid(),
          role: "assistant",
          content: res.answer,
          route: res.route,
          citations: res.citations,
          trace: res.trace,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "An unexpected error occurred";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const clear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, loading, error, send, clear };
}
