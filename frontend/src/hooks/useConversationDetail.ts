/**
 * Hook for loading a conversation's persisted detail from the backend
 * introspection endpoint.
 *
 * Refetches whenever `conversationId` or `refetchKey` changes -- the
 * caller passes their message-count or any other monotonically-changing
 * value as `refetchKey` to keep the detail fresh as the chat progresses.
 *
 * Failures are silent: this is a debug surface, so a transient network
 * problem shouldn't break the chat UI. The hook simply doesn't update
 * state when the fetch fails.
 *
 * Cancellation: each fetch carries an AbortSignal; a stale request is
 * aborted when the dependencies change so an out-of-order response can't
 * clobber a fresher one.
 */

import { useEffect, useState } from "react";
import { getConversation } from "../api/client";
import type { ConversationDetailOut } from "../api/types";

export interface UseConversationDetailReturn {
  detail: ConversationDetailOut | null;
}

export function useConversationDetail(
  conversationId: string | null,
  refetchKey: number,
): UseConversationDetailReturn {
  const [detail, setDetail] = useState<ConversationDetailOut | null>(null);

  useEffect(() => {
    if (!conversationId) {
      setDetail(null);
      return;
    }

    const controller = new AbortController();
    let cancelled = false;

    getConversation(conversationId, controller.signal)
      .then((value) => {
        if (!cancelled) setDetail(value);
      })
      .catch(() => {
        // Silent degrade -- a debug surface that occasionally fails
        // shouldn't surface a user-facing error.
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [conversationId, refetchKey]);

  return { detail };
}
