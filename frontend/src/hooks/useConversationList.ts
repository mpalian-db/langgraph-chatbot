/**
 * Hook for the conversation-list sidebar.
 *
 * Fetches `GET /api/conversations` on mount and whenever `refetchKey`
 * changes -- the caller passes any monotonically-changing value (e.g.
 * the active conversation_id, or a turn count) to keep the sidebar
 * fresh as the user chats.
 *
 * Failures degrade silently: the sidebar is a debug-grade UX and a
 * transient network problem shouldn't break chat. Each fetch carries
 * an AbortSignal so a stale response can't clobber a fresher one.
 */

import { useCallback, useEffect, useState } from "react";
import { listConversations } from "../api/client";
import type { ConversationOverviewOut } from "../api/types";

export interface UseConversationListReturn {
  overviews: ConversationOverviewOut[];
  refetch: () => void;
}

export function useConversationList(refetchKey: unknown): UseConversationListReturn {
  const [overviews, setOverviews] = useState<ConversationOverviewOut[]>([]);
  const [forceKey, setForceKey] = useState(0);

  // Imperative refetch so the UI can refresh after a destructive action
  // (e.g. just-deleted a conversation -- re-pull the list immediately).
  const refetch = useCallback(() => setForceKey((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    listConversations(controller.signal)
      .then((data) => {
        if (!cancelled) setOverviews(data);
      })
      .catch(() => {
        // Silent: degrade rather than break the UI.
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [refetchKey, forceKey]);

  return { overviews, refetch };
}
