/**
 * ConversationListSidebar -- left-rail navigator over every persisted
 * conversation. Click a row to load it as the active conversation; the
 * next message attaches to the existing server-side history.
 *
 * Sourced from `GET /api/conversations`, refreshed on conversationId or
 * messages.length change so the row metadata (turn_count, has_summary)
 * stays current as the user chats.
 */

import type { ConversationOverviewOut } from "../api/types";

interface Props {
  overviews: ConversationOverviewOut[];
  activeConversationId: string | null;
  onSelect: (id: string) => void;
}

function formatTimestamp(epochSeconds: number | null): string {
  if (epochSeconds == null) return "—";
  const ms = epochSeconds * 1000;
  const date = new Date(ms);
  // Same-day: HH:MM. Older: short date.
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  if (sameDay) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function ConversationListSidebar({
  overviews,
  activeConversationId,
  onSelect,
}: Props) {
  return (
    <aside
      className="flex w-56 flex-col border-r border-gray-700"
      aria-label="Conversation list"
    >
      <div className="border-b border-gray-700 px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
          Conversations
        </h2>
      </div>
      {overviews.length === 0 ? (
        <p className="px-4 py-3 text-xs italic text-gray-500">
          No conversations yet. Send a message to start one.
        </p>
      ) : (
        <ul className="flex-1 overflow-y-auto">
          {overviews.map((conv) => {
            const isActive = conv.conversation_id === activeConversationId;
            return (
              <li
                key={conv.conversation_id}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(conv.conversation_id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(conv.conversation_id);
                  }
                }}
                className={`cursor-pointer border-b border-gray-800 px-3 py-2 text-xs transition-colors ${
                  isActive
                    ? "bg-indigo-900/40 text-indigo-200"
                    : "text-gray-300 hover:bg-gray-800"
                }`}
                aria-current={isActive ? "true" : undefined}
              >
                <div className="flex items-center justify-between gap-2">
                  <code className="font-mono text-[11px] text-gray-400">
                    {conv.conversation_id.slice(0, 8)}
                  </code>
                  <span className="text-[10px] text-gray-500 tabular-nums">
                    {formatTimestamp(conv.last_updated_at)}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-2 text-[10px] text-gray-500">
                  <span>
                    {conv.turn_count} turn{conv.turn_count !== 1 && "s"}
                  </span>
                  {conv.has_summary && (
                    <span className="rounded bg-indigo-900/40 px-1 py-px text-indigo-300">
                      summarised
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
