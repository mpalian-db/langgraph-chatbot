/**
 * ConversationListSidebar -- left-rail navigator over every persisted
 * conversation. Click a row to load it as the active conversation; click
 * the pencil icon to rename. The pencil-on-hover affordance keeps the
 * primary click target (the row) free for the most common action while
 * keeping the rename action discoverable.
 *
 * Sourced from `GET /api/conversations`, refreshed on conversationId or
 * messages.length change so the row metadata (turn_count, has_summary,
 * title) stays current as the user chats.
 */

import { useState } from "react";
import type { ConversationOverviewOut } from "../api/types";

interface Props {
  overviews: ConversationOverviewOut[];
  activeConversationId: string | null;
  onSelect: (id: string) => void;
  /** Called when the user submits a rename. The sidebar handles the
   *  edit-mode UI; the parent owns the network call + cache refresh. */
  onRename?: (id: string, newTitle: string) => void;
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

interface RowProps {
  conv: ConversationOverviewOut;
  isActive: boolean;
  onSelect: (id: string) => void;
  onRename?: (id: string, newTitle: string) => void;
}

function ConversationRow({ conv, isActive, onSelect, onRename }: RowProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const startEdit = () => {
    setDraft(conv.title ?? "");
    setIsEditing(true);
  };

  const cancelEdit = () => {
    setIsEditing(false);
    setDraft("");
  };

  const commitEdit = () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === conv.title) {
      cancelEdit();
      return;
    }
    onRename?.(conv.conversation_id, trimmed);
    setIsEditing(false);
    setDraft("");
  };

  return (
    <li
      // The row itself is a button: click to load. The pencil button and
      // input both stopPropagation so they don't double-fire onSelect.
      role="button"
      tabIndex={0}
      onClick={() => !isEditing && onSelect(conv.conversation_id)}
      onKeyDown={(e) => {
        if (isEditing) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(conv.conversation_id);
        }
      }}
      className={`group cursor-pointer border-b border-gray-800 px-3 py-2 text-xs transition-colors ${
        isActive
          ? "bg-indigo-900/40 text-indigo-200"
          : "text-gray-300 hover:bg-gray-800"
      }`}
      aria-current={isActive ? "true" : undefined}
    >
      <div className="flex items-center justify-between gap-2">
        {isEditing ? (
          <input
            autoFocus
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") {
                e.preventDefault();
                commitEdit();
              } else if (e.key === "Escape") {
                e.preventDefault();
                cancelEdit();
              }
            }}
            onBlur={cancelEdit}
            maxLength={200}
            aria-label="Conversation title"
            className="min-w-0 flex-1 rounded border border-indigo-500 bg-gray-900 px-1 py-0.5 font-medium text-gray-100 outline-none"
          />
        ) : (
          <>
            <span
              className="truncate font-medium text-gray-200"
              title={conv.title ?? conv.conversation_id}
            >
              {conv.title ?? (
                <code className="font-mono text-[11px] text-gray-400">
                  {conv.conversation_id.slice(0, 8)}
                </code>
              )}
            </span>
            <div className="flex items-center gap-1">
              {onRename && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    startEdit();
                  }}
                  className="hidden text-gray-500 transition-colors hover:text-indigo-300 group-hover:inline"
                  aria-label="Rename conversation"
                  title="Rename"
                >
                  ✎
                </button>
              )}
              <span className="shrink-0 text-[10px] text-gray-500 tabular-nums">
                {formatTimestamp(conv.last_updated_at)}
              </span>
            </div>
          </>
        )}
      </div>
      {!isEditing && (
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
      )}
    </li>
  );
}

export default function ConversationListSidebar({
  overviews,
  activeConversationId,
  onSelect,
  onRename,
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
          {overviews.map((conv) => (
            <ConversationRow
              key={conv.conversation_id}
              conv={conv}
              isActive={conv.conversation_id === activeConversationId}
              onSelect={onSelect}
              onRename={onRename}
            />
          ))}
        </ul>
      )}
    </aside>
  );
}
