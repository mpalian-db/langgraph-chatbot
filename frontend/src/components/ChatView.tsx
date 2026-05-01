/**
 * Chat interface -- message input, message bubbles with route badges,
 * expandable citations, and trace details.
 */

import {
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useChat } from "../hooks/useChat";
import { useConversationDetail } from "../hooks/useConversationDetail";
import { useConversationList } from "../hooks/useConversationList";
import { deleteConversation, listCollections } from "../api/client";
import ConversationHistoryPanel from "./ConversationHistoryPanel";
import ConversationListSidebar from "./ConversationListSidebar";
import TraceView from "./TraceView";
import type { CitationOut, Message } from "../api/types";

// ---------------------------------------------------------------------------
// Route badge colour mapping
// ---------------------------------------------------------------------------

const ROUTE_COLOURS: Record<string, string> = {
  chat: "bg-emerald-700 text-emerald-100",
  rag: "bg-indigo-700 text-indigo-100",
  tool: "bg-amber-700 text-amber-100",
};

function RouteBadge({ route }: { route: string }) {
  const colour = ROUTE_COLOURS[route] ?? "bg-gray-700 text-gray-200";
  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${colour}`}
    >
      {route}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Citation list
// ---------------------------------------------------------------------------

function Citations({ items }: { items: CitationOut[] }) {
  const [expanded, setExpanded] = useState(false);

  if (items.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="text-xs font-medium text-indigo-400 hover:text-indigo-300"
      >
        {expanded ? "Hide" : "Show"} {items.length} citation
        {items.length !== 1 && "s"}
      </button>
      {expanded && (
        <ul className="mt-1 space-y-1 text-xs text-gray-400">
          {items.map((c, i) => (
            <li
              key={`${c.chunk_id}-${i}`}
              className="rounded border border-gray-700 bg-gray-800 p-2"
            >
              <span className="font-medium text-gray-300">
                [{c.collection}] {c.chunk_id}
              </span>
              <p className="mt-0.5 whitespace-pre-wrap text-gray-400">
                {c.text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single message bubble
// ---------------------------------------------------------------------------

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-lg px-4 py-3 ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-gray-800 text-gray-100"
        }`}
      >
        {/* Route badge for assistant messages */}
        {!isUser && msg.route && (
          <div className="mb-1">
            <RouteBadge route={msg.route} />
          </div>
        )}

        <p className="whitespace-pre-wrap text-sm">{msg.content}</p>

        {/* Citations and trace -- assistant only */}
        {!isUser && msg.citations && (
          <Citations items={msg.citations} />
        )}
        {!isUser && msg.trace && <TraceView entries={msg.trace} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatView
// ---------------------------------------------------------------------------

export default function ChatView() {
  const {
    messages,
    loading,
    activeNode,
    error,
    conversationId,
    send,
    clear,
    loadConversation,
  } = useChat();
  const [input, setInput] = useState("");
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollection, setSelectedCollection] = useState("");
  const [historyPanelOpen, setHistoryPanelOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch persisted conversation detail (summary + turns) for the header
  // chip and the optional history panel. Refetches whenever a new message
  // lands so the count stays in sync as the chat progresses.
  const { detail: conversationDetail } = useConversationDetail(
    conversationId,
    messages.length,
  );

  // Fetch the full conversation list for the left-rail sidebar. Refetched
  // on activeId or message-count change so newly-created conversations
  // and updated metadata (summarised badge, turn_count) appear without a
  // manual reload.
  const { overviews: conversationOverviews, refetch: refetchConversationList } =
    useConversationList(`${conversationId ?? ""}-${messages.length}`);

  // Fetch collections on mount so the user can scope queries.
  useEffect(() => {
    listCollections()
      .then(setCollections)
      .catch(() => {
        // Backend may not be running -- leave empty.
      });
  }, []);

  // Auto-scroll to newest message.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      const query = input.trim();
      if (!query || loading) return;
      setInput("");
      void send(query, selectedCollection || undefined);
    },
    [input, loading, send, selectedCollection],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e as unknown as FormEvent);
      }
    },
    [handleSubmit],
  );

  const handleDeleteConversation = useCallback(async () => {
    if (!conversationId) return;
    // Confirm before destroying persisted state. Inline confirm() is
    // intentionally low-tech for a debug surface; a proper modal can be
    // added later if this graduates beyond dev.
    if (!window.confirm("Delete this conversation? This cannot be undone.")) {
      return;
    }
    try {
      await deleteConversation(conversationId);
    } catch {
      // Silent: the next refetch will reflect server state regardless.
    }
    // Reset chat state so the next message starts a fresh conversation,
    // matching the "New conversation" affordance.
    clear();
    setHistoryPanelOpen(false);
    // Force the sidebar to re-pull immediately so the deleted row drops
    // out without waiting for the next conversationId change.
    refetchConversationList();
  }, [conversationId, clear, refetchConversationList]);

  return (
    <div className="flex h-full">
      {/* Leftmost panel -- conversation navigator. Always visible: empty
          state shows a hint to start a new conversation. */}
      <ConversationListSidebar
        overviews={conversationOverviews}
        activeConversationId={conversationId}
        onSelect={loadConversation}
      />

      {/* Middle panel -- collection picker */}
      {collections.length > 0 && (
        <aside className="flex w-52 flex-col border-r border-gray-700">
          <div className="border-b border-gray-700 px-4 py-3">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
              Collections
            </h2>
          </div>
          <ul className="flex-1 overflow-y-auto">
            <li
              role="button"
              tabIndex={0}
              onClick={() => setSelectedCollection("")}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") setSelectedCollection("");
              }}
              className={`cursor-pointer px-4 py-2 text-sm transition-colors ${
                selectedCollection === ""
                  ? "bg-indigo-900/40 text-indigo-300"
                  : "text-gray-300 hover:bg-gray-800"
              }`}
            >
              All collections
            </li>
            {collections.map((c) => (
              <li
                key={c}
                role="button"
                tabIndex={0}
                onClick={() => setSelectedCollection(c)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") setSelectedCollection(c);
                }}
                className={`cursor-pointer px-4 py-2 text-sm transition-colors ${
                  selectedCollection === c
                    ? "bg-indigo-900/40 text-indigo-300"
                    : "text-gray-300 hover:bg-gray-800"
                }`}
              >
                {c}
              </li>
            ))}
          </ul>
        </aside>
      )}

      {/* Right panel -- chat messages and input */}
      <div className="flex flex-1 flex-col">
        {/* Conversation header: shows id of current conversation and a
            button to start a new one. Only visible once a conversation has
            actually started -- no chrome on the empty state. */}
        {conversationId && (
          <div className="flex items-center justify-between border-b border-gray-700 bg-gray-900/40 px-4 py-2 text-xs">
            <div className="flex items-center gap-3 text-gray-500">
              <span>
                Conversation{" "}
                <code className="rounded bg-gray-800 px-1.5 py-0.5 font-mono text-gray-400">
                  {conversationId.slice(0, 8)}
                </code>
              </span>
              {conversationDetail && (
                <>
                  <span className="text-gray-600">·</span>
                  <span
                    title={
                      conversationDetail.summary
                        ? "Verbatim turns kept after summarisation; older turns are folded into the summary."
                        : `turn count for ${conversationId}`
                    }
                  >
                    {conversationDetail.turns.length}{" "}
                    turn{conversationDetail.turns.length !== 1 && "s"}
                  </span>
                  {conversationDetail.summary && (
                    <>
                      <span className="text-gray-600">·</span>
                      <span
                        title="A rolling summary has been generated for older turns. Hover the turn count to see how the verbatim window is computed."
                        className="rounded bg-indigo-900/40 px-1.5 py-0.5 text-indigo-300"
                      >
                        summarised
                      </span>
                    </>
                  )}
                </>
              )}
            </div>
            <div className="flex items-center gap-2">
              {conversationDetail && (
                <button
                  type="button"
                  onClick={() => setHistoryPanelOpen((v) => !v)}
                  className="rounded border border-gray-700 px-2 py-1 text-gray-400 transition-colors hover:border-indigo-500 hover:text-indigo-400"
                  aria-expanded={historyPanelOpen}
                >
                  {historyPanelOpen ? "Hide" : "View"} history
                </button>
              )}
              <button
                type="button"
                onClick={clear}
                className="rounded border border-gray-700 px-2 py-1 text-gray-400 transition-colors hover:border-indigo-500 hover:text-indigo-400"
              >
                New conversation
              </button>
            </div>
          </div>
        )}

        {/* History panel surfaces what's persisted server-side -- the rolling
            summary plus verbatim post-boundary turns. Renders below the
            header so it's clearly tied to the active conversation. */}
        {conversationId && historyPanelOpen && conversationDetail && (
          <ConversationHistoryPanel
            detail={conversationDetail}
            onDelete={handleDeleteConversation}
          />
        )}

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && !loading && (
            <p className="pt-16 text-center text-sm text-gray-500">
              Ask a question to start the conversation.
            </p>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-lg bg-gray-800 px-4 py-3 text-sm text-gray-400">
                {activeNode ? (
                  <span>
                    Running <span className="font-medium text-indigo-400">{activeNode}</span>...
                  </span>
                ) : (
                  "Thinking..."
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-800 bg-red-900/30 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form
          onSubmit={handleSubmit}
          className="flex items-end gap-2 border-t border-gray-700 bg-gray-800 p-4"
        >
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="flex-1 resize-none rounded border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:border-indigo-500 focus:outline-none"
          />

          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
