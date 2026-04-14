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
import { listCollections } from "../api/client";
import type { CitationOut, Message, TraceEntryOut } from "../api/types";

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
// Trace details
// ---------------------------------------------------------------------------

function Trace({ entries }: { entries: TraceEntryOut[] }) {
  const [expanded, setExpanded] = useState(false);

  if (entries.length === 0) return null;

  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="text-xs font-medium text-gray-500 hover:text-gray-400"
      >
        {expanded ? "Hide" : "Show"} trace ({entries.length} node
        {entries.length !== 1 && "s"})
      </button>
      {expanded && (
        <ul className="mt-1 space-y-1 text-xs text-gray-500">
          {entries.map((t, i) => (
            <li key={`${t.node}-${i}`}>
              <span className="font-medium text-gray-400">{t.node}</span>{" "}
              <span>{t.duration_ms.toFixed(0)}ms</span>
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
        {!isUser && msg.trace && <Trace entries={msg.trace} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatView
// ---------------------------------------------------------------------------

export default function ChatView() {
  const { messages, loading, error, send, clear } = useChat();
  const [input, setInput] = useState("");
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollection, setSelectedCollection] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

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

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
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
              Thinking...
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

      {/* Input area */}
      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-2 border-t border-gray-700 bg-gray-800 p-4"
      >
        {/* Collection selector */}
        {collections.length > 0 && (
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            className="rounded border border-gray-600 bg-gray-700 px-2 py-2 text-sm text-gray-200 focus:border-indigo-500 focus:outline-none"
          >
            <option value="">All collections</option>
            {collections.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}

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

        <button
          type="button"
          onClick={clear}
          className="rounded border border-gray-600 px-3 py-2 text-sm text-gray-400 transition-colors hover:border-gray-500 hover:text-gray-200"
        >
          Clear
        </button>
      </form>
    </div>
  );
}
