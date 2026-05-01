/**
 * ConversationHistoryPanel -- read-only debug view of what's persisted
 * server-side for the active conversation.
 *
 * Sourced from `GET /api/conversations/{id}` (the introspection endpoint),
 * which returns the rolling summary plus every verbatim turn since the
 * summary's boundary. Useful to verify in dev that summarisation produced
 * something coherent and that history is being persisted correctly --
 * without dropping to `sqlite3 data/conversations.sqlite` on the CLI.
 */

import type { ConversationDetailOut } from "../api/types";

interface Props {
  detail: ConversationDetailOut;
  onDelete?: () => void;
}

const ROLE_COLOURS: Record<string, string> = {
  user: "border-emerald-700/40 bg-emerald-900/10",
  assistant: "border-indigo-700/40 bg-indigo-900/10",
};

export default function ConversationHistoryPanel({ detail, onDelete }: Props) {
  return (
    <div className="border-b border-gray-700 bg-gray-900/30 px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Persisted history
        </h3>
        {onDelete && (
          <button
            type="button"
            onClick={onDelete}
            className="rounded border border-red-900/50 px-2 py-0.5 text-[10px] uppercase tracking-wide text-red-400 transition-colors hover:border-red-700 hover:text-red-300"
            aria-label="Delete this conversation"
          >
            Delete
          </button>
        )}
      </div>

      {/* Summary box: only present once summarisation has triggered. Fold-old-
          turns-into-text contract is documented in ADR 0002. */}
      {detail.summary && (
        <details
          open
          className="mb-3 rounded border border-purple-700/40 bg-purple-900/20 p-2"
        >
          <summary className="cursor-pointer text-xs font-medium text-purple-300">
            Rolling summary
          </summary>
          <p className="mt-1 whitespace-pre-wrap text-xs text-gray-300">
            {detail.summary}
          </p>
        </details>
      )}

      {/* Verbatim turns. With a summary present these are post-boundary;
          without a summary they're the entire conversation. */}
      {detail.turns.length === 0 ? (
        <p className="text-xs italic text-gray-500">
          No verbatim turns past the summary boundary.
        </p>
      ) : (
        <ol className="space-y-1.5">
          {detail.turns.map((turn, i) => {
            const colour =
              ROLE_COLOURS[turn.role] ?? "border-gray-700 bg-gray-800/40";
            return (
              <li
                // We don't get stable ids back from the introspection
                // endpoint (by design -- it returns the port's view, not
                // storage rows). Index keying is fine here because the list
                // is read-only and re-renders on each re-fetch.
                key={`${turn.role}-${i}`}
                className={`rounded border ${colour} px-2 py-1.5`}
              >
                <span className="font-medium text-gray-400 text-[10px] uppercase tracking-wide">
                  {turn.role}
                </span>
                <p className="mt-0.5 whitespace-pre-wrap text-xs text-gray-300">
                  {turn.content}
                </p>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
