/**
 * TraceView -- renders the execution trace returned by the /chat endpoint.
 *
 * Each entry shows the node name, duration, and any key data fields
 * (e.g. route, chunks_retrieved, outcome, score).
 */

import { useState } from "react";
import type { TraceEntryOut } from "../api/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a duration in milliseconds to a readable string. */
function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms.toFixed(0)}ms`;
}

/** Render a single data field value as a compact string. */
function renderValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toFixed(2);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// Per-node-type colour palette for the duration bar. Falls back to a neutral
// indigo for unknown nodes so a future node type renders cleanly without a
// frontend change.
const NODE_COLOURS: Record<string, string> = {
  memory_load: "bg-purple-700/70",
  router: "bg-emerald-700/70",
  chat_agent: "bg-emerald-700/70",
  retrieval: "bg-amber-700/70",
  answer_generation: "bg-blue-700/70",
  verifier: "bg-rose-700/70",
  tool_agent: "bg-amber-700/70",
};

// ---------------------------------------------------------------------------
// TraceEntry row
// ---------------------------------------------------------------------------

function TraceRow({
  entry,
  maxMs,
}: {
  entry: TraceEntryOut;
  maxMs: number;
}) {
  const dataEntries = Object.entries(entry.data);
  // Width is a fraction of the slowest node's duration. The visual lets the
  // eye spot which node dominated the request -- e.g. memory_load taking
  // 80% of the time when summarisation triggered.
  const widthPct = maxMs > 0 ? (entry.duration_ms / maxMs) * 100 : 0;
  const colour = NODE_COLOURS[entry.node] ?? "bg-indigo-700/70";

  return (
    <li className="flex flex-col gap-0.5 rounded border border-gray-700 bg-gray-800/60 px-2 py-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-gray-300 text-xs">{entry.node}</span>
        <span className="text-gray-500 text-xs tabular-nums">
          {formatMs(entry.duration_ms)}
        </span>
      </div>
      {/* Duration bar: only render when there's at least one nonzero entry,
          otherwise we get a sliver of nothing for empty traces. */}
      {maxMs > 0 && (
        <div className="h-1 w-full overflow-hidden rounded-sm bg-gray-700/50">
          <div
            className={`h-full ${colour}`}
            style={{ width: `${widthPct}%` }}
            aria-hidden="true"
          />
        </div>
      )}
      {dataEntries.length > 0 && (
        <dl className="flex flex-wrap gap-x-3 gap-y-0.5">
          {dataEntries.map(([key, val]) => (
            <div key={key} className="flex items-center gap-1">
              <dt className="text-gray-600 text-xs">{key}</dt>
              <dd className="text-gray-400 text-xs">{renderValue(val)}</dd>
            </div>
          ))}
        </dl>
      )}
    </li>
  );
}

// ---------------------------------------------------------------------------
// TraceView (exported)
// ---------------------------------------------------------------------------

interface TraceViewProps {
  entries: TraceEntryOut[];
}

export default function TraceView({ entries }: TraceViewProps) {
  const [expanded, setExpanded] = useState(false);

  if (entries.length === 0) return null;

  const totalMs = entries.reduce((sum, e) => sum + e.duration_ms, 0);
  const maxMs = entries.reduce((m, e) => Math.max(m, e.duration_ms), 0);

  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="text-xs font-medium text-gray-500 hover:text-gray-400"
      >
        {expanded ? "Hide" : "Show"} trace ({entries.length} node
        {entries.length !== 1 && "s"}, {formatMs(totalMs)} total)
      </button>
      {expanded && (
        <ul className="mt-1 space-y-1">
          {entries.map((entry, i) => (
            <TraceRow key={`${entry.node}-${i}`} entry={entry} maxMs={maxMs} />
          ))}
        </ul>
      )}
    </div>
  );
}
