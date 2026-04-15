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

// ---------------------------------------------------------------------------
// TraceEntry row
// ---------------------------------------------------------------------------

function TraceRow({ entry }: { entry: TraceEntryOut }) {
  const dataEntries = Object.entries(entry.data);

  return (
    <li className="flex flex-col gap-0.5 rounded border border-gray-700 bg-gray-800/60 px-2 py-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-gray-300 text-xs">{entry.node}</span>
        <span className="text-gray-500 text-xs tabular-nums">
          {formatMs(entry.duration_ms)}
        </span>
      </div>
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
            <TraceRow key={`${entry.node}-${i}`} entry={entry} />
          ))}
        </ul>
      )}
    </div>
  );
}
