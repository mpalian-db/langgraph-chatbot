/**
 * SummarisationToast -- ephemeral "earlier turns just got compressed"
 * notification.
 *
 * Triggered by ChatView when a chat round's trace includes a
 * `memory_load` entry with `summarisation_triggered: true`. Auto-dismisses
 * after `durationMs`. The user can dismiss manually too.
 *
 * Why a toast and not a permanent banner: the event is per-chat-round
 * (it fires when the threshold is crossed; doesn't keep firing). A
 * permanent indicator would lie about the current state. The toast
 * captures "this just happened" without sticking around.
 */

import { useEffect, useState } from "react";

interface Props {
  /** Monotonically-increasing key. Each new value re-triggers the toast.
   *  Pass any value that changes when summarisation fires (e.g. a
   *  counter, or a trace-entry id). 0 means "never shown yet". */
  triggerKey: number;
  /** Auto-dismiss timeout. Defaults to 4500ms. */
  durationMs?: number;
}

export default function SummarisationToast({ triggerKey, durationMs = 4500 }: Props) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (triggerKey === 0) return; // initial render, nothing to show

    setVisible(true);
    const handle = window.setTimeout(() => setVisible(false), durationMs);
    return () => window.clearTimeout(handle);
  }, [triggerKey, durationMs]);

  if (!visible) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed bottom-6 right-6 z-50 max-w-sm rounded-lg border border-purple-700/60 bg-purple-900/30 px-4 py-3 shadow-lg backdrop-blur-sm"
    >
      <div className="flex items-start gap-3">
        <span aria-hidden="true" className="text-purple-300">
          ◆
        </span>
        <div className="flex-1">
          <h4 className="text-sm font-medium text-purple-200">
            Conversation compressed
          </h4>
          <p className="mt-0.5 text-xs text-purple-300/80">
            Older turns folded into a rolling summary. View the
            history panel to see what was preserved.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setVisible(false)}
          className="text-purple-400 transition-colors hover:text-purple-200"
          aria-label="Dismiss notification"
        >
          ×
        </button>
      </div>
    </div>
  );
}
