/**
 * Collections management view -- create/delete collections, view stats,
 * select a collection, and upload documents for ingestion.
 */

import {
  type ChangeEvent,
  type FormEvent,
  useCallback,
  useRef,
  useState,
} from "react";
import { useCollections } from "../hooks/useCollections";

// ---------------------------------------------------------------------------
// CollectionsView
// ---------------------------------------------------------------------------

export default function CollectionsView() {
  const {
    names,
    stats,
    selected,
    loading,
    error,
    refresh,
    select,
    create,
    remove,
    upload,
  } = useCollections();

  const [newName, setNewName] = useState("");
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // -- Create collection ---------------------------------------------------

  const handleCreate = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const name = newName.trim();
      if (!name) return;
      try {
        await create(name);
        setNewName("");
      } catch {
        // Error is surfaced via the hook.
      }
    },
    [newName, create],
  );

  // -- Delete collection ---------------------------------------------------

  const handleDelete = useCallback(
    async (name: string) => {
      const confirmed = window.confirm(
        `Delete collection "${name}" and all its vectors?`,
      );
      if (!confirmed) return;
      try {
        await remove(name);
      } catch {
        // Error is surfaced via the hook.
      }
    },
    [remove],
  );

  // -- Upload document -----------------------------------------------------

  const handleFileChange = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      if (!selected) return;
      const file = e.target.files?.[0];
      if (!file) return;

      setUploadStatus(`Uploading ${file.name}...`);
      try {
        const result = await upload(selected, file);
        setUploadStatus(
          `Ingested ${result.filename}: ${result.chunk_count} chunks`,
        );
      } catch {
        setUploadStatus("Upload failed");
      }

      // Reset the input so the same file can be re-uploaded.
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [selected, upload],
  );

  // -- Render --------------------------------------------------------------

  return (
    <div className="flex h-full">
      {/* Left panel -- collection list */}
      <aside className="flex w-72 flex-col border-r border-gray-700 bg-gray-850">
        <div className="flex items-center justify-between border-b border-gray-700 px-4 py-3">
          <h2 className="text-sm font-semibold text-gray-200">Collections</h2>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={loading}
            className="text-xs text-gray-400 hover:text-gray-200"
          >
            Refresh
          </button>
        </div>

        {/* Create form */}
        <form onSubmit={handleCreate} className="flex gap-2 p-3">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="New collection..."
            className="flex-1 rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-100 placeholder-gray-500 focus:border-indigo-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!newName.trim()}
            className="rounded bg-indigo-600 px-3 py-1 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            Create
          </button>
        </form>

        {/* List */}
        <ul className="flex-1 overflow-y-auto">
          {names.map((name) => (
            <li
              key={name}
              role="button"
              tabIndex={0}
              onClick={() => select(name)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") select(name);
              }}
              className={`flex cursor-pointer items-center justify-between px-4 py-2 text-left text-sm transition-colors ${
                selected === name
                  ? "bg-indigo-900/40 text-indigo-300"
                  : "text-gray-300 hover:bg-gray-800"
              }`}
            >
              <span className="truncate">{name}</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  void handleDelete(name);
                }}
                className="ml-2 text-xs text-red-500 hover:text-red-400"
              >
                Delete
              </button>
            </li>
          ))}
          {names.length === 0 && !loading && (
            <li className="px-4 py-6 text-center text-xs text-gray-500">
              No collections yet
            </li>
          )}
        </ul>

        {error && (
          <div className="border-t border-red-800 bg-red-900/30 px-4 py-2 text-xs text-red-300">
            {error}
          </div>
        )}
      </aside>

      {/* Right panel -- details and upload */}
      <section className="flex flex-1 flex-col p-6">
        {!selected ? (
          <p className="pt-16 text-center text-sm text-gray-500">
            Select or create a collection to view details and upload documents.
          </p>
        ) : (
          <>
            <h2 className="text-lg font-semibold text-gray-100">{selected}</h2>

            {/* Stats */}
            {stats && (
              <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
                <div className="rounded border border-gray-700 bg-gray-800 p-4">
                  <dt className="text-xs font-medium text-gray-400">
                    Vectors
                  </dt>
                  <dd className="mt-1 text-xl font-semibold text-gray-100">
                    {stats.vectors_count}
                  </dd>
                </div>
                <div className="rounded border border-gray-700 bg-gray-800 p-4">
                  <dt className="text-xs font-medium text-gray-400">Points</dt>
                  <dd className="mt-1 text-xl font-semibold text-gray-100">
                    {stats.points_count}
                  </dd>
                </div>
              </dl>
            )}

            {/* Upload */}
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-200">
                Upload document
              </h3>
              <p className="mt-1 text-xs text-gray-500">
                Select a Markdown or text file to ingest into this collection.
              </p>

              <input
                ref={fileInputRef}
                type="file"
                accept=".md,.txt,.markdown"
                onChange={handleFileChange}
                className="mt-3 block text-sm text-gray-400 file:mr-3 file:rounded file:border-0 file:bg-indigo-600 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-indigo-500"
              />

              {uploadStatus && (
                <p className="mt-2 text-xs text-gray-400">{uploadStatus}</p>
              )}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
