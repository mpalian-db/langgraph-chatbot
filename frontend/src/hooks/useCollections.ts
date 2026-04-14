/**
 * Hook for managing collection state -- listing, creating, deleting
 * collections, fetching stats, and uploading documents.
 */

import { useCallback, useEffect, useState } from "react";
import {
  createCollection,
  deleteCollection,
  getCollectionStats,
  listCollections,
  uploadDocument,
} from "../api/client";
import type { CollectionStats, IngestResponse } from "../api/types";

export interface UseCollectionsReturn {
  /** All known collection names. */
  names: string[];
  /** Stats for the currently selected collection, if any. */
  stats: CollectionStats | null;
  /** Currently selected collection name. */
  selected: string | null;
  loading: boolean;
  error: string | null;

  refresh: () => Promise<void>;
  select: (name: string | null) => void;
  create: (name: string) => Promise<void>;
  remove: (name: string) => Promise<void>;
  upload: (collection: string, file: File) => Promise<IngestResponse>;
}

export function useCollections(): UseCollectionsReturn {
  const [names, setNames] = useState<string[]>([]);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await listCollections();
      setNames(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load collections");
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch the list on mount.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Fetch stats whenever the selection changes.
  useEffect(() => {
    if (!selected) {
      setStats(null);
      return;
    }

    let cancelled = false;

    async function fetchStats() {
      try {
        const result = await getCollectionStats(selected!);
        if (!cancelled) setStats(result);
      } catch {
        if (!cancelled) setStats(null);
      }
    }

    void fetchStats();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const select = useCallback((name: string | null) => {
    setSelected(name);
  }, []);

  const create = useCallback(
    async (name: string) => {
      setError(null);
      try {
        await createCollection({ name, vector_size: 768 });
        await refresh();
        setSelected(name);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to create collection");
        throw err;
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (name: string) => {
      setError(null);
      try {
        await deleteCollection(name);
        if (selected === name) {
          setSelected(null);
          setStats(null);
        }
        await refresh();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to delete collection",
        );
        throw err;
      }
    },
    [refresh, selected],
  );

  const upload = useCallback(
    async (collection: string, file: File): Promise<IngestResponse> => {
      setError(null);
      try {
        const result = await uploadDocument(collection, file);
        // Refresh stats for the active collection after ingestion.
        if (selected === collection) {
          try {
            const updated = await getCollectionStats(collection);
            setStats(updated);
          } catch {
            // Non-critical -- stats will refresh on next select.
          }
        }
        return result;
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to upload document",
        );
        throw err;
      }
    },
    [selected],
  );

  return {
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
  };
}
