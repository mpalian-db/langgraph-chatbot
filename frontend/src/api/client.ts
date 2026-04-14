/**
 * Fetch-based API client for the FastAPI backend.
 *
 * All paths are relative -- the Vite dev server proxies /api/* to the
 * backend at http://localhost:8000.
 */

import type {
  ChatRequest,
  ChatResponse,
  CollectionStats,
  CreateCollectionRequest,
  DocumentOut,
  IngestResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(path, init);

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }

  // 204 No Content -- nothing to parse.
  if (res.status === 204) {
    return undefined as unknown as T;
  }

  return (await res.json()) as T;
}

function jsonBody(data: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export async function sendChatMessage(
  body: ChatRequest,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", jsonBody(body));
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

export async function listCollections(): Promise<string[]> {
  return request<string[]>("/api/collections");
}

export async function createCollection(
  body: CreateCollectionRequest,
): Promise<{ name: string; status: string }> {
  return request<{ name: string; status: string }>(
    "/api/collections",
    jsonBody({ name: body.name, vector_size: body.vector_size ?? 768 }),
  );
}

export async function getCollectionStats(
  name: string,
): Promise<CollectionStats> {
  return request<CollectionStats>(`/api/collections/${encodeURIComponent(name)}`);
}

export async function deleteCollection(name: string): Promise<void> {
  return request<void>(`/api/collections/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export async function uploadDocument(
  collection: string,
  file: File,
): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);

  return request<IngestResponse>(
    `/api/collections/${encodeURIComponent(collection)}/documents`,
    { method: "POST", body: form },
  );
}

export async function listDocuments(
  collection: string,
  limit = 100,
  offset = 0,
): Promise<DocumentOut[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<DocumentOut[]>(
    `/api/collections/${encodeURIComponent(collection)}/documents?${params}`,
  );
}
