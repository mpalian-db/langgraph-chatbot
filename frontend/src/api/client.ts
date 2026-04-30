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
  ConversationDetailOut,
  CreateCollectionRequest,
  DocumentOut,
  IngestResponse,
  StreamEvent,
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

export async function* sendChatStream(
  body: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  // Forward the signal so the underlying TCP/fetch can be aborted mid-stream.
  // Without this, a hung backend would leave the UI permanently in `loading`
  // state with no escape path -- see the "New conversation" affordance in
  // useChat.ts which calls AbortController.abort() to recover.
  const res = await fetch("/api/chat/stream", { ...jsonBody(body), signal });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text || res.statusText);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) {
        yield JSON.parse(trimmed) as StreamEvent;
      }
    }
  }

  // Flush any remaining data in the buffer.
  if (buffer.trim()) {
    yield JSON.parse(buffer.trim()) as StreamEvent;
  }
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

// ---------------------------------------------------------------------------
// Conversations (debug/introspection)
// ---------------------------------------------------------------------------

export async function getConversation(
  conversationId: string,
): Promise<ConversationDetailOut> {
  return request<ConversationDetailOut>(
    `/api/conversations/${encodeURIComponent(conversationId)}`,
  );
}
