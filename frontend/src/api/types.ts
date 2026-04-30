/**
 * TypeScript types matching the FastAPI backend response models.
 *
 * These are kept in sync with the Pydantic schemas defined in
 * backend/app/api/routes/{chat,collections,documents}.py.
 */

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatRequest {
  query: string;
  collection?: string;
  // Bind this turn to an existing conversation. When omitted the server
  // generates a new uuid and returns it on the response, which the client
  // should then echo back on follow-up turns to maintain history.
  conversation_id?: string;
}

export interface CitationOut {
  chunk_id: string;
  text: string;
  collection: string;
}

export interface TraceEntryOut {
  node: string;
  duration_ms: number;
  data: Record<string, unknown>;
}

export interface ChatResponse {
  answer: string;
  // Always populated by the backend -- generated on first turn, echoed
  // verbatim on subsequent turns of the same conversation.
  conversation_id: string;
  route: string | null;
  citations: CitationOut[];
  trace: TraceEntryOut[];
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

export interface CollectionStats {
  name: string;
  vectors_count: number;
  points_count: number;
}

export interface CreateCollectionRequest {
  name: string;
  vector_size?: number;
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export interface IngestResponse {
  filename: string;
  collection: string;
  chunk_count: number;
}

export interface DocumentOut {
  id: string;
  text: string;
  collection: string;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Streaming chat events (NDJSON from /api/chat/stream)
// ---------------------------------------------------------------------------

export interface StreamEventNodeStart {
  event: "node_start";
  node: string;
}

export interface StreamEventNodeEnd {
  event: "node_end";
  node: string;
}

export interface StreamEventResult {
  event: "result";
  data: ChatResponse;
}

export type StreamEvent =
  | StreamEventNodeStart
  | StreamEventNodeEnd
  | StreamEventResult;

// ---------------------------------------------------------------------------
// Conversations (debug/introspection)
// ---------------------------------------------------------------------------

export interface ConversationOverviewOut {
  conversation_id: string;
  turn_count: number;
  has_summary: boolean;
  last_updated_at: number | null;
}

export interface TurnOut {
  role: string;
  content: string;
}

export interface ConversationDetailOut {
  conversation_id: string;
  summary: string | null;
  turns: TurnOut[];
}

// ---------------------------------------------------------------------------
// UI-level types (not from backend)
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  route?: string | null;
  citations?: CitationOut[];
  trace?: TraceEntryOut[];
}
