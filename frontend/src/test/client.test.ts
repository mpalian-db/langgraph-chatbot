import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  sendChatMessage,
  listCollections,
  createCollection,
  getCollectionStats,
  deleteCollection,
  deleteConversation,
  listConversations,
} from "../api/client";

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "OK",
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

describe("API client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch(200, {}));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("sendChatMessage", () => {
    it("POSTs to /api/chat and returns the response", async () => {
      const response = {
        answer: "LangGraph is a library.",
        route: "rag",
        citations: [],
        trace: [],
      };
      vi.stubGlobal("fetch", mockFetch(200, response));

      const result = await sendChatMessage({ query: "What is LangGraph?" });

      expect(fetch).toHaveBeenCalledWith(
        "/api/chat",
        expect.objectContaining({ method: "POST" }),
      );
      expect(result).toEqual(response);
    });

    it("throws ApiError on non-2xx response", async () => {
      vi.stubGlobal("fetch", mockFetch(500, "Internal Server Error"));
      await expect(sendChatMessage({ query: "test" })).rejects.toThrow();
    });
  });

  describe("listCollections", () => {
    it("GETs /api/collections and returns string array", async () => {
      vi.stubGlobal("fetch", mockFetch(200, ["langgraph-docs", "notion-docs"]));
      const result = await listCollections();
      expect(fetch).toHaveBeenCalledWith("/api/collections", undefined);
      expect(result).toEqual(["langgraph-docs", "notion-docs"]);
    });
  });

  describe("createCollection", () => {
    it("POSTs with name and default vector_size 768", async () => {
      vi.stubGlobal("fetch", mockFetch(200, { name: "my-col", status: "created" }));
      await createCollection({ name: "my-col" });
      const [, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
      const body = JSON.parse(init.body);
      expect(body).toEqual({ name: "my-col", vector_size: 768 });
    });
  });

  describe("getCollectionStats", () => {
    it("GETs the correct path with URL encoding", async () => {
      vi.stubGlobal(
        "fetch",
        mockFetch(200, { name: "my col", vectors_count: 10, points_count: 10 }),
      );
      await getCollectionStats("my col");
      expect(fetch).toHaveBeenCalledWith(
        "/api/collections/my%20col",
        undefined,
      );
    });
  });

  describe("deleteCollection", () => {
    it("sends DELETE to the correct path", async () => {
      vi.stubGlobal("fetch", mockFetch(204, null));
      await deleteCollection("my-col");
      expect(fetch).toHaveBeenCalledWith(
        "/api/collections/my-col",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
  });

  describe("listConversations", () => {
    it("GETs /api/conversations and returns the array", async () => {
      const overviews = [
        {
          conversation_id: "abc",
          turn_count: 2,
          has_summary: false,
          last_updated_at: 1700000000,
        },
      ];
      vi.stubGlobal("fetch", mockFetch(200, overviews));
      const result = await listConversations();
      expect(fetch).toHaveBeenCalledWith(
        "/api/conversations",
        expect.objectContaining({ signal: undefined }),
      );
      expect(result).toEqual(overviews);
    });
  });

  describe("deleteConversation", () => {
    it("sends DELETE to /api/conversations/{id}", async () => {
      vi.stubGlobal("fetch", mockFetch(204, null));
      await deleteConversation("conv-abc");
      expect(fetch).toHaveBeenCalledWith(
        "/api/conversations/conv-abc",
        expect.objectContaining({ method: "DELETE" }),
      );
    });

    it("encodes conversation ids that contain reserved characters", async () => {
      vi.stubGlobal("fetch", mockFetch(204, null));
      await deleteConversation("conv with/slash");
      expect(fetch).toHaveBeenCalledWith(
        "/api/conversations/conv%20with%2Fslash",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
  });
});
