from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentConfig(_StrictModel):
    mode: Literal["local", "cloudflare"] = "local"
    log_level: str = "info"


class LLMConfig(_StrictModel):
    provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"


class TracingConfig(_StrictModel):
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_project: str = "langgraph-chatbot"


class VectorStoreConfig(_StrictModel):
    provider: Literal["qdrant", "vectorize"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    # Vectorize settings (for Cloudflare deployment -- credentials via env vars)
    vectorize_index_name: str = "langgraph-chatbot"
    # Known logical collections for Vectorize (Vectorize cannot enumerate them natively).
    known_collections: list[str] = ["langgraph-docs", "notion-docs", "edgenotes"]


class EmbeddingsConfig(_StrictModel):
    provider: Literal["ollama", "workers-ai"] = "ollama"
    # Ollama settings (default for local dev)
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    # Workers AI settings (for Cloudflare deployment)
    workers_ai_model: str = "@cf/baai/bge-small-en-v1.5"
    workers_ai_base_url: str = "https://api.cloudflare.com/client/v4/accounts"


class IngestionConfig(_StrictModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    supported_formats: list[str] = ["md", "txt", "pdf"]


class NotionConfig(_StrictModel):
    default_collection: str = "notion-docs"


class WorklogConfig(_StrictModel):
    pass


class WebhooksConfig(_StrictModel):
    edgenotes_secret: str = ""
    edgenotes_collection: str = "edgenotes"


class SystemConfig(_StrictModel):
    environment: EnvironmentConfig = EnvironmentConfig()
    llm: LLMConfig = LLMConfig()
    tracing: TracingConfig = TracingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    ingestion: IngestionConfig = IngestionConfig()
    notion: NotionConfig = NotionConfig()
    worklog: WorklogConfig = WorklogConfig()
    webhooks: WebhooksConfig = WebhooksConfig()


# A node config may set `provider` to override the system default LLM (e.g. the
# verifier may run on Anthropic while the rest of the graph stays on Ollama).
# `None` means "use system default", set in config/config.toml under [llm].
ProviderOverride = Literal["ollama", "anthropic"] | None


class RouterConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"
    provider: ProviderOverride = None
    prompt: str = ""
    routes: list[str] = ["chat", "rag", "tool"]


class ChatAgentConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"
    provider: ProviderOverride = None
    system_prompt: str = "You are a helpful assistant. Answer clearly and concisely."
    max_tokens: int = 2048


class RetrievalConfig(_StrictModel):
    enabled: bool = True
    top_k: int = 10
    score_threshold: float = 0.7
    rerank: bool = True
    default_collection: str = "langgraph-docs"


class AnswerGenerationConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.1:8b"
    provider: ProviderOverride = None
    prompt_template: str = (
        "Answer the user's question using only the evidence provided below. "
        "Cite chunk IDs inline where you use them.\n\nEvidence:\n{evidence}\n\nQuestion: {query}"
    )
    max_tokens: int = 2048


class VerifierConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.1:8b"
    provider: ProviderOverride = None
    score_threshold: float = 0.75
    citation_coverage_min: float = 0.8
    max_retries: int = 2
    checks: list[str] = ["score_threshold", "support_analysis", "citation_coverage"]


class ToolAgentConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"
    provider: ProviderOverride = None
    system_prompt: str = (
        "You are a tool-using assistant. Use the available tools to answer the user's request. "
        "Always call the appropriate tool first, then report what the tool returned. "
        "Never guess or make up results -- only report what the tools actually return."
    )
    allowed_tools: list[str] = []
    max_tool_calls: int = 5


class WorklogAgentConfig(_StrictModel):
    enabled: bool = True
    model: str = "llama3.2:3b"
    provider: ProviderOverride = None
    system_prompt: str = (
        "You are a worklog assistant. You help the user understand their logged hours "
        "and worklog plans. You can list existing plans, show plan details, and generate "
        "new plans. You CANNOT apply plans or modify Jira/Tempo data -- only read and generate. "
        "Present information clearly and concisely."
    )
    max_tokens: int = 2048


class AgentsConfig(_StrictModel):
    router: RouterConfig = RouterConfig()
    chat_agent: ChatAgentConfig = ChatAgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    answer_generation: AnswerGenerationConfig = AnswerGenerationConfig()
    verifier: VerifierConfig = VerifierConfig()
    tool_agent: ToolAgentConfig = ToolAgentConfig()
    worklog_agent: WorklogAgentConfig = WorklogAgentConfig()
