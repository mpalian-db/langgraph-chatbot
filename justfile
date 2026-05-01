set dotenv-load := true

# ─── Ollama ───────────────────────────────────────────────────────────────────

OLLAMA_MODELS := "llama3.2:3b llama3.1:8b nomic-embed-text"

# Start the Ollama server (no-op if already running)
ollama-start:
    ollama serve &>/dev/null & sleep 2 && echo "Ollama running"

# Check that all required models are pulled; pull any that are missing
ollama-check:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    for model in {{OLLAMA_MODELS}}; do
        if ollama show "$model" &>/dev/null; then
            echo "  ok  $model"
        else
            echo "  --  $model (missing)"
            missing+=("$model")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "Pulling ${#missing[@]} missing model(s)..."
        for model in "${missing[@]}"; do
            echo "  pulling $model"
            ollama pull "$model"
        done
        echo "All models ready."
    else
        echo "All models already pulled."
    fi

# ─── Ingestion ────────────────────────────────────────────────────────────────

# Ingest all files from corpus/<collection> into the named collection.
# Usage: just ingest langgraph-docs
# Requires the backend to be running on localhost:8000.
ingest collection:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="corpus/{{collection}}"
    if [ ! -d "$dir" ]; then
        echo "Directory $dir not found"
        exit 1
    fi
    # Create the collection (ignore 409 if it already exists).
    status=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/api/collections \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"{{collection}}\", \"vector_size\": 768}")
    if [ "$status" != "200" ] && [ "$status" != "409" ]; then
        echo "Failed to create collection (HTTP $status)"
        exit 1
    fi
    echo "Ingesting files from $dir into collection '{{collection}}'..."
    count=0
    for f in "$dir"/*.md "$dir"/*.txt "$dir"/*.pdf; do
        [ -f "$f" ] || continue
        filename=$(basename "$f")
        echo "  uploading $filename"
        curl -s -X POST "http://localhost:8000/api/collections/{{collection}}/documents" \
            -F "file=@$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'    -> {d[\"chunk_count\"]} chunks')"
        count=$((count + 1))
    done
    echo "Done. Ingested $count file(s) into '{{collection}}'."

# ─── Dev ──────────────────────────────────────────────────────────────────────

# Start Qdrant + Langfuse + backend + frontend for local dev.
# Note: requires frontend/ to be scaffolded (see Task 25) before first run.
dev:
    docker compose up qdrant langfuse-db langfuse-server -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev

# Start full stack including n8n
dev-full:
    docker compose up qdrant langfuse-db langfuse-server -d
    just n8n-up
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev

# Run unit tests only
test:
    cd backend && uv run pytest tests/unit -v

# Run all tests including integration
test-all:
    cd backend && uv run pytest tests/ -v

# Run a single test file
test-one FILE:
    cd backend && uv run pytest {{FILE}} -v

# Lint (no auto-fix). Matches CI exactly: ruff + mypy.
# Mypy was missing before -- a type error could pass `just lint` locally
# but fail CI. Now they stay in lockstep.
lint:
    cd backend && uv run ruff check . && uv run ruff format --check .
    cd backend && uv run mypy app --ignore-missing-imports

# Auto-format
format:
    cd backend && uv run ruff format . && uv run ruff check --fix .

# Run every gate that CI runs: backend lint + tests, frontend type-check
# + tests + build. The fast confidence check before pushing.
check:
    just lint
    just test
    cd frontend && npx tsc --noEmit && npm test -- --run && npm run build

# Stop all local services
stop:
    docker compose down

# ─── Langfuse ─────────────────────────────────────────────────────────────────

# Start Langfuse (Postgres + server)
langfuse-up:
    docker compose up langfuse-db langfuse-server -d
    @echo "Langfuse starting at http://localhost:3000 -- allow 20-30 seconds"

# Stop Langfuse
langfuse-down:
    docker compose stop langfuse-db langfuse-server

# Open Langfuse UI in the default browser
langfuse-open:
    open http://localhost:3000

# ─── n8n ──────────────────────────────────────────────────────────────────────

# Start n8n as a standalone container (UI at http://localhost:5678)
n8n-up:
    docker run -d --rm \
      --name n8n \
      -p 5678:5678 \
      -e GENERIC_TIMEZONE="Europe/Warsaw" \
      -e TZ="Europe/Warsaw" \
      -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true \
      -e N8N_RUNNERS_ENABLED=true \
      -e N8N_SECURE_COOKIE=false \
      -v n8n_data:/home/node/.n8n \
      docker.n8n.io/n8nio/n8n
    @echo "n8n UI: http://localhost:5678"

# Stop the n8n container
n8n-down:
    docker stop n8n

# Tail n8n logs
n8n-logs:
    docker logs n8n -f

# Open n8n UI in the browser
n8n-open:
    open http://localhost:5678

# ─── Tailscale ────────────────────────────────────────────────────────────────

# Configure Tailscale Serve to proxy tailnet HTTPS to Vite dev server
tailscale-setup:
    tailscale serve https / http://localhost:5173
    @echo "App available at https://$(tailscale status --json | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"Self\"][\"DNSName\"].rstrip(\".\"))')"

# Show current Tailscale Serve status
tailscale-status:
    tailscale serve status

# Remove Tailscale Serve configuration
tailscale-reset:
    tailscale serve reset

# Configure Tailscale Serve for tailnet access (alias for tailscale-setup)
dev-serve: tailscale-setup

# Enable Tailscale Funnel for temporary public internet access
funnel-start:
    tailscale funnel 443 on
    @echo "Funnel enabled -- app is publicly reachable"

# Disable Tailscale Funnel
funnel-stop:
    tailscale funnel 443 off
    @echo "Funnel disabled"
