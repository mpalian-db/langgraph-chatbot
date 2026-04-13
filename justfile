set dotenv-load := true

# Start Qdrant + Langfuse + backend + frontend for local dev
dev:
    docker compose up qdrant langfuse-db langfuse-server -d
    cd backend && uv run uvicorn app.main:app --reload --port 8000 &
    cd frontend && npm run dev

# Start full stack including n8n
dev-full:
    docker compose up qdrant langfuse-db langfuse-server n8n -d
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

# Lint (no auto-fix)
lint:
    cd backend && uv run ruff check . && uv run ruff format --check .

# Auto-format
format:
    cd backend && uv run ruff format . && uv run ruff check --fix .

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

# Start the n8n container (UI at http://localhost:5678)
n8n-up:
    docker compose up n8n -d
    @echo "n8n UI: http://localhost:5678"

# Stop the n8n container
n8n-down:
    docker compose stop n8n

# Tail n8n logs
n8n-logs:
    docker compose logs n8n -f

# Open n8n UI in the browser
n8n-open:
    open http://localhost:5678

# Export all n8n workflows to n8n/workflows/ as JSON
n8n-export:
    docker compose exec n8n n8n export:workflow --all --output=/home/node/workflows/
    @echo "Workflows exported to n8n/workflows/"

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
