from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.config.models import WorklogAgentConfig
from app.core.graph.nodes import worklog_agent
from app.core.graph.nodes.worklog_agent import _extract_plan_key, _fetch_worklog_context
from app.core.graph.state import GraphState
from app.ports.worklog import WorklogPlan, WorklogPlanSummary


@pytest.fixture
def mock_worklog():
    wl = AsyncMock()
    wl.list_plans = AsyncMock(
        return_value=[
            WorklogPlanSummary(key="2026-W16", created_at="2026-04-14", total_hours=40.0),
            WorklogPlanSummary(key="2026-W17", created_at="2026-04-21", total_hours=37.5),
        ]
    )
    wl.get_plan = AsyncMock(
        return_value=WorklogPlan(
            key="2026-W16",
            created_at="2026-04-14",
            total_hours=40.0,
            entries=[{"issue": "PROJ-1", "hours": 8}],
        )
    )
    wl.generate_plan = AsyncMock(
        return_value=WorklogPlan(
            key="2026-W18",
            created_at="2026-04-28",
            total_hours=40.0,
            entries=[{"issue": "PROJ-2", "hours": 8}],
        )
    )
    return wl


# ---------------------------------------------------------------------------
# _extract_plan_key
# ---------------------------------------------------------------------------


def test_extract_plan_key_colon_syntax():
    assert _extract_plan_key("show me plan:2026-W16 details") == "2026-W16"


def test_extract_plan_key_bare_key():
    assert _extract_plan_key("show me 2026-W16") == "2026-W16"


def test_extract_plan_key_returns_none_for_no_match():
    assert _extract_plan_key("list all plans") is None


def test_extract_plan_key_ignores_jira_issue_ids():
    # PROJ-1 must not be mistaken for a plan key
    assert _extract_plan_key("generate a report from PROJ-1") is None


def test_extract_plan_key_ignores_version_strings():
    assert _extract_plan_key("upgrade to v2-feature branch") is None


def test_extract_plan_key_strips_trailing_punctuation():
    # Regex uses word boundary -- punctuation terminates the match cleanly
    assert _extract_plan_key("see 2026-W16, please") == "2026-W16"


# ---------------------------------------------------------------------------
# _fetch_worklog_context -- intent routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_context_explicit_generate(mock_worklog):
    result = await _fetch_worklog_context("generate a new plan for this week", mock_worklog)
    mock_worklog.generate_plan.assert_awaited_once()
    assert "generate_plan" in result
    assert "2026-W18" in result


@pytest.mark.asyncio
async def test_fetch_context_generate_does_not_trigger_on_read_queries(mock_worklog):
    # "generate a report FROM my plan" is a read intent, not create
    result = await _fetch_worklog_context("generate a report from my latest plan", mock_worklog)
    mock_worklog.generate_plan.assert_not_called()
    assert "list_plans" in result


@pytest.mark.asyncio
async def test_fetch_context_create_does_not_trigger_on_read_queries(mock_worklog):
    # "create a summary of plan 2026-W16" contains "create" but has an explicit key
    result = await _fetch_worklog_context("create a summary of plan 2026-W16", mock_worklog)
    mock_worklog.generate_plan.assert_not_called()
    mock_worklog.get_plan.assert_awaited_once_with("2026-W16")
    assert "get_plan" in result


@pytest.mark.asyncio
async def test_fetch_context_get_plan_by_key(mock_worklog):
    result = await _fetch_worklog_context("show me 2026-W16", mock_worklog)
    mock_worklog.get_plan.assert_awaited_once_with("2026-W16")
    assert "get_plan" in result
    assert "PROJ-1" in result


@pytest.mark.asyncio
async def test_fetch_context_list_plans_default(mock_worklog):
    result = await _fetch_worklog_context("what are my worklogs?", mock_worklog)
    mock_worklog.list_plans.assert_awaited_once()
    assert "list_plans" in result
    assert "2026-W16" in result
    assert "2026-W17" in result


# ---------------------------------------------------------------------------
# worklog_agent.run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worklog_agent_sets_final_answer(mock_llm, mock_worklog):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "You have 2 plans: W16 (40h) and W17 (37.5h).",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    )
    state = GraphState(query="show my worklog plans", route="worklog")
    config = WorklogAgentConfig()

    result = await worklog_agent.run(state, config=config, llm=mock_llm, worklog=mock_worklog)

    assert result["final_answer"] == "You have 2 plans: W16 (40h) and W17 (37.5h)."
    assert len(result["execution_trace"]) == 1
    assert result["execution_trace"][0].node == "worklog_agent"


@pytest.mark.asyncio
async def test_worklog_agent_appends_to_existing_trace(mock_llm, mock_worklog):
    from app.core.models.types import TraceEntry

    mock_llm.complete = AsyncMock(
        return_value={
            "text": "Here are your plans.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    prior_entry = TraceEntry(node="router", duration_ms=5.0, data={})
    state = GraphState(query="show plans", route="worklog", execution_trace=[prior_entry])
    config = WorklogAgentConfig()

    result = await worklog_agent.run(state, config=config, llm=mock_llm, worklog=mock_worklog)

    assert len(result["execution_trace"]) == 2
    assert result["execution_trace"][0].node == "router"
    assert result["execution_trace"][1].node == "worklog_agent"


@pytest.mark.asyncio
async def test_worklog_agent_generate_triggers_generate_plan(mock_llm, mock_worklog):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "Generated plan 2026-W18 with 40h.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 60, "output_tokens": 15},
        }
    )
    state = GraphState(query="generate a new plan for this week", route="worklog")
    config = WorklogAgentConfig()

    result = await worklog_agent.run(state, config=config, llm=mock_llm, worklog=mock_worklog)

    mock_worklog.generate_plan.assert_awaited_once()
    assert "Generated plan" in result["final_answer"]


@pytest.mark.asyncio
async def test_worklog_agent_passes_context_to_llm(mock_llm, mock_worklog):
    mock_llm.complete = AsyncMock(
        return_value={
            "text": "Here are your plans.",
            "tool_use": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 10},
        }
    )
    state = GraphState(query="show my worklogs", route="worklog")
    config = WorklogAgentConfig()

    await worklog_agent.run(state, config=config, llm=mock_llm, worklog=mock_worklog)

    call_kwargs = mock_llm.complete.call_args.kwargs
    assert call_kwargs["system"] == config.system_prompt
    user_message = call_kwargs["messages"][0]["content"]
    assert "list_plans" in user_message
