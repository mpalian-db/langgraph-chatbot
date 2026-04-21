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


def test_extract_plan_key_dash_with_digits():
    assert _extract_plan_key("show me 2026-W16") == "2026-W16"


def test_extract_plan_key_returns_none_for_no_match():
    assert _extract_plan_key("list all plans") is None


# ---------------------------------------------------------------------------
# _fetch_worklog_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_context_generate(mock_worklog):
    result = await _fetch_worklog_context("generate a new plan", mock_worklog)
    mock_worklog.generate_plan.assert_awaited_once()
    assert "generate_plan" in result
    assert "2026-W18" in result


@pytest.mark.asyncio
async def test_fetch_context_get_plan(mock_worklog):
    result = await _fetch_worklog_context("show plan:2026-W16 details", mock_worklog)
    mock_worklog.get_plan.assert_awaited_once_with("2026-W16")
    assert "get_plan" in result
    assert "PROJ-1" in result


@pytest.mark.asyncio
async def test_fetch_context_list_plans(mock_worklog):
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
