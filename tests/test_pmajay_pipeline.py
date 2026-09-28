"""
PM-AJAY Virtual Livelihood Assistant — Integration Test Suite
=============================================================

Tests the three pillars of the new architecture:

  1. Ingestion & Vector Search  — JSON document store populated from
     data/schemes.json and data/centers.json; BM25 keyword retrieval.

  2. 4-Turn State Machine       — Session tracking in _CALL_SESSIONS,
     correct intake questions per turn, personalised RAG query construction.

  3. Health Endpoint            — FastAPI /health confirms the service
     is running under its new identity.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import importlib
import re
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Thin wrapper so tests can call async code without pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# 1. Ingestion & Vector Search
# ===========================================================================

class TestIngestionAndSearch:
    """Verify the pure-Python JSON vector store is populated and searchable."""

    def test_store_has_data(self):
        """auto_initialize_vector_db should succeed and persist documents."""
        from app.services.ingestion import _store_has_data, auto_initialize_vector_db

        # Run initialisation (no-op if already done).
        run(auto_initialize_vector_db())

        assert _store_has_data(), (
            "Vector store is empty after initialization — check data/schemes.json "
            "and data/centers.json are present."
        )

    def test_scheme_overview_retrievable(self):
        """The PM-AJAY overview chunk must be returned for a direct scheme query."""
        from app.services.ingestion import query_vector_db

        results = run(query_vector_db("PM-AJAY scheme SC family income eligibility", n_results=5))

        assert results, "query_vector_db returned no results for PM-AJAY overview query"
        combined = " ".join(results).lower()
        assert "pm-ajay" in combined or "pradhan mantri" in combined, (
            f"Expected PM-AJAY scheme text in results, got: {results[0][:200]}"
        )

    def test_training_center_retrievable(self):
        """Training center records from centers.json should match location queries."""
        from app.services.ingestion import query_vector_db

        results = run(query_vector_db("skill training center Visakhapatnam ITI electrician", n_results=5))

        assert results, "No results for training center location query"
        combined = " ".join(results).lower()
        assert "visakhapatnam" in combined or "electrician" in combined or "iti" in combined, (
            f"Expected center data in results, got: {results[0][:200]}"
        )

    def test_livelihood_model_retrievable(self):
        """Livelihood model chunks (SEED / WOW / mushroom / poultry) must be returned."""
        from app.services.ingestion import query_vector_db

        results = run(query_vector_db("mushroom poultry dairy farming self employment", n_results=5))

        assert results, "No results for livelihood model query"
        combined = " ".join(results).lower()
        assert any(kw in combined for kw in ["mushroom", "poultry", "dairy", "seed", "model"]), (
            f"Expected livelihood model keyword in results, got: {results[0][:200]}"
        )

    def test_credit_scheme_retrievable(self):
        """NSFDC / MUDRA credit scheme details must be searchable."""
        from app.services.ingestion import query_vector_db

        results = run(query_vector_db("NSFDC loan credit SC beneficiary interest rate", n_results=5))

        assert results, "No results for credit scheme query"
        combined = " ".join(results).lower()
        assert any(kw in combined for kw in ["nsfdc", "loan", "credit", "mudra", "interest"]), (
            f"Expected credit scheme info in results, got: {results[0][:200]}"
        )

    def test_query_returns_at_most_n_results(self):
        """query_vector_db must respect the n_results cap."""
        from app.services.ingestion import query_vector_db

        for n in (1, 3, 5):
            results = run(query_vector_db("skill training trade employment", n_results=n))
            assert len(results) <= n, f"Expected ≤{n} results, got {len(results)}"

    def test_asset_subsidy_info_searchable(self):
        """Asset subsidy text (Rs 50,000 / 50%) should be findable."""
        from app.services.ingestion import query_vector_db

        results = run(query_vector_db("asset subsidy sewing machine e-rickshaw 50000", n_results=5))

        assert results, "No results for asset subsidy query"
        combined = " ".join(results).lower()
        assert any(kw in combined for kw in ["subsidy", "asset", "50,000", "50000", "rickshaw"]), (
            f"Expected subsidy info in results, got: {results[0][:200]}"
        )


# ===========================================================================
# 2. 4-Turn State Machine
# ===========================================================================

class TestExotelStateMachine:
    """Verify the PM-AJAY 4-turn intake state machine in exotel.py."""

    def setup_method(self):
        """Clear _CALL_SESSIONS before each test to avoid cross-test contamination."""
        from app.routes.exotel import _CALL_SESSIONS
        _CALL_SESSIONS.clear()

    # ── Helper ──────────────────────────────────────────────────────────────

    def _get_session(self, call_sid: str) -> dict:
        from app.routes.exotel import _get_session
        return _get_session(call_sid)

    def _intake_q(self, turn: int, lang: str = "en-IN") -> str:
        from app.routes.exotel import _intake_question
        return _intake_question(turn, lang)

    # ── Turn 0: greeting ────────────────────────────────────────────────────

    def test_turn0_question_asks_education(self):
        q = self._intake_q(0)
        assert q, "Turn-0 question is empty"
        lower = q.lower()
        assert any(kw in lower for kw in ["education", "school", "pass", "primary", "middle"]), (
            f"Turn-0 should ask about education level. Got: {q}"
        )

    def test_turn0_in_telugu(self):
        q = self._intake_q(0, "te-IN")
        assert q, "Turn-0 Telugu question is empty"
        # Should contain PM-AJAY reference
        assert "Kaushal" in q or "PM-AJAY" in q or "స్వాగతం" in q or "primary" in q.lower(), (
            f"Turn-0 Telugu question unexpected: {q}"
        )

    def test_turn0_in_hindi(self):
        q = self._intake_q(0, "hi-IN")
        assert q, "Turn-0 Hindi question is empty"
        assert "Kaushal" in q or "PM-AJAY" in q or "स्वागत" in q or "पढ़ाई" in q, (
            f"Turn-0 Hindi question unexpected: {q}"
        )

    def test_turn0_fallback_to_english_for_unknown_lang(self):
        """An unsupported language code must fall back to English question."""
        q = self._intake_q(0, "xx-XX")
        q_en = self._intake_q(0, "en-IN")
        assert q == q_en, (
            f"Unknown language should fall back to English. Got: {q!r}, expected: {q_en!r}"
        )

    # ── Turn 1: family trade ─────────────────────────────────────────────────

    def test_turn1_question_asks_family_trade(self):
        q = self._intake_q(1)
        assert q, "Turn-1 question is empty"
        lower = q.lower()
        assert any(kw in lower for kw in ["work", "trade", "family", "farming", "tailor", "construction"]), (
            f"Turn-1 should ask about family trade. Got: {q}"
        )

    # ── Turn 2: district ─────────────────────────────────────────────────────

    def test_turn2_question_asks_district(self):
        q = self._intake_q(2)
        assert q, "Turn-2 question is empty"
        lower = q.lower()
        assert any(kw in lower for kw in ["district", "live", "location", "center"]), (
            f"Turn-2 should ask about district. Got: {q}"
        )

    # ── Turn 3: employment preference ────────────────────────────────────────

    def test_turn3_question_asks_preference(self):
        q = self._intake_q(3)
        assert q, "Turn-3 question is empty"
        lower = q.lower()
        assert any(kw in lower for kw in ["wage", "employment", "business", "self", "job"]), (
            f"Turn-3 should ask about employment preference. Got: {q}"
        )

    # ── Session state progression ─────────────────────────────────────────────

    def test_new_session_starts_at_turn_0(self):
        sess = self._get_session("test_call_001")
        assert sess["turn"] == 0, f"New session should start at turn 0, got {sess['turn']}"

    def test_session_profile_fields_initialised_null(self):
        sess = self._get_session("test_call_002")
        profile = sess["profile"]
        for field in ["education", "family_trade", "district", "preference", "physical_constraint"]:
            assert field in profile, f"Missing profile field: {field}"
            assert profile[field] is None, f"Field {field} should be None initially"

    def test_session_persists_across_calls_to_get_session(self):
        sid = "test_call_003"
        sess1 = self._get_session(sid)
        sess1["turn"] = 2
        sess1["profile"]["education"] = "10th pass"

        sess2 = self._get_session(sid)
        assert sess2["turn"] == 2, "Session should persist turn state"
        assert sess2["profile"]["education"] == "10th pass", "Session should persist profile data"

    def test_different_call_sids_have_independent_sessions(self):
        sid_a, sid_b = "call_A", "call_B"
        sess_a = self._get_session(sid_a)
        sess_a["turn"] = 3

        sess_b = self._get_session(sid_b)
        assert sess_b["turn"] == 0, "Session B should be independent of A"

    # ── Recommendation query builder ──────────────────────────────────────────

    def test_recommendation_query_contains_profile_data(self):
        from app.routes.exotel import _build_recommendation_query

        profile = {
            "education": "10th pass",
            "family_trade": "electrical work",
            "district": "Visakhapatnam",
            "preference": "self-employment",
            "physical_constraint": None,
        }
        query = _build_recommendation_query(profile, "en-IN")

        assert "10th pass" in query, "Query should contain education level"
        assert "electrical work" in query, "Query should contain family trade"
        assert "Visakhapatnam" in query, "Query should contain district"
        assert "self-employment" in query, "Query should contain employment preference"
        assert "PM-AJAY" in query, "Query should reference PM-AJAY"

    def test_recommendation_query_mentions_disability_when_set(self):
        from app.routes.exotel import _build_recommendation_query

        profile = {
            "education": "5th pass",
            "family_trade": "farming",
            "district": "Ananthapuramu",
            "preference": "wage employment",
            "physical_constraint": "hearing impairment",
        }
        query = _build_recommendation_query(profile, "en-IN")
        assert "disability" in query.lower() or "physical" in query.lower(), (
            "Query should mention disability when physical_constraint is set"
        )

    def test_recommendation_query_skips_disability_when_none(self):
        from app.routes.exotel import _build_recommendation_query

        profile = {
            "education": "middle school",
            "family_trade": "farming",
            "district": "Guntur",
            "preference": "wage employment",
            "physical_constraint": None,
        }
        query = _build_recommendation_query(profile, "en-IN")
        assert "disability" not in query.lower(), (
            "Query must NOT mention disability when physical_constraint is None"
        )

    # ── All 4 turns covered ───────────────────────────────────────────────────

    def test_all_four_turns_have_non_empty_english_questions(self):
        for turn in range(4):
            q = self._intake_q(turn)
            assert q, f"Turn-{turn} English question is empty"
            assert len(q) > 10, f"Turn-{turn} question too short: {q!r}"

    def test_all_four_turns_have_non_empty_telugu_questions(self):
        for turn in range(4):
            q = self._intake_q(turn, "te-IN")
            assert q, f"Turn-{turn} Telugu question is empty"
            assert len(q) > 10, f"Turn-{turn} Telugu question too short: {q!r}"


# ===========================================================================
# 3. Health Endpoint
# ===========================================================================

class TestHealthEndpoint:
    """Verify the FastAPI /health endpoint reflects the new PM-AJAY identity."""

    def test_health_check_live_server(self):
        """If the dev server is running on localhost:8000, validate health JSON."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")

        async def _check():
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:8000/health")
                return resp

        try:
            resp = run(_check())
        except Exception as exc:
            pytest.skip(f"Dev server not reachable (start uvicorn first): {exc}")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data.get("status") == "ok", f"Expected status=ok, got {data}"
        assert data.get("service") == "Kaushal Vaani - Virtual Livelihood Assistant", (
            f"Wrong service name in health response: {data}"
        )

    def test_health_response_via_test_client(self):
        """Use FastAPI's TestClient so no live server is needed."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx / starlette TestClient not available")

        # Patch lifespan hooks to avoid external I/O during unit test
        with (
            patch("app.services.ingestion.auto_initialize_vector_db", new_callable=AsyncMock),
            patch("app.services.rag_middleware.warm_rag_index", new_callable=AsyncMock),
            patch("app.services.rag_middleware.close_groq_client", new_callable=AsyncMock),
            patch("app.services.sarvam.close_sarvam_client", new_callable=AsyncMock),
            patch("app.services.bhashini.close_bhashini_client", new_callable=AsyncMock),
        ):
            from app.main import app
            with TestClient(app, raise_server_exceptions=True) as client:
                resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "Kaushal Vaani - Virtual Livelihood Assistant"


# ===========================================================================
# 4. Rag Middleware Language Resolution
# ===========================================================================

class TestRagMiddlewareLanguage:
    """Verify language profile resolution and system prompt generation."""

    def test_english_profile_returned_for_en_in(self):
        from app.services.rag_middleware import get_language_profile
        profile = get_language_profile("en-IN")
        assert profile["name"] == "English"

    def test_telugu_profile_returned(self):
        from app.services.rag_middleware import get_language_profile
        profile = get_language_profile("te-IN")
        assert profile["name"] == "Telugu"

    def test_unknown_code_falls_back_to_english(self):
        from app.services.rag_middleware import get_language_profile
        profile = get_language_profile("xx-XX")
        assert profile["name"] == "English", f"Should default to English, got: {profile['name']}"

    def test_none_code_falls_back_to_english(self):
        from app.services.rag_middleware import get_language_profile
        profile = get_language_profile(None)
        assert profile["name"] == "English"

    def test_fallback_message_is_pm_ajay_not_ncert(self):
        """Fallback messages must not reference NCERT or academic subjects."""
        from app.services.rag_middleware import get_language_profile
        for lang in ["en-IN", "te-IN", "hi-IN", "ta-IN"]:
            profile = get_language_profile(lang)
            fallback = profile.get("fallback", "")
            assert "ncert" not in fallback.lower(), f"{lang} fallback mentions NCERT: {fallback}"
            assert "textbook" not in fallback.lower(), f"{lang} fallback mentions textbook: {fallback}"
            assert "cpu" not in fallback.lower(), f"{lang} fallback mentions CPU: {fallback}"

    def test_system_prompt_is_pm_ajay_not_academic(self):
        """The LLM system prompt must describe PM-AJAY counsellor, not Shiksha Vani tutor."""
        from app.services.rag_middleware import PMAJAY_VOICE_SYSTEM_PROMPT
        prompt_lower = PMAJAY_VOICE_SYSTEM_PROMPT.lower()

        # Must be PM-AJAY identity
        assert "pm-ajay" in prompt_lower or "livelihood" in prompt_lower, (
            "System prompt must reference PM-AJAY or livelihoods"
        )
        # Must NOT be old academic tutor
        assert "shiksha vani" not in prompt_lower, "System prompt still references Shiksha Vani"
        assert "cpu scheduling" not in prompt_lower, "System prompt still references CPU scheduling"
        assert "operating system" not in prompt_lower or "MoSJE" in PMAJAY_VOICE_SYSTEM_PROMPT, (
            "System prompt references OS concepts without PM-AJAY context"
        )

    def test_resolve_target_language_detects_telugu_script(self):
        from app.services.rag_middleware import _resolve_target_language
        # Telugu Unicode characters
        lang = _resolve_target_language("నాకు PM-AJAY గురించి చెప్పండి", "en-IN")
        assert lang == "te-IN", f"Should detect Telugu, got {lang}"

    def test_resolve_target_language_respects_detected_code(self):
        from app.services.rag_middleware import _resolve_target_language
        lang = _resolve_target_language("I want information about training", "hi-IN")
        assert lang == "hi-IN", f"Should preserve hi-IN, got {lang}"

    def test_language_mirroring(self):
        """Simulate an English user query and assert the response is strictly in English."""
        from app.services.rag_middleware import query_rag, _resolve_target_language

        english_query = "I want a wage job in IT or computers, I have completed 12th class in Visakhapatnam."
        lang = _resolve_target_language(english_query, "en-IN")
        assert lang == "en-IN", f"Expected language en-IN, got {lang}"

        # Test query_rag with language="en-IN"
        resp = run(query_rag(english_query, "test_mirror_session", language="en-IN"))
        assert resp, "Expected non-empty response from query_rag"

        # Assert returned response is in English (not forced into Telugu or Hindi characters)
        assert not re.search(r"[\u0C00-\u0C7F]", resp), f"Response contains Telugu characters: {resp}"
        assert not re.search(r"[\u0900-\u097F]", resp), f"Response contains Hindi characters: {resp}"
        assert any(kw in resp.lower() for kw in ["job", "course", "training", "pm-ajay", "associate", "security", "it", "computer", "salary", "scheme", "12th", "eligible"]), (
            f"Expected English response text, got: {resp}"
        )

    def test_build_livelihood_recommendation_contains_language_instructions(self):
        """build_livelihood_recommendation must format prompt with language instruction."""
        from app.services.rag_middleware import build_livelihood_recommendation

        prompt_en = build_livelihood_recommendation("I need training", ["Context chunk 1"], language="en-IN")
        assert "English" in prompt_en
        assert "Context chunk 1" in prompt_en

        prompt_te = build_livelihood_recommendation("నాకు శిక్షణ కావాలి", ["Context chunk 1"], language="te-IN")
        assert "Telugu" in prompt_te

    def test_detect_transcript_language(self):
        """detect_transcript_language must correctly identify English, Telugu, and Hindi transcripts."""
        from app.routes.exotel import detect_transcript_language

        assert detect_transcript_language("I want a wage job in IT or computers, I have completed 12th class in Visakhapatnam.") == "en-IN"
        assert detect_transcript_language("నాకు PM-AJAY లో కంప్యూటర్ ట్రైనింగ్ కావాలి") == "te-IN"
        assert detect_transcript_language("मुझे सिलाई और दर्जी का काम सीखना है") == "hi-IN"
        assert detect_transcript_language("mujhe tailoring kaam chahiye") == "hi-IN"
        assert detect_transcript_language("naaku tailoring training kaavali") == "te-IN"


# ===========================================================================
# 5. Zero-Delay Conversational Loop & District Resolution
# ===========================================================================

class TestZeroDelayTelephonyLoop:
    """Verify zero-holding prompt optimization, fast VAD, center matching, and SMS follow-up."""

    def test_no_fillers_directory(self):
        """No holding phrases or filler audio files should exist in app/assets/fillers."""
        from pathlib import Path
        fillers_dir = Path(__file__).resolve().parents[1] / "app" / "assets" / "fillers"
        if fillers_dir.exists():
            files = list(fillers_dir.glob("*.pcm"))
            assert len(files) == 0, f"Found lingering filler audio files: {files}"

    def test_fast_vad_threshold(self):
        """Silence timeout must be tuned to ~600ms - 800ms for fast turn completion."""
        from app.core.config import get_settings
        settings = get_settings()
        assert 0.5 <= settings.vad_silence_seconds <= 0.8, (
            f"VAD silence timeout should be in 600ms-800ms range, got {settings.vad_silence_seconds}"
        )

    def test_turn0_asks_name_and_education(self):
        from app.routes.exotel import _intake_question
        q = _intake_question(0, "en-IN")
        assert "name" in q.lower(), f"Turn 0 must ask for name: {q}"
        assert "education" in q.lower(), f"Turn 0 must ask for education: {q}"

    def test_turn1_asks_family_trade_experience(self):
        from app.routes.exotel import _intake_question
        q = _intake_question(1, "en-IN")
        assert any(kw in q.lower() for kw in ["work", "trade", "experience"]), f"Turn 1 must ask for trade/work: {q}"

    def test_turn2_asks_district_and_preference(self):
        from app.routes.exotel import _intake_question
        q = _intake_question(2, "en-IN")
        assert "district" in q.lower(), f"Turn 2 must ask for district: {q}"
        assert "wage" in q.lower() or "self-employment" in q.lower() or "job" in q.lower(), (
            f"Turn 2 must ask for preference: {q}"
        )

    def test_resolve_nearest_center_visakhapatnam(self):
        from app.routes.exotel import _resolve_nearest_center
        center = _resolve_nearest_center("Visakhapatnam", "self-employment")
        assert center is not None, "Failed to resolve center for Visakhapatnam"
        assert "Visakhapatnam" in center.get("center_name", "") or "Visakhapatnam" in center.get("district", "")
        assert "rseti" in center.get("type", "").lower() or "self" in center.get("type", "").lower()

    def test_resolve_nearest_center_ananthapuramu(self):
        from app.routes.exotel import _resolve_nearest_center
        center = _resolve_nearest_center("Ananthapuramu", "wage employment")
        assert center is not None, "Failed to resolve center for Ananthapuramu"
        assert "Ananthapuramu" in center.get("center_name", "") or "Ananthapuramu" in center.get("district", "")
        assert "iti" in center.get("type", "").lower()

    def test_send_sms_followup_handles_invalid_number_cleanly(self):
        from app.routes.exotel import send_sms_followup
        res = run(send_sms_followup("", "Test message"))
        assert res is False

    def test_send_sms_followup_dispatches_with_clean_number(self):
        from app.routes.exotel import send_sms_followup
        res = run(send_sms_followup("+919876543210", "Test recommendation message"))
        assert isinstance(res, bool)

    def test_processing_keepalive_streams_silence_only(self):
        """_stream_processing_keepalive must send only silence frames (b'\x00'), never voice audio."""
        from app.routes.exotel import _stream_processing_keepalive
        sent_chunks = []

        class MockSender:
            async def send_pcm(self, chunk, rate):
                sent_chunks.append(chunk)

        stop_event = asyncio.Event()

        async def _test():
            task = asyncio.create_task(
                _stream_processing_keepalive(MockSender(), 8000, stop_event, "test_call", "te-IN")
            )
            await asyncio.sleep(0.05)
            stop_event.set()
            await task

        run(_test())
        for chunk in sent_chunks:
            assert chunk == b"\x00" * len(chunk), "Keepalive must be pure silence!"

    def test_pcm_chunking_format(self):
        """Assert that audio chunks are correctly formatted as 16-bit PCM and divided into uniform 20ms blocks."""
        from app.routes.exotel import _exotel_frame_size, _stream_pcm_to_exotel, DEFAULT_TELEPHONY_FRAME_MS
        from app.services.sarvam import sarvam_audio_to_pcm16, pcm16_to_wav

        # 1. Verify frame size formula adheres to strict 20ms telephony standards
        # At 8kHz mono 16-bit: 8000 * 2 * 0.02 = 320 bytes
        assert _exotel_frame_size(8000, 20) == 320, "8kHz 20ms frame must be 320 bytes"
        # At 16kHz mono 16-bit: 16000 * 2 * 0.02 = 640 bytes
        assert _exotel_frame_size(16000, 20) == 640, "16kHz 20ms frame must be 640 bytes"

        # 2. Test sarvam_audio_to_pcm16 output format
        # 1 second of 8kHz 16-bit PCM = 16,000 bytes
        raw_pcm = b"\x01\x00\xfe\xff" * 4000
        wav_container = pcm16_to_wav(raw_pcm, sample_rate=8000)
        pcm_out = sarvam_audio_to_pcm16(wav_container, target_sample_rate=8000)

        assert pcm_out, "PCM output is empty"
        assert len(pcm_out) % 2 == 0, "PCM audio must be 16-bit aligned (even number of bytes)"
        assert not (len(pcm_out) >= 4 and pcm_out[:4] == b"RIFF"), "Output must be raw PCM without RIFF/WAV header"

        # 3. Test _stream_pcm_to_exotel chunking & transmission
        sent_frames = []

        class MockSender:
            async def send_pcm(self, chunk: bytes, rate: int):
                sent_frames.append(chunk)

        # 1000 bytes of audio at 8kHz: 320 byte chunks -> should produce 4 frames (padded to 320 each)
        test_audio = b"\x10\x20" * 500  # 1000 bytes
        frames_count = run(_stream_pcm_to_exotel(
            websocket=None,  # type: ignore
            stream_sid="test_stream_001",
            audio=test_audio,
            sample_rate=8000,
            sender=MockSender(),  # type: ignore
            frame_duration_ms=20,
        ))

        assert frames_count == 4, f"Expected 4 chunks for 1000 bytes @ 320 bytes/chunk, got {frames_count}"
        assert len(sent_frames) == 4
        for i, frame in enumerate(sent_frames):
            assert len(frame) == 320, f"Frame {i} size is {len(frame)}, expected 320 bytes"
            assert len(frame) % 2 == 0, "Frame must be 16-bit aligned"

    def test_barge_in_cancels_playback_stream_immediately(self):
        """Barge-in event must immediately abort _stream_pcm_to_exotel without sending remaining frames."""
        from app.routes.exotel import _stream_pcm_to_exotel

        sent_frames = []
        barge_in_event = asyncio.Event()

        class MockSender:
            async def send_pcm(self, chunk: bytes, rate: int):
                sent_frames.append(chunk)
                if len(sent_frames) == 2:
                    # Simulate user interrupt after 2 frames
                    barge_in_event.set()

        # 10 frames worth of audio (3200 bytes)
        long_audio = b"\x01\x02" * 1600
        frames_sent = run(_stream_pcm_to_exotel(
            websocket=None,  # type: ignore
            stream_sid="test_barge_in_stream",
            audio=long_audio,
            sample_rate=8000,
            sender=MockSender(),  # type: ignore
            barge_in_event=barge_in_event,
            frame_duration_ms=20,
        ))

        assert frames_sent < 10, f"Streaming did not stop on barge-in! Frames sent: {frames_sent}"
        assert len(sent_frames) <= 3, f"Too many frames sent after barge-in: {len(sent_frames)}"



# ===========================================================================
# 6. Vocational Trade Semantic Search & Sector Boosting
# ===========================================================================

class TestTradeSemanticSearch:
    """Verify keyword-to-sector boosting and suppression of irrelevant tech courses."""

    def test_vector_store_records_have_trade_keywords(self):
        """Every qualification in vector store must include clean searchable trade_keywords tags."""
        from app.services.ingestion import _get_docs
        docs = _get_docs()
        qual_docs = [d for d in docs if d.get("source") == "Qualifications.xlsx"]
        assert len(qual_docs) >= 1500, f"Expected >=1500 qualifications, got {len(qual_docs)}"
        for d in qual_docs[:50]:
            assert "trade_keywords" in d, f"Missing trade_keywords in {d.get('title')}"
            assert isinstance(d["trade_keywords"], list)

    def test_tailoring_query_returns_apparel_trades(self):
        """Tailoring query must strictly prioritize tailoring/apparel trades, not IT/tech."""
        from app.services.ingestion import query_vector_db
        res = run(query_vector_db("I want to learn tailoring and sewing clothes", n_results=5))
        assert res, "No results returned for tailoring query"
        combined = " ".join(res).lower()
        assert "tailor" in combined or "sewing" in combined or "apparel" in combined, (
            f"Expected tailoring/apparel in results, got: {res[0]}"
        )
        # Verify tech courses like IoT, Web Dev, Cyber Security are suppressed
        assert "cyber security" not in combined
        assert "web development" not in combined
        assert "iot" not in combined

    def test_agriculture_query_returns_farming_trades(self):
        """Farming/dairy query must prioritize agriculture and allied trades."""
        from app.services.ingestion import query_vector_db
        res = run(query_vector_db("dairy farming cattle and crop cultivation", n_results=5))
        assert res, "No results returned for farming query"
        combined = " ".join(res).lower()
        assert any(kw in combined for kw in ["farm", "dairy", "agriculture", "crop", "cultivat", "livestock"]), (
            f"Expected agriculture/dairy in results, got: {res[0]}"
        )
        assert "cyber security" not in combined
        assert "computer concepts" not in combined

    def test_solar_electrician_query_returns_electrical_trades(self):
        """Solar/electrician query must return Electronics & HW or Green Jobs."""
        from app.services.ingestion import query_vector_db
        res = run(query_vector_db("solar panel wiring and electrician training", n_results=5))
        assert res, "No results returned for solar query"
        combined = " ".join(res).lower()
        assert "solar" in combined or "electric" in combined, f"Expected solar/electric in results: {res[0]}"

    def test_telugu_tailoring_intent_boosting(self):
        """Telugu tailoring query must detect intent and boost tailoring roles."""
        from app.services.ingestion import query_vector_db
        res = run(query_vector_db("నాకు కుట్టుపని లేదా టెయిలరింగ్ లో ట్రైనింగ్ కావాలి", n_results=3))
        assert res, "No results for Telugu tailoring query"
        combined = " ".join(res).lower()
        assert "tailor" in combined or "sewing" in combined or "apparel" in combined, (
            f"Expected tailoring in Telugu query results: {res[0]}"
        )

    def test_hindi_tailoring_intent_boosting(self):
        """Hindi tailoring query must detect intent and boost tailoring roles."""
        from app.services.ingestion import query_vector_db
        res = run(query_vector_db("मुझे सिलाई और दर्जी का काम सीखना है", n_results=3))
        assert res, "No results for Hindi tailoring query"
        combined = " ".join(res).lower()
        assert "tailor" in combined or "sewing" in combined or "apparel" in combined, (
            f"Expected tailoring in Hindi query results: {res[0]}"
        )
