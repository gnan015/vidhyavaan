"""Automated verification script for Sarvam AI STT & TTS pipeline with RAG."""

import asyncio
from app.services.sarvam import (
    synthesize_speech_sarvam,
    transcribe_audio_sarvam,
    sarvam_audio_to_pcm16,
)
from app.services.rag_middleware import query_rag


async def run_checks() -> None:
    print("--- 1. Testing Sarvam Telugu TTS ---")
    te_text = "నమస్కారం, విద్యావాణి వాయిస్‌బాట్‌కి స్వాగతం."
    te_wav = await synthesize_speech_sarvam(te_text, target_language_code="te-IN")
    assert len(te_wav) > 1000, "TTS audio WAV too small"
    te_pcm = sarvam_audio_to_pcm16(te_wav, target_sample_rate=8000)
    print(f"Telugu WAV: {len(te_wav)} bytes | PCM: {len(te_pcm)} bytes")
    assert len(te_pcm) > 0, "PCM audio empty"

    print("\n--- 2. Testing Sarvam Telugu STT ---")
    stt_res = await transcribe_audio_sarvam(te_wav, language_code="te-IN")
    print(f"Transcript: {stt_res['transcript']}")
    print(f"Language: {stt_res['language_code']}")
    assert len(stt_res["transcript"]) > 0, "Transcript empty"

    print("\n--- 3. Testing RAG + Sarvam Full Flow ---")
    user_query = "What is CPU scheduling?"
    rag_answer = await query_rag(user_query, session_id="test-session", language_code="te-IN")
    print(f"RAG Answer: {rag_answer[:120]}...")
    assert len(rag_answer) > 0, "RAG answer empty"

    rag_wav = await synthesize_speech_sarvam(rag_answer, target_language_code="te-IN")
    rag_pcm = sarvam_audio_to_pcm16(rag_wav, target_sample_rate=8000)
    print(f"Synthesized RAG audio: {len(rag_wav)} WAV bytes | {len(rag_pcm)} PCM frames")
    assert len(rag_pcm) > 0, "RAG PCM empty"

    print("\n======================================")
    print("ALL SARVAM VERIFICATION CHECKS PASSED!")
    print("======================================")


if __name__ == "__main__":
    asyncio.run(run_checks())
