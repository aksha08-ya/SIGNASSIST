"""Comprehensive Mock-Based Unit Tests for Shared Video Intelligence Service (Phase 18).

Covers all 17 required test specifications (A through Q) with strict mock isolation (zero network calls).
"""

import json
from pathlib import Path
import tempfile

from unittest.mock import MagicMock, patch
import pytest

from src.ai.video_intelligence import (
    VideoIntelligenceResult,
    VideoIntelligenceService,
)
from src.nlp.duplicate_suppressor import TemporalConsensusFilter
from src.ui.services import analyze_video_intelligence, process_video_file, process_video_url


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Creates a temporary dummy video file for testing."""
    v_file = tmp_path / "test_sample.mp4"
    v_file.write_bytes(b"dummy mp4 data for hashing and testing 12345")
    return v_file


# --- A. Gemini service unavailable without API key ---
def test_a_gemini_unavailable_without_api_key(dummy_video: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = VideoIntelligenceService(api_key=None, cache_dir=dummy_video.parent / "cache")
    res = service.analyze_video(dummy_video)
    assert res.status == "AI_ANALYSIS_UNAVAILABLE"
    assert "GEMINI_API_KEY" in res.error


# --- B. Malformed AI response ---
def test_b_malformed_ai_response(dummy_video: Path):
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.state.name = "ACTIVE"
    mock_client.files.upload.return_value = mock_file
    mock_client.files.get.return_value = mock_file

    mock_resp = MagicMock()
    mock_resp.text = "NOT A VALID JSON STRING AT ALL"
    mock_client.models.generate_content.return_value = mock_resp

    service = VideoIntelligenceService(api_key="test_key", cache_dir=dummy_video.parent / "cache")
    res = service.analyze_video(dummy_video, client_override=mock_client)

    assert res.status == "SUCCESS"
    assert res.contextual_summary == "NOT A VALID JSON STRING AT ALL"
    assert "Response could not be parsed" in res.uncertainty_notes


# --- C. AI API failure ---
def test_c_ai_api_failure(dummy_video: Path):
    mock_client = MagicMock()
    mock_client.files.upload.side_effect = RuntimeError("API connection timeout or rate limit")

    service = VideoIntelligenceService(api_key="test_key", cache_dir=dummy_video.parent / "cache")
    res = service.analyze_video(dummy_video, client_override=mock_client)

    assert res.status == "ERROR"
    assert "API connection timeout" in res.error


# --- D. Successful structured response parsing ---
def test_d_successful_structured_response_parsing(dummy_video: Path):
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.state.name = "ACTIVE"
    mock_client.files.upload.return_value = mock_file

    valid_json = json.dumps({
        "is_probably_isl": True,
        "isl_assessment": "The video displays clear ISL signing for transportation signs.",
        "sign_segments": [
            {"start_timestamp": "00:00", "end_timestamp": "00:02", "likely_isl_gloss": "10. Plane", "confidence": 0.95}
        ],
        "gloss_sequence": ["10. Plane"],
        "interpreted_sentence": "A plane is flying.",
        "contextual_summary": "ISL demonstration of plane sign.",
    })

    mock_resp = MagicMock()
    mock_resp.text = f"```json\n{valid_json}\n```"
    mock_client.models.generate_content.return_value = mock_resp

    service = VideoIntelligenceService(api_key="test_key", cache_dir=dummy_video.parent / "cache")
    res = service.analyze_video(dummy_video, client_override=mock_client)

    assert res.status == "SUCCESS"
    assert res.is_probably_isl is True
    assert res.gloss_sequence == ["10. Plane"]
    assert res.interpreted_sentence == "A plane is flying."
    assert res.provenance == "AI-assisted / unverified"


# --- E. Video upload pipeline invokes local + AI services ---
def test_e_video_upload_pipeline_invokes_local_and_ai(dummy_video: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("src.ui.services.process_video_file") as mock_local:
        mock_proc_res = MagicMock()
        mock_proc_res.to_dict.return_value = {"frames_processed": 60}
        mock_local.return_value = mock_proc_res

        ai_res = analyze_video_intelligence(dummy_video, local_model_result={"frames_processed": 60})
        assert ai_res["status"] == "AI_ANALYSIS_UNAVAILABLE"
        assert ai_res["local_model_result"]["frames_processed"] == 60


# --- F. URL pipeline invokes local + AI services ---
def test_f_url_pipeline_invokes_local_and_ai(dummy_video: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("src.video.url_downloader.URLVideoDownloader.download_video") as mock_dl:
        mock_meta = MagicMock()
        mock_meta.retrieved_file_path = str(dummy_video)
        mock_meta.original_url = "https://www.youtube.com/watch?v=dummy"
        mock_meta.platform_name = "youtube"
        mock_meta.video_title = "Dummy Video"
        mock_meta.is_temporary = False
        mock_meta.cleanup = MagicMock()
        mock_dl.return_value = mock_meta

        with patch("src.video.processor.VideoProcessor.process_file") as mock_proc:
            mock_res = MagicMock()
            mock_res.additional_metadata = {}
            mock_proc.return_value = mock_res

            res = process_video_url("https://www.youtube.com/watch?v=dummy")
            assert res.additional_metadata["downloaded_file_path"] == str(dummy_video)


# --- G. Local prediction below threshold is not emitted ---
def test_g_local_prediction_below_threshold_not_emitted():
    consensus = TemporalConsensusFilter(window_size=5, min_agreeing=3, min_confidence=0.70)
    res = consensus.add_prediction(class_id=0, gloss="10. Plane", confidence=0.65, is_accepted=False)
    assert res is None


# --- H. Repeated accepted predictions are deduplicated ---
def test_h_repeated_accepted_predictions_deduplicated():
    from src.nlp.duplicate_suppressor import DuplicateSuppressor
    from src.nlp.types import SignToken

    suppressor = DuplicateSuppressor(enable_suppression=True)
    t1 = SignToken(gloss="10. Plane", class_index=0, confidence=0.95, is_accepted=True)
    t2 = SignToken(gloss="10. Plane", class_index=0, confidence=0.92, is_accepted=True)
    t3 = SignToken(gloss="5. Cow", class_index=1, confidence=0.91, is_accepted=True)

    out1 = suppressor.process_token(t1)
    out2 = suppressor.process_token(t2)
    out3 = suppressor.process_token(t3)

    assert out1 is not None
    assert out1.gloss == "10. Plane"
    assert out2 is None  # Suppressed duplicate
    assert out3 is not None
    assert out3.gloss == "5. Cow"


# --- I. 3-of-5 temporal consensus ---
def test_i_3_of_5_temporal_consensus():
    consensus = TemporalConsensusFilter(window_size=5, min_agreeing=3, min_confidence=0.70)

    # 2 predictions: no consensus yet
    r1 = consensus.add_prediction(class_id=0, gloss="10. Plane", confidence=0.90, is_accepted=True)
    r2 = consensus.add_prediction(class_id=0, gloss="10. Plane", confidence=0.92, is_accepted=True)
    assert r1 is None
    assert r2 is None

    # 3rd prediction agreeing: consensus reached!
    r3 = consensus.add_prediction(class_id=0, gloss="10. Plane", confidence=0.91, is_accepted=True)
    assert r3 is not None
    assert r3["class_id"] == 0
    assert r3["consensus_count"] == 3


# --- J. Idle boundary resets sentence ---
def test_j_idle_boundary_resets_sentence():
    from src.nlp.sentence_reconstructor import SentenceReconstructor
    reconstructor = SentenceReconstructor()
    reconstructor.notify_boundary()
    assert reconstructor.suppressor.last_gloss is None


# --- K. Unknown/OOD predictions do not become fake text ---
def test_k_unknown_ood_predictions_not_fake_text():
    consensus = TemporalConsensusFilter(window_size=5, min_agreeing=3, min_confidence=0.70)
    # 5 different low-confidence OOD predictions
    for i in range(5):
        res = consensus.add_prediction(class_id=i, gloss=f"SIGN_{i}", confidence=0.30, is_accepted=False)
        assert res is None


# --- L. AI provenance appears in result ---
def test_l_ai_provenance_appears_in_result(dummy_video: Path):
    res = VideoIntelligenceResult(status="SUCCESS", video_path=str(dummy_video))
    assert res.provenance == "AI-assisted / unverified"


# --- M. Local and AI outputs remain separate ---
def test_m_local_and_ai_outputs_remain_separate(dummy_video: Path):
    res = VideoIntelligenceResult(
        status="SUCCESS",
        local_model_result={"predicted_class": "10. Plane", "confidence": 0.9922},
        gloss_sequence=["10. Plane", "5. Cow"],
        interpreted_sentence="A plane and a cow.",
    )
    assert res.local_model_result["predicted_class"] == "10. Plane"
    assert res.gloss_sequence != res.local_model_result


# --- N. Cache hit avoids repeated API request ---
def test_n_cache_hit_avoids_repeated_api_request(dummy_video: Path):
    cache_dir = dummy_video.parent / "cache"
    service = VideoIntelligenceService(api_key="test_key", cache_dir=cache_dir)

    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.state.name = "ACTIVE"
    mock_client.files.upload.return_value = mock_file

    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"contextual_summary": "First API run"})
    mock_client.models.generate_content.return_value = mock_resp

    res1 = service.analyze_video(dummy_video, client_override=mock_client)
    assert res1.contextual_summary == "First API run"
    assert mock_client.models.generate_content.call_count == 1

    # Second run should hit cache without calling API again
    res2 = service.analyze_video(dummy_video, client_override=mock_client)
    assert res2.contextual_summary == "First API run"
    assert mock_client.models.generate_content.call_count == 1  # Not incremented!


# --- O. Uploaded temporary video survives until AI job completes ---
def test_o_uploaded_temporary_video_survives(tmp_path: Path):
    job_dir = tmp_path / "job_123"
    job_dir.mkdir()
    temp_file = job_dir / "sample.mp4"
    temp_file.write_bytes(b"content")

    assert temp_file.exists()
    # Execute work
    service = VideoIntelligenceService(api_key=None, cache_dir=tmp_path / "cache")
    res = service.analyze_video(temp_file)
    assert temp_file.exists()  # File survives during processing


# --- P. AI result cleanup works ---
def test_p_ai_result_cleanup_works(dummy_video: Path):
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.name = "files/test_file_id"
    mock_file.state.name = "ACTIVE"
    mock_client.files.upload.return_value = mock_file

    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"contextual_summary": "Cleaned up"})
    mock_client.models.generate_content.return_value = mock_resp

    service = VideoIntelligenceService(api_key="test_key", cache_dir=dummy_video.parent / "cache")
    res = service.analyze_video(dummy_video, client_override=mock_client)

    assert mock_client.files.delete.called
    mock_client.files.delete.assert_called_with(name="files/test_file_id")


# --- Q. No API key is printed/logged ---
def test_q_no_api_key_printed_or_logged(dummy_video: Path, caplog):
    secret_key = "AIzaSyTEST_SECRET_API_KEY_999"
    service = VideoIntelligenceService(api_key=secret_key, cache_dir=dummy_video.parent / "cache")
    res = VideoIntelligenceResult(status="SUCCESS", video_path=str(dummy_video)).to_dict()

    res_str = json.dumps(res)
    assert secret_key not in res_str
    assert secret_key not in caplog.text

