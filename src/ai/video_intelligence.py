"""Shared Video Intelligence Service for SIGNASSIST.

Provides multimodal AI video analysis and research grounding using Google GenAI SDK.
Maintains strict separation between local 42-class ST-GCN predictions and AI-assisted interpretation.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from src.utils.config import AppConfig, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try importing google-genai
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False
    genai = None
    types = None


@dataclass
class VideoIntelligenceResult:
    """Structured result container for AI-assisted video interpretation and grounding."""

    status: str = "AI_ANALYSIS_UNAVAILABLE"  # "SUCCESS", "AI_ANALYSIS_UNAVAILABLE", "ERROR", "NO_ISL_DETECTED"
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    video_path: Optional[str] = None
    is_probably_isl: Optional[bool] = None
    isl_assessment: Optional[str] = None
    sign_segments: List[Dict[str, Any]] = field(default_factory=list)
    gloss_sequence: List[str] = field(default_factory=list)
    interpreted_sentence: Optional[str] = None
    spoken_audio_transcript: Optional[str] = None
    visible_text: Optional[str] = None
    contextual_summary: Optional[str] = None
    uncertainty_notes: Optional[str] = None
    evidence_sources: List[Dict[str, Any]] = field(default_factory=list)
    web_research_performed: bool = False
    local_model_result: Optional[Dict[str, Any]] = None
    timestamps: Dict[str, Any] = field(default_factory=dict)
    raw_provider_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provenance: str = "AI-assisted / unverified"

    def to_dict(self) -> Dict[str, Any]:
        """Converts result container to a standard serializable dictionary."""
        return asdict(self)


STRICT_ISL_PROMPT_VERSION = "v1.0"
STRICT_ISL_PROMPT = """You are assisting SIGNASSIST, an Indian Sign Language research system.

Analyze this video carefully.

Your task is NOT to guess.

First determine whether the video visibly contains Indian Sign Language (ISL), another sign language, ordinary gestures, or insufficient evidence.

If ISL appears:

1. Identify the signer(s).
2. Determine approximate sign/phrase boundaries.
3. Inspect the visual signing sequence.
4. For each identifiable segment provide:
   - start timestamp
   - end timestamp
   - observed visual description
   - likely ISL gloss
   - confidence
   - alternative interpretation if uncertain
5. Produce the most likely gloss sequence.
6. Produce a possible English interpretation only when sufficiently supported.
7. Clearly distinguish:
   - observed visual evidence
   - inferred gloss
   - inferred English meaning
8. If spoken audio exists, transcribe it separately.
9. If subtitles/captions/text exist, report them separately.
10. Do not use spoken audio to pretend that the visual signing means the same thing.
11. Do not assume English word order equals ISL grammar.
12. Do not invent signs that cannot be observed.
13. If a sign is unknown or ambiguous, explicitly mark it UNKNOWN/UNCERTAIN.
14. If the video is not actually ISL, say so.
15. If the visual quality is insufficient, say so.
16. Include timestamps for important observations.

Return your analysis as a valid JSON object matching the following structure:
{
  "is_probably_isl": true/false,
  "isl_assessment": "...",
  "sign_segments": [
    {
      "start_timestamp": "00:01",
      "end_timestamp": "00:03",
      "visual_description": "...",
      "likely_isl_gloss": "...",
      "confidence": 0.85,
      "alternative_interpretation": "..."
    }
  ],
  "gloss_sequence": ["GLOSS1", "GLOSS2"],
  "interpreted_sentence": "...",
  "spoken_audio_transcript": "...",
  "visible_text": "...",
  "contextual_summary": "...",
  "uncertainty_notes": "..."
}
"""


class VideoIntelligenceService:
    """Service handling Gemini multimodal video analysis, grounding, and response caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        cache_dir: Optional[Union[str, Path]] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.config = config or load_config()

        c_dir = cache_dir or Path("data/processed/ai_video_cache")
        self.cache_dir = Path(c_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        if self.api_key and _GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize GenAI client: {e}")

    def _compute_sha256(self, video_path: Path) -> str:
        """Computes SHA-256 hash of a local video file."""
        sha = hashlib.sha256()
        with open(video_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def _get_cache_path(self, sha256_hash: str) -> Path:
        """Constructs deterministic cache file path."""
        safe_model = self.model_name.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{sha256_hash}_{safe_model}_{STRICT_ISL_PROMPT_VERSION}.json"

    def analyze_video(
        self,
        video_path: Union[str, Path],
        local_model_result: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        client_override: Optional[Any] = None,
    ) -> VideoIntelligenceResult:
        """Runs Gemini video intelligence analysis or retrieves cached analysis."""
        v_path = Path(video_path)
        if not v_path.exists():
            return VideoIntelligenceResult(
                status="ERROR",
                video_path=str(video_path),
                local_model_result=local_model_result,
                error=f"Video file does not exist: {video_path}",
            )

        if not self.api_key:
            logger.info("GEMINI_API_KEY missing; returning AI_ANALYSIS_UNAVAILABLE.")
            return VideoIntelligenceResult(
                status="AI_ANALYSIS_UNAVAILABLE",
                video_path=str(v_path),
                local_model_result=local_model_result,
                error="GEMINI_API_KEY environment variable is not configured.",
            )

        if not _GENAI_AVAILABLE and client_override is None:
            return VideoIntelligenceResult(
                status="AI_ANALYSIS_UNAVAILABLE",
                video_path=str(v_path),
                local_model_result=local_model_result,
                error="google-genai SDK is not installed.",
            )

        # Check cache
        sha_hash = self._compute_sha256(v_path)
        cache_file = self._get_cache_path(sha_hash)

        if not force_refresh and cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                res = VideoIntelligenceResult(**cached_data)
                res.local_model_result = local_model_result
                logger.info(f"Loaded cached AI analysis from {cache_file.name}")
                return res
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")

        # Active Client
        active_client = client_override or self.client
        if active_client is None:
            return VideoIntelligenceResult(
                status="AI_ANALYSIS_UNAVAILABLE",
                video_path=str(v_path),
                local_model_result=local_model_result,
                error="GenAI client unavailable.",
            )

        start_t = time.time()
        uploaded_file = None

        try:
            # 1. Upload file via Files API
            logger.info(f"Uploading video {v_path.name} via Gemini Files API...")
            uploaded_file = active_client.files.upload(file=str(v_path))

            # 2. Wait for processing state to become ACTIVE
            poll_count = 0
            while uploaded_file.state.name == "PROCESSING" and poll_count < 30:
                time.sleep(2)
                uploaded_file = active_client.files.get(name=uploaded_file.name)
                poll_count += 1

            if uploaded_file.state.name != "ACTIVE":
                return VideoIntelligenceResult(
                    status="ERROR",
                    video_path=str(v_path),
                    local_model_result=local_model_result,
                    error=f"Video file processing failed with state: {uploaded_file.state.name}",
                )

            # 3. Request generation with search grounding
            config_kwargs = {}
            if types is not None:
                try:
                    config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                    config_kwargs["response_mime_type"] = "application/json"
                except Exception as ex:
                    logger.debug(f"Types configuration warning: {ex}")

            logger.info(f"Generating video interpretation using {self.model_name}...")
            response = active_client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, STRICT_ISL_PROMPT],
                config=types.GenerateContentConfig(**config_kwargs) if types and config_kwargs else None,
            )

            end_t = time.time()

            # Parse Structured Output
            res_text = response.text if hasattr(response, "text") else ""
            parsed = self._parse_json_response(res_text)

            # Process grounding sources
            evidence_sources = self._extract_grounding_sources(response)

            res = VideoIntelligenceResult(
                status="SUCCESS",
                provider="gemini",
                model=self.model_name,
                video_path=str(v_path),
                is_probably_isl=parsed.get("is_probably_isl"),
                isl_assessment=parsed.get("isl_assessment"),
                sign_segments=parsed.get("sign_segments", []),
                gloss_sequence=parsed.get("gloss_sequence", []),
                interpreted_sentence=parsed.get("interpreted_sentence"),
                spoken_audio_transcript=parsed.get("spoken_audio_transcript"),
                visible_text=parsed.get("visible_text"),
                contextual_summary=parsed.get("contextual_summary"),
                uncertainty_notes=parsed.get("uncertainty_notes"),
                evidence_sources=evidence_sources,
                web_research_performed=len(evidence_sources) > 0,
                local_model_result=local_model_result,
                timestamps={
                    "start_time": start_t,
                    "end_time": end_t,
                    "duration_sec": round(end_t - start_t, 2),
                },
                provenance="AI-assisted / unverified",
            )

            # Save to cache
            try:
                cache_file.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
            except Exception as ce:
                logger.warning(f"Could not save AI result to cache: {ce}")

            return res

        except Exception as err:
            logger.error(f"Error during Gemini video intelligence execution: {err}")
            return VideoIntelligenceResult(
                status="ERROR",
                video_path=str(v_path),
                local_model_result=local_model_result,
                error=str(err),
            )
        finally:
            # Clean up remote file if possible
            if uploaded_file and hasattr(active_client, "files"):
                try:
                    active_client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Safely parses JSON from response string, handling markdown code blocks."""
        if not text:
            return {}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            logger.warning(f"Could not parse raw AI response as JSON: {text[:100]}...")
            return {
                "contextual_summary": text,
                "uncertainty_notes": "Response could not be parsed as structured JSON.",
            }

    def _extract_grounding_sources(self, response: Any) -> List[Dict[str, Any]]:
        """Extracts Google Search grounding sources from response metadata."""
        sources = []
        try:
            candidates = getattr(response, "candidates", [])
            if candidates:
                cand = candidates[0]
                grounding_meta = getattr(cand, "grounding_metadata", None)
                if grounding_meta:
                    search_chunks = getattr(grounding_meta, "grounding_chunks", [])
                    for chunk in search_chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            sources.append({
                                "title": getattr(web, "title", "Web Resource"),
                                "url": getattr(web, "uri", ""),
                                "relevance": "Retrieved via Gemini Google Search Grounding",
                                "support_level": "Contextual evidence only",
                            })
        except Exception as e:
            logger.debug(f"Error extracting grounding metadata: {e}")
        return sources
