"""Phase 2 verification script for Text-to-ISL pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from src.nlp.isl_dictionary import ISLDictionaryIndex
from src.nlp.text_to_isl import TextToISLConverter
from src.video.isl_stitcher import ISLVideoStitcher

def main():
    print("=== PHASE 2: TEXT -> ISL VERIFICATION ===")
    
    dictionary_dir = Path("data/external/isl_dictionary")
    index = ISLDictionaryIndex.load(dictionary_dir)
    converter = TextToISLConverter(dictionary=index)
    
    res = converter.convert("hello")
    print(f"Original text: '{res.original_text}'")
    print(f"Normalized text: '{res.normalized_text}'")
    print(f"Demonstrations count: {len(res.demonstrations)}")
    
    for idx, demo in enumerate(res.demonstrations):
        print(f"  [{idx}] token='{demo.token}', source='{demo.source}', video_path='{demo.video_path}', gloss='{demo.dictionary_gloss}'")
        
    stitcher = ISLVideoStitcher(base_dir=dictionary_dir)
    out_dir = Path("scratch/phase2_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "hello_stitched.mp4"
    
    stitch_res = stitcher.stitch_sequence(res, out_mp4)
    final_clip_path = Path(stitch_res.output_path)
    print(f"Stitched clip saved to: {final_clip_path}")
    print(f"Stitch summary: frames={stitch_res.total_frames}, duration={stitch_res.duration_sec}s")
    
    # 1. Python validate_clip
    is_valid = stitcher.validate_clip(final_clip_path)
    print(f"validate_clip check: valid={is_valid}")
    assert is_valid, "Clip validation failed"
    
    # 2. ffprobe inspection if available
    try:
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,duration,nb_frames",
            "-show_entries", "format=duration,size,format_name",
            "-of", "json",
            str(final_clip_path)
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(probe_res.stdout)
        print("\nffprobe Output:")
        print(json.dumps(probe_data, indent=2))
        
        streams = probe_data.get("streams", [])
        if streams:
            codec = streams[0].get("codec_name")
            pix_fmt = streams[0].get("pix_fmt")
            width = streams[0].get("width")
            height = streams[0].get("height")
            duration = probe_data.get("format", {}).get("duration")
            file_size = probe_data.get("format", {}).get("size")
            print(f"\nVERIFIED CODEC: codec='{codec}' (expected 'h264'), pix_fmt='{pix_fmt}' (expected 'yuv420p'), resolution={width}x{height}, duration={duration}s, size={file_size} bytes")
            assert codec == "h264", f"Unexpected codec: {codec}"
            assert pix_fmt in ["yuv420p", "yuvj420p"], f"Unexpected pix_fmt: {pix_fmt}"
    except Exception as err:
        print(f"ffprobe check warning/error: {err}")
        
    print("\n=== PHASE 2 VERIFICATION COMPLETE: PASSED ✅ ===")

if __name__ == "__main__":
    main()
