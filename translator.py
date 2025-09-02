import re
import srt
from pathlib import Path
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import time
import argparse
from langdetect import detect
import yaml
import json
import logging

LANG_CODE_MAP = {
    # English & European
    "english": "eng_Latn",
    "en": "eng_Latn",
    "spanish": "spa_Latn",
    "es": "spa_Latn",
    "french": "fra_Latn",
    "fr": "fra_Latn",
    "german": "deu_Latn",
    "de": "deu_Latn",
    "italian": "ita_Latn",
    "it": "ita_Latn",
    "portuguese": "por_Latn",
    "pt": "por_Latn",
    "russian": "rus_Cyrl",
    "ru": "rus_Cyrl",
    "ukrainian": "ukr_Cyrl",
    "uk": "ukr_Cyrl",
    "dutch": "nld_Latn",
    "nl": "nld_Latn",
    "polish": "pol_Latn",
    "pl": "pol_Latn",
    "swedish": "swe_Latn",
    "sv": "swe_Latn",
    "norwegian": "nob_Latn",
    "no": "nob_Latn",
    "danish": "dan_Latn",
    "da": "dan_Latn",
    "finnish": "fin_Latn",
    "fi": "fin_Latn",
    "icelandic": "isl_Latn",
    "is": "isl_Latn",
    "hungarian": "hun_Latn",
    "hu": "hun_Latn",
    "czech": "ces_Latn",
    "cs": "ces_Latn",
    "slovak": "slk_Latn",
    "sk": "slk_Latn",
    "slovenian": "slv_Latn",
    "sl": "slv_Latn",
    "greek": "ell_Grek",
    "el": "ell_Grek",

    # Arabic & Middle Eastern
    "arabic": "arb_Arab",      # Standard Arabic
    "ar": "arb_Arab",
    "egyptian_arabic": "arz_Arab",
    "levantine_arabic": "apc_Arab",
    "gulf_arabic": "acm_Arab",
    "maghrebi_arabic": "ary_Arab",
    "iraqi_arabic": "acq_Arab",
    "persian": "pes_Arab",
    "fa": "pes_Arab",
    "dari": "prs_Arab",
    "pashto": "pbt_Arab",
    "hebrew": "heb_Hebr",
    "he": "heb_Hebr",
    "turkish": "tur_Latn",
    "tr": "tur_Latn",
    "kurdish": "kmr_Latn",
    "kurmanji": "kmr_Latn",
    "sorani": "ckb_Arab",
    "urdu": "urd_Arab",
    "ur": "urd_Arab",

    # South & East Asian
    "hindi": "hin_Deva",
    "hi": "hin_Deva",
    "bengali": "ben_Beng",
    "bn": "ben_Beng",
    "tamil": "tam_Taml",
    "ta": "tam_Taml",
    "telugu": "tel_Telu",
    "te": "tel_Telu",
    "malayalam": "mal_Mlym",
    "ml": "mal_Mlym",
    "kannada": "kan_Knda",
    "kn": "kan_Knda",
    "marathi": "mar_Deva",
    "mr": "mar_Deva",
    "gujarati": "guj_Gujr",
    "pa": "pan_Guru",
    "punjabi": "pan_Guru",
    "nepali": "npi_Deva",
    "ne": "npi_Deva",
    "sinhala": "sin_Sinh",
    "si": "sin_Sinh",
    "thai": "tha_Thai",
    "th": "tha_Thai",
    "lao": "lao_Laoo",
    "lo": "lao_Laoo",
    "burmese": "mya_Mymr",
    "myanmar": "mya_Mymr",
    "my": "mya_Mymr",
    "chinese": "zho_Hans",
    "zh": "zho_Hans",
    "chinese_simplified": "zho_Hans",
    "chinese_traditional": "zho_Hant",
    "zh_tw": "zho_Hant",
    "japanese": "jpn_Jpan",
    "ja": "jpn_Jpan",
    "korean": "kor_Hang",
    "ko": "kor_Hang",
    "vietnamese": "vie_Latn",
    "vi": "vie_Latn",
    "indonesian": "ind_Latn",
    "id": "ind_Latn",
    "malay": "zsm_Latn",
    "ms": "zsm_Latn",
    "tagalog": "tgl_Latn",
    "filipino": "tgl_Latn",
    "tl": "tgl_Latn",

    # African
    "swahili": "swh_Latn",
    "sw": "swh_Latn",
    "yoruba": "yor_Latn",
    "yo": "yor_Latn",
    "igbo": "ibo_Latn",
    "ig": "ibo_Latn",
    "zulu": "zul_Latn",
    "zu": "zul_Latn",
    "xhosa": "xho_Latn",
    "xh": "xho_Latn",
    "hausa": "hau_Latn",
    "ha": "hau_Latn",
    "amharic": "amh_Ethi",
    "am": "amh_Ethi",

    # Others
    "esperanto": "epo_Latn",
    "eo": "epo_Latn",
    "latin": "lat_Latn" if "lat_Latn" in locals() else "ltz_Latn",
}

def load_config(path: str):
    if path.endswith((".yaml", ".yml")):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be YAML or JSON")

tokenizer = None
model = None

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))

def load_subtitles(folder: str) -> List[srt.Subtitle]:
    """Load all subtitles from folder (first .srt file for now)."""
    files = list(Path(folder).glob("*.srt"))
    if not files:
        raise FileNotFoundError("No .srt files found in folder")
    text = files[0].read_text(encoding="utf-8")
    return list(srt.parse(text))

def detect_subtitle_lang(subs):
    sample_text = " ".join(sub.content for sub in subs[:5])
    lang_code = detect(sample_text)
    return LANG_CODE_MAP.get(lang_code, None)

def create_smart_batches(subs: List[srt.Subtitle], tokenizer, max_tokens: int = 3500) -> List[List[srt.Subtitle]]:
    """
    Create batches of subtitle blocks without splitting, limited by max token budget.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for sub in subs:
        text = sub.content.strip()
        # Estimate how many tokens this subtitle will take
        token_len = len(tokenizer(text, return_tensors="pt").input_ids[0])

        # If adding this subtitle exceeds token budget → close current batch
        if current_batch and current_tokens + token_len > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(sub)
        current_tokens += token_len

    if current_batch:
        batches.append(current_batch)

    return batches

def normalize_slang(text: str, slang_map: dict) -> str:
    if not slang_map:
        return text
    words = text.split()
    return " ".join(slang_map.get(w.lower(), w) for w in words)

# Defer slang loading until after we know src_lang
def load_slang_dict(src_lang, config):
    slang_map = {}

    if not config.get("normalize_slang", False):
        # slang disabled → just return empty
        return slang_map

    slang_path = Path(config.get("slang_dict_path", "")) if config.get("slang_dict_path") else None

    if slang_path and slang_path.exists():
        with slang_path.open(encoding="utf-8") as f:
            slang_map = json.load(f)
        print(f"📖 Loaded slang dictionary from {slang_path} ({len(slang_map)} entries)")
    else:
        slang_lang = src_lang.split("_")[0] if src_lang else None
        if slang_lang:
            fallback_path = Path(f"slang_dicts/slang_{slang_lang}.json")
            if fallback_path.exists():
                with fallback_path.open(encoding="utf-8") as f:
                    slang_map = json.load(f)
                print(f"📖 Loaded slang dictionary for {slang_lang} ({len(slang_map)} entries)")
            else:
                print(f"⚠️ No slang dictionary found for {src_lang}, skipping normalization")
        else:
            print("⚠️ Slang normalization enabled but no src_lang available yet")

    return slang_map

def clean_for_model(text: str, config) -> Tuple[str, List[str]]:
    """
    Prepare subtitle text for translation while preserving:
      - HTML/format tags (<i>, <b>, etc.)
      - Hearing-impaired cues ([ ... ], ( ... ))
      - Speaker labels (ALL CAPS:)
    
    Returns:
        clean_text: text with cues removed for translation
        preserved: list of preserved tags/cues for later restoration
    """
    preserved = []

    # --- 1. Preserve format tags like <i>, <b> ---
    if config.get("preserve_tags", True):
        for match in re.findall(r"<[^>]+>", text):
            preserved.append(match)
        text = re.sub(r"<[^>]+>", "<<<TAG>>>", text)

    # --- 2. Preserve [sounds] ---
    if config.get("preserve_sounds", True):
        for match in re.findall(r"\[.*?\]", text):
            preserved.append(match)
        text = re.sub(r"\[.*?\]", "<<<SOUND>>>", text)

        for match in re.findall(r"\(.*?\)", text):
            preserved.append(match)
        text = re.sub(r"\(.*?\)", "<<<SOUND>>>", text)

    # --- 4. Preserve ALL CAPS: speaker labels ---
    if config.get("preserve_speaker_labels", True):
        for match in re.findall(r"^([A-Z][A-Z ]+):", text):
            preserved.append(match + ":")
        text = re.sub(r"^([A-Z][A-Z ]+):", "<<<NAME>>>", text)

    return text.strip(), preserved

def prepare_batch_for_model(batch: List[srt.Subtitle]):
    block_texts = []
    preserved_all = []
    for idx, sub in enumerate(batch, start=1):
        clean_text, preserved = clean_for_model(sub.content.strip(), config)
        block_marker = config.get("block_format", "[BLOCK{num:04d}]").format(num=idx)
        block_texts.append(f"{block_marker} {clean_text}")
        preserved_all.append(preserved)
    return block_texts, preserved_all

def restore_preserved(translated: str, preserved: List[str], original: str) -> str:
    """
    Restore preserved items (tags, sounds, names) back into the translated text
    at approximately the same relative positions as they were in the original.
    """
    result = translated
    if not preserved:
        return result

    # Find where markers were in the cleaned original
    markers = []
    temp = original
    for marker in ["<<<TAG>>>", "<<<SOUND>>>", "<<<NAME>>>"]:
        while marker in temp:
            idx = temp.index(marker)
            rel_pos = idx / len(original)  # relative position
            markers.append((rel_pos, marker))
            temp = temp.replace(marker, "<<<USED>>>", 1)

    # Sort markers by relative position
    markers.sort(key=lambda x: x[0])

    # Now reinsert preserved items at matching positions
    for (_, marker), item in zip(markers, preserved):
        if marker in result:
            result = result.replace(marker, item, 1)

    # If any preserved items remain unused, append them at the end (failsafe)
    while preserved:
        result += " " + preserved.pop(0)

    return result

def translate_batch(texts, src_lang="eng_Latn", tgt_lang="arb_Arab", max_length=256):
    """
    Translate a batch of texts using NLLB on GPU.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256   # smaller input truncation
    ).to(model.device)

    translated_tokens = model.generate(
        **inputs,
        max_new_tokens=config.get("max_output_tokens", 128),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def restore_to_srt(batches, translations, preserved_all, output_path):
    restored_subs = []
    counter = 1  # subtitle index
    
    for batch, batch_translations, batch_preserved in zip(batches, translations, preserved_all):
        for sub, trans_text, preserved in zip(batch, batch_translations, batch_preserved):
            # 1. Remove [BLOCKx] marker if present
            clean_translation = re.sub(r"\[BLOCK\d{4}\]\s*", "", trans_text).strip()
            
            # 2. Restore preserved tags (formatting, sounds, speaker labels)
            final_text = restore_preserved(clean_translation, preserved, sub.content.strip())
            
            # 3. Rebuild subtitle with original timing
            restored_subs.append(srt.Subtitle(
                index=counter,
                start=sub.start,
                end=sub.end,
                content=final_text
            ))
            counter += 1

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(restored_subs))
    
    print(f"✅ Restored subtitles saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate subtitles with NLLB")
    parser.add_argument("-s", "--src", help="Source language (if omitted, auto-detect)")
    parser.add_argument("-t", "--tgt", help="Target language (default: spa_Latn or config)")
    parser.add_argument("-i", "--input", help="Input subtitle file or folder")
    parser.add_argument("-o", "--output", help="Output subtitle file or folder")
    parser.add_argument("-c", "--config", help="Path to YAML/JSON config file")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config) if args.config else {}

    # --- Slang loading ---
    slang_map = {}

    # --- Source language ---
    if args.src:
        src_lang = LANG_CODE_MAP.get(args.src.lower(), args.src)
    elif "src_lang" in config:
        src_lang = LANG_CODE_MAP.get(config["src_lang"].lower(), config["src_lang"])
    else:
        src_lang = None

    # --- Target language ---
    if args.tgt:
        tgt_lang = LANG_CODE_MAP.get(args.tgt.lower(), args.tgt)
    elif "tgt_lang" in config:
        tgt_lang = LANG_CODE_MAP.get(config["tgt_lang"].lower(), config["tgt_lang"])
    else:
        tgt_lang = "spa_Latn"

    # --- Model settings ---
    model_name = config.get("model_name", "facebook/nllb-200-distilled-600M")
    dtype = torch.float16 if config.get("dtype", "float16") == "float16" else torch.float32
    device_map = config.get("device_map", "auto")

    print(f"Loading model {model_name}...")
    with tqdm(total=2, desc="Loading", unit="step") as pbar:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pbar.update(1)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map
        )
        pbar.update(1)
    print("✅ Model loaded successfully!")

    # Add special tokens for block markers
    special_tokens = [f"[BLOCK{i:04d}]" for i in range(1, 1001)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # --- Subtitles ---
    input_path = Path(args.input) if args.input else Path(config.get("input"))
    output_path = Path(args.output) if args.output else Path(config.get("output"))

    if not input_path or not output_path:
        raise ValueError("❌ You must provide input/output either via CLI or config")

    # Collect input files
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.glob("*.srt"))
        if not input_files:
            raise FileNotFoundError(f"❌ No .srt files found in {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=getattr(logging, config.get("log_level", "INFO")))
    logger = logging.getLogger(__name__)

    # Process each subtitle file
    for in_file in input_files:
        subs = list(srt.parse(in_file.read_text(encoding="utf-8")))

        # Auto-detect source language if none provided
        if not src_lang:
            detected = detect_subtitle_lang(subs)
            if detected:
                src_lang = detected
                print(f"🌍 Detected source language for {in_file.name}: {src_lang}")
            else:
                raise ValueError(f"❌ Could not detect source language for {in_file.name} and no --src provided")
        
        slang_map = load_slang_dict(src_lang, config)

        # Apply slang normalization
        if slang_map:
            for sub in subs:
                sub.content = normalize_slang(sub.content, slang_map)

        max_input_tokens = config.get("max_input_tokens", 3500)
        batches = create_smart_batches(subs, tokenizer, max_tokens=max_input_tokens)

        logger.info(f"📦 Processing {in_file.name} → {len(batches)} batches")

        all_translations = []
        all_preserved = []

        for i, batch in enumerate(tqdm(batches, desc=f"Translating {in_file.name}", unit="batch")):
            start_time = time.time()

            block_texts, preserved_all = prepare_batch_for_model(batch)
            translations = translate_batch(
                block_texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )

            elapsed = time.time() - start_time
            print(f"✅ Finished batch {i+1}/{len(batches)} in {elapsed:.2f} sec")

            all_translations.append(translations)
            all_preserved.append(preserved_all)

        # Decide output file
        if input_path.is_file():
            out_file = output_path
        else:
            out_file = output_path / in_file.name

        restore_to_srt(batches, all_translations, all_preserved, out_file)