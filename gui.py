import os
import sys
import torch
import logging
from yt_dlp import YoutubeDL
import gradio as gr
import argparse
from audio_separator.separator import Separator
import numpy as np
import librosa
import soundfile as sf
from ensemble import ensemble_files  # ensemble.py'dan import
import shutil  # Klasör temizleme için eklendi

device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model dictionaries organized by category
ROFORMER_MODELS = {
    "Vocals": {
        'MelBand Roformer | Vocals by Kimberley Jensen': 'vocals_mel_band_roformer.ckpt',
        'MelBand Roformer | Vocals by becruily': 'mel_band_roformer_vocals_becruily.ckpt',
        'MelBand Roformer | Vocals Fullness by Aname': 'mel_band_roformer_vocal_fullness_aname.ckpt',
        'BS Roformer | Vocals by Gabox': 'bs_roformer_vocals_gabox.ckpt',
        'MelBand Roformer | Vocals by Gabox': 'mel_band_roformer_vocals_gabox.ckpt',
        'MelBand Roformer | Vocals FV1 by Gabox': 'mel_band_roformer_vocals_fv1_gabox.ckpt',
        'MelBand Roformer | Vocals FV2 by Gabox': 'mel_band_roformer_vocals_fv2_gabox.ckpt',
        'MelBand Roformer | Vocals FV3 by Gabox': 'mel_band_roformer_vocals_fv3_gabox.ckpt',
        'MelBand Roformer | Vocals FV4 by Gabox': 'mel_band_roformer_vocals_fv4_gabox.ckpt',
        'BS Roformer | Chorus Male-Female by Sucial': 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',
        'BS Roformer | Male-Female by aufr33': 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
    },
    "Instrumentals": {
        'MelBand Roformer Kim | Inst V1 by Unwa': 'melband_roformer_inst_v1.ckpt',
        'MelBand Roformer Kim | Inst V1 (E) by Unwa': 'melband_roformer_inst_v1e.ckpt',
        'MelBand Roformer Kim | Inst V2 by Unwa': 'melband_roformer_inst_v2.ckpt',
        'MelBand Roformer | Instrumental by becruily': 'mel_band_roformer_instrumental_becruily.ckpt',
        'MelBand Roformer | Instrumental by Gabox': 'mel_band_roformer_instrumental_gabox.ckpt',
        'MelBand Roformer | Instrumental 2 by Gabox': 'mel_band_roformer_instrumental_2_gabox.ckpt',
        'MelBand Roformer | Instrumental 3 by Gabox': 'mel_band_roformer_instrumental_3_gabox.ckpt',
        'MelBand Roformer | Instrumental Bleedless V1 by Gabox': 'mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt',
        'MelBand Roformer | Instrumental Bleedless V2 by Gabox': 'mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt',
        'MelBand Roformer | Instrumental Fullness V1 by Gabox': 'mel_band_roformer_instrumental_fullness_v1_gabox.ckpt',
        'MelBand Roformer | Instrumental Fullness V2 by Gabox': 'mel_band_roformer_instrumental_fullness_v2_gabox.ckpt',
        'MelBand Roformer | Instrumental Fullness V3 by Gabox': 'mel_band_roformer_instrumental_fullness_v3_gabox.ckpt',
        'MelBand Roformer | Instrumental Fullness Noisy V4 by Gabox': 'mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt',
        'MelBand Roformer | INSTV5 by Gabox': 'mel_band_roformer_instrumental_instv5_gabox.ckpt',
        'MelBand Roformer | INSTV5N by Gabox': 'mel_band_roformer_instrumental_instv5n_gabox.ckpt',
        'MelBand Roformer | INSTV6 by Gabox': 'mel_band_roformer_instrumental_instv6_gabox.ckpt',
        'MelBand Roformer | INSTV6N by Gabox': 'mel_band_roformer_instrumental_instv6n_gabox.ckpt',
        'MelBand Roformer | INSTV7 by Gabox': 'mel_band_roformer_instrumental_instv7_gabox.ckpt',
    },
    "InstVoc Duality": {
        'MelBand Roformer Kim | InstVoc Duality V1 by Unwa': 'melband_roformer_instvoc_duality_v1.ckpt',
        'MelBand Roformer Kim | InstVoc Duality V2 by Unwa': 'melband_roformer_instvox_duality_v2.ckpt',
    },
    "De-Reverb": {
        'BS-Roformer-De-Reverb': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
        'MelBand Roformer | De-Reverb by anvuew': 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
        'MelBand Roformer | De-Reverb Less Aggressive by anvuew': 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
        'MelBand Roformer | De-Reverb Mono by anvuew': 'dereverb_mel_band_roformer_mono_anvuew.ckpt',
        'MelBand Roformer | De-Reverb Big by Sucial': 'dereverb_big_mbr_ep_362.ckpt',
        'MelBand Roformer | De-Reverb Super Big by Sucial': 'dereverb_super_big_mbr_ep_346.ckpt',
        'MelBand Roformer | De-Reverb-Echo by Sucial': 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
        'MelBand Roformer | De-Reverb-Echo V2 by Sucial': 'dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt',
        'MelBand Roformer | De-Reverb-Echo Fused by Sucial': 'dereverb_echo_mbr_fused.ckpt',
    },
    "Denoise": {
        'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
        'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
        'MelBand Roformer | Denoise-Debleed by Gabox': 'mel_band_roformer_denoise_debleed_gabox.ckpt',
    },
    "Karaoke": {
        'Mel-Roformer-Karaoke-Aufr33-Viperx': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
        'MelBand Roformer | Karaoke by Gabox': 'mel_band_roformer_karaoke_gabox.ckpt',
    },
    "General Purpose": {
        'BS-Roformer-Viperx-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'BS-Roformer-Viperx-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
        'BS-Roformer-Viperx-1053': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
        'Mel-Roformer-Viperx-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
        'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
        'MelBand Roformer Kim | FT by unwa': 'mel_band_roformer_kim_ft_unwa.ckpt',
        'MelBand Roformer Kim | FT 2 by unwa': 'mel_band_roformer_kim_ft2_unwa.ckpt',
        'MelBand Roformer Kim | FT 2 Bleedless by unwa': 'mel_band_roformer_kim_ft2_bleedless_unwa.ckpt',
        'MelBand Roformer Kim | SYHFT by SYH99999': 'MelBandRoformerSYHFT.ckpt',
        'MelBand Roformer Kim | SYHFT V2 by SYH99999': 'MelBandRoformerSYHFTV2.ckpt',
        'MelBand Roformer Kim | SYHFT V2.5 by SYH99999': 'MelBandRoformerSYHFTV2.5.ckpt',
        'MelBand Roformer Kim | SYHFT V3 by SYH99999': 'MelBandRoformerSYHFTV3Epsilon.ckpt',
        'MelBand Roformer Kim | Big SYHFT V1 by SYH99999': 'MelBandRoformerBigSYHFTV1.ckpt',
        'MelBand Roformer Kim | Big Beta 4 FT by unwa': 'melband_roformer_big_beta4.ckpt',
        'MelBand Roformer Kim | Big Beta 5e FT by unwa': 'melband_roformer_big_beta5e.ckpt',
        'MelBand Roformer | Big Beta 6 by unwa': 'melband_roformer_big_beta6.ckpt',
        'MelBand Roformer | Aspiration by Sucial': 'aspiration_mel_band_roformer_sdr_18.9845.ckpt',
        'MelBand Roformer | Aspiration Less Aggressive by Sucial': 'aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt',
        'MelBand Roformer | Bleed Suppressor V1 by unwa-97chris': 'mel_band_roformer_bleed_suppressor_v1.ckpt',
    }
}

OUTPUT_FORMATS = ['wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a', 'aiff', 'ac3']

# CSS (değişmedi, aynı kalıyor)
CSS = """
/* İnce ve Ortalanmış Tema */
#app-container {
    max-width: 600px;
    width: 100%;
    margin: 0 auto;
    padding: 0.2rem;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    background-color: #2d0b0b;
    position: relative;
}
body {
    background: url('/content/logo.jpg') no-repeat center center fixed;
    background-size: cover;
    margin: 0;
    padding: 0;
    font-family: 'Poppins', sans-serif;
    color: #C0C0C0;
    display: flex;
    justify-content: center;
}
body::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(45, 11, 11, 0.85);
    z-index: -1;
}
.logo-container {
    position: fixed;
    top: 0.2rem;
    left: 50%;
    transform: translateX(-50%);
    z-index: 2000;
}
.logo-img {
    width: 60px;
    height: auto;
}
.header-text {
    text-align: center;
    padding: 2rem 0.3rem 0.3rem;
    color: #ff4040;
    font-size: 1.4rem;
    font-weight: 700;
    text-shadow: 0 0 5px rgba(255, 64, 64, 0.4);
    z-index: 1500;
}
.dubbing-theme {
    background: linear-gradient(to bottom, #800000, #2d0b0b);
    border-radius: 6px;
    padding: 0.4rem;
    box-shadow: 0 3px 10px rgba(255, 64, 64, 0.2);
    width: 100%;
}
.footer {
    text-align: center;
    padding: 0.2rem;
    color: #ff4040;
    font-size: 10px;
    position: fixed;
    bottom: 0;
    width: 100%;
    max-width: 600px;
    background: rgba(45, 11, 11, 0.7);
    z-index: 1001;
    left: 50%;
    transform: translateX(-50%);
}
button {
    background: #800000 !important;
    border: 1px solid #ff4040 !important;
    color: #C0C0C0 !important;
    border-radius: 4px !important;
    padding: 4px 8px !important;
    font-size: 0.75rem !important;
    transition: all 0.2s ease !important;
}
button:hover {
    transform: scale(1.03) !important;
    background: #ff4040 !important;
    box-shadow: 0 3px 12px rgba(255, 64, 64, 0.5) !important;
}
.compact-upload.horizontal {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    max-width: 200px !important;
    height: 28px !important;
    padding: 0 5px !important;
    border: 1px solid #ff4040 !important;
    background: rgba(128, 0, 0, 0.5) !important;
    border-radius: 4px !important;
    color: #C0C0C0 !important;
}
.compact-upload.horizontal:hover {
    border-color: #ff6b6b !important;
    background: rgba(128, 0, 0, 0.7) !important;
}
.compact-upload.horizontal button {
    padding: 2px 6px !important;
    font-size: 0.6rem !important;
    height: 20px !important;
    min-width: 40px !important;
}
.gr-tab {
    background: rgba(128, 0, 0, 0.5) !important;
    border-radius: 5px 5px 0 0 !important;
    padding: 0.3rem 0.6rem !important;
    margin: 0 1px !important;
    color: #C0C0C0 !important;
    border: 1px solid #ff4040 !important;
    z-index: 1500;
    font-size: 0.8rem !important;
}
.gr-tab-selected {
    background: #800000 !important;
    color: #ffffff !important;
    border: 1px solid #ff6b6b !important;
    box-shadow: 0 2px 5px rgba(255, 64, 64, 0.5) !important;
}
.compact-grid {
    gap: 0.15rem !important;
    max-height: 30vh;
    overflow-y: auto;
    padding: 0.3rem;
    background: rgba(128, 0, 0, 0.3) !important;
    border-radius: 5px;
    border: 1px solid #ff4040 !important;
    width: 100%;
}
.compact-dropdown {
    padding: 4px 6px !important;
    border-radius: 5px !important;
    border: 1px solid #ff4040 !important;
    background: rgba(128, 0, 0, 0.5) !important;
    color: #C0C0C0 !important;
    width: 100%;
    font-size: 0.8rem !important;
}
.gr-slider input[type="range"] {
    -webkit-appearance: none !important;
    width: 100% !important;
    height: 5px !important;
    background: #ff4040 !important;
    border-radius: 2px !important;
    outline: none !important;
}
.gr-slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    width: 12px !important;
    height: 12px !important;
    background: #800000 !important;
    border: 1px solid #ff6b6b !important;
    border-radius: 50% !important;
    cursor: pointer !important;
}
.gr-slider input[type="range"]::-moz-range-thumb {
    width: 12px !important;
    height: 12px !important;
    background: #800000 !important;
    border: 1px solid #ff6b6b !important;
    border-radius: 50% !important;
    cursor: pointer !important;
}
@media (max-width: 768px) {
    #app-container {
        max-width: 100%;
        padding: 0.1rem;
    }
    .header-text {
        font-size: 1.2rem;
        padding: 1.5rem 0.3rem 0.2rem;
    }
    .logo-img {
        width: 40px;
    }
    .compact-upload.horizontal {
        max-width: 100% !important;
    }
    .compact-grid {
        max-height: 25vh;
    }
    .footer {
        max-width: 100%;
    }
}
"""

# Fonksiyonlar
def download_audio(url, out_dir="ytdl"):
    if not url:
        raise ValueError("No URL provided.")
    
    # ytdl klasörünü temizle
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            info_dict = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

def roformer_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, output_dir, out_format, norm_thresh, amp_thresh, batch_size, exclude_stems="", progress=gr.Progress(track_tqdm=True)):
    if not audio:
        raise ValueError("No audio file provided.")
    
    # output klasörünü temizle
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio))[0]
    for category, models in ROFORMER_MODELS.items():
        if model_key in models:
            model = models[model_key]
            break
    else:
        raise ValueError(f"Model '{model_key}' not found.")
    
    logger.info(f"Separating {base_name} with {model_key}")
    try:
        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=model_dir,
            output_dir=output_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={"segment_size": seg_size, "override_model_segment_size": override_seg_size, "batch_size": batch_size, "overlap": overlap, "pitch_shift": pitch_shift}
        )
        progress(0.2, desc="Loading model...")
        separator.load_model(model_filename=model)
        progress(0.7, desc="Separating audio...")
        separation = separator.separate(audio)
        stems = [os.path.join(output_dir, file_name) for file_name in separation]
        
        # Exclude stems filtresi
        if exclude_stems.strip():
            excluded = [s.strip().lower() for s in exclude_stems.split(',')]
            filtered_stems = [stem for stem in stems if not any(ex in os.path.basename(stem).lower() for ex in excluded)]
            return filtered_stems[0] if filtered_stems else None, filtered_stems[1] if len(filtered_stems) > 1 else None
        return stems[0], stems[1] if len(stems) > 1 else None
    except Exception as e:
        logger.error(f"Separation failed: {e}")
        raise RuntimeError(f"Separation failed: {e}")

def auto_ensemble_process(audio, model_keys, seg_size, overlap, out_format, use_tta, model_dir, output_dir, norm_thresh, amp_thresh, batch_size, ensemble_method, exclude_stems="", weights=None, progress=gr.Progress()):
    if not audio or not model_keys:
        raise ValueError("Audio or models missing.")
    
    # output klasörünü temizle
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio))[0]
    logger.info(f"Ensemble for {base_name} with {model_keys}")
    
    all_stems = []  # Tüm modellerden kalan stem'ler burada toplanacak
    total_models = len(model_keys)
    
    # Her model için ayrıştırma yap
    for i, model_key in enumerate(model_keys):
        for category, models in ROFORMER_MODELS.items():
            if model_key in models:
                model = models[model_key]
                break
        else:
            continue
        
        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=model_dir,
            output_dir=output_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={"segment_size": seg_size, "overlap": overlap, "use_tta": use_tta, "batch_size": batch_size}
        )
        progress(0.1 + (0.4 / total_models) * i, desc=f"Loading {model_key}")
        separator.load_model(model_filename=model)
        progress(0.5 + (0.4 / total_models) * i, desc=f"Separating with {model_key}")
        separation = separator.separate(audio)
        stems = [os.path.join(output_dir, file_name) for file_name in separation]
        
        # Exclude stems filtresi
        if exclude_stems.strip():
            excluded = [s.strip().lower() for s in exclude_stems.split(',')]
            filtered_stems = [stem for stem in stems if not any(ex in os.path.basename(stem).lower() for ex in excluded)]
            all_stems.extend(filtered_stems)
        else:
            all_stems.extend(stems)  # Eğer exclude_stems yoksa tüm stem'leri al
    
    if not all_stems:
        raise ValueError("No valid stems for ensemble after exclusion.")
    
    # Weights kontrolü
    if weights is None or len(weights) != len(model_keys):
        weights = [1.0] * len(model_keys)
    
    # Tüm kalan stem'leri birleştir
    output_file = os.path.join(output_dir, f"{base_name}_ensemble_{ensemble_method}.{out_format}")
    ensemble_args = [
        "--files", *all_stems,
        "--type", ensemble_method,
        "--weights", *[str(w) for w in weights[:len(all_stems)]],  # Stem sayısına göre weights kes
        "--output", output_file
    ]
    progress(0.9, desc="Running ensemble...")
    ensemble_files(ensemble_args)
    
    progress(1.0, desc="Ensemble complete")
    return output_file, f"Ensemble completed with {ensemble_method}, excluded: {exclude_stems if exclude_stems else 'None'}"

def update_roformer_models(category):
    return gr.update(choices=list(ROFORMER_MODELS[category].keys()))

def update_ensemble_models(category):
    return gr.update(choices=list(ROFORMER_MODELS[category].keys()))

# Arayüzü bir fonksiyon olarak tanımla
def create_interface():
    with gr.Blocks(title="🎵 Audio-Separator 🎵", css=CSS, elem_id="app-container") as app:
        gr.Markdown("<h1 class='header-text'>🎵 Audio-Separator 🎵</h1>")
        
        with gr.Tabs():
            with gr.Tab("⚙️ Settings"):
                model_file_dir = gr.Textbox(value="/tmp/audio-separator-models/", label="📂 Model Cache", placeholder="/tmp/audio-separator-models/")
                output_dir = gr.Textbox(value="output", label="📤 Output Dir", placeholder="output")
                output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMATS, label="🎶 Format")
                norm_threshold = gr.Slider(0.1, 1, value=0.9, step=0.1, label="🔊 Norm Thresh")
                amp_threshold = gr.Slider(0.1, 1, value=0.6, step=0.1, label="📈 Amp Thresh")
                batch_size = gr.Slider(1, 16, value=1, step=1, label="⚡ Batch Size")

            with gr.Tab("🎤 Roformer"):
                roformer_audio = gr.Audio(label="🎧 Input Audio", type="filepath")
                roformer_exclude_stems = gr.Textbox(label="🚫 Exclude Stems (comma-separated)", placeholder="e.g., Vocals, Drums")
                roformer_category = gr.Dropdown(label="📚 Category", choices=list(ROFORMER_MODELS.keys()), value="General Purpose")
                roformer_model = gr.Dropdown(label="🛠️ Model", choices=list(ROFORMER_MODELS["General Purpose"].keys()))
                with gr.Row():
                    roformer_seg_size = gr.Slider(32, 4000, value=256, step=32, label="📏 Seg Size")
                    roformer_overlap = gr.Slider(2, 10, value=8, step=1, label="🔄 Overlap")
                with gr.Row():
                    roformer_pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="🎵 Pitch")
                    roformer_override_seg_size = gr.Checkbox(value=False, label="🔧 Override Seg")
                with gr.Row():
                    url_ro = gr.Textbox(label="🔗 URL", placeholder="Audio/Video URL")
                    download_roformer = gr.Button("⬇️ Download")
                roformer_button = gr.Button("✂️ Separate!", variant="primary")
                with gr.Row():
                    roformer_stem1 = gr.Audio(label="🎸 Stem 1", type="filepath", interactive=False)
                    roformer_stem2 = gr.Audio(label="🥁 Stem 2", type="filepath", interactive=False)

            with gr.Tab("🎚️ Auto Ensemble"):
                ensemble_audio = gr.Audio(label="🎧 Input Audio", type="filepath")
                ensemble_exclude_stems = gr.Textbox(label="🚫 Exclude Stems (comma-separated)", placeholder="e.g., Vocals, Drums")
                ensemble_category = gr.Dropdown(label="📚 Category", choices=list(ROFORMER_MODELS.keys()), value="Instrumentals")
                ensemble_models = gr.Dropdown(label="🛠️ Models", choices=list(ROFORMER_MODELS["Instrumentals"].keys()), multiselect=True)
                with gr.Row():
                    ensemble_seg_size = gr.Slider(32, 4000, value=256, step=32, label="📏 Seg Size")
                    ensemble_overlap = gr.Slider(2, 10, value=8, step=1, label="🔄 Overlap")
                with gr.Row():
                    ensemble_use_tta = gr.Checkbox(value=False, label="🔍 TTA")
                ensemble_method = gr.Dropdown(label="⚙️ Method", choices=['avg_wave', 'median_wave', 'max_wave', 'min_wave', 'avg_fft', 'median_fft', 'max_fft', 'min_fft'], value='avg_wave')
                ensemble_weights = gr.Textbox(label="⚖️ Weights (comma-separated)", placeholder="1.0, 1.0, ...", value="")
                with gr.Row():
                    url_ensemble = gr.Textbox(label="🔗 URL", placeholder="Audio/Video URL")
                    download_ensemble = gr.Button("⬇️ Download")
                ensemble_button = gr.Button("🎛️ Run Ensemble!", variant="primary")
                ensemble_output = gr.Audio(label="🎶 Output", type="filepath", interactive=False)
                ensemble_status = gr.Textbox(label="📢 Status", interactive=False)

        gr.HTML("<div class='footer'>Powered by Audio-Separator 🌟🎶</div>")

        # Event Handlers
        roformer_category.change(update_roformer_models, inputs=[roformer_category], outputs=[roformer_model])
        download_roformer.click(fn=download_audio, inputs=[url_ro], outputs=[roformer_audio])
        roformer_button.click(
            roformer_separator,
            inputs=[roformer_audio, roformer_model, roformer_seg_size, roformer_override_seg_size, roformer_overlap, roformer_pitch_shift, model_file_dir, output_dir, output_format, norm_threshold, amp_threshold, batch_size, roformer_exclude_stems],
            outputs=[roformer_stem1, roformer_stem2]
        )
        ensemble_category.change(update_ensemble_models, inputs=[ensemble_category], outputs=[ensemble_models])
        download_ensemble.click(fn=download_audio, inputs=[url_ensemble], outputs=[ensemble_audio])
        ensemble_button.click(
            lambda *args: auto_ensemble_process(
                *args[:-1],  # Tüm argümanlar hariç sonuncusu (weights hariç)
                weights=[float(w.strip()) for w in args[-1].split(',')] if args[-1] else None  # Weights'i parse et
            ),
            inputs=[ensemble_audio, ensemble_models, ensemble_seg_size, ensemble_overlap, output_format, ensemble_use_tta, model_file_dir, output_dir, norm_threshold, amp_threshold, batch_size, ensemble_method, ensemble_exclude_stems, ensemble_weights],
            outputs=[ensemble_output, ensemble_status]
        )
    
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Source Separation Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the UI on")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Ngrok token for tunneling")
    args = parser.parse_args()

    app = create_interface()
    if args.ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(args.ngrok_token)
        public_url = ngrok.connect(args.port, "http")
        print(f"Ngrok URL: {public_url}")
        app.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        app.launch(server_name="0.0.0.0", server_port=args.port, share=True)