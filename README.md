---

```markdown
# SESA-Fast-Separation 🎵

**SESA-Fast-Separation** is a powerful music source separation tool with a user-friendly web interface, designed to split audio files into vocals, instrumentals, and other components. It supports Roformer models and ensemble techniques, and includes a feature to download audio directly from YouTube, all wrapped in a sleek Gradio-based UI.

## Features
- **Roformer Separation**: Separate audio using Roformer models from various categories (Vocals, Instrumentals, De-Reverb, etc.).
- **Auto Ensemble**: Combine multiple models for enhanced results (average, median, max, min methods).
- **YouTube Support**: Download audio directly from a URL.
- **Customizable Settings**: Adjust segment size, overlap, pitch shift, and more.
- **Multiple Output Formats**: Export in WAV, MP3, FLAC, and other formats.
- **GPU Support**: Accelerated processing on CUDA-compatible devices.

## Requirements
- Python 3.8+
- CUDA (optional, for GPU support)
- Required Python libraries (listed below)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[your-username]/sesa-fast-separation.git
   cd sesa-fast-separation
   ```

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the following libraries manually:
   ```bash
   pip install torch yt-dlp gradio audio-separator numpy librosa soundfile
   ```

3. **Download Model Files**:
   - Place Roformer models in `/tmp/audio-separator-models/` or specify your own model directory.
   - Model files are listed in the `ROFORMER_MODELS` dictionary as `.ckpt` files.

## Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```
   - Runs on the default port `7860`. To specify a different port:
     ```bash
     python app.py
     ```
   - A public URL is generated by default with `share=True`. Alternatively, use ngrok for tunneling:
     ```

2. **Separate Audio**:
   - **Roformer Tab**: Upload an audio file or enter a YouTube URL, select a model, and click "Separate!".
   - **Auto Ensemble Tab**: Choose multiple models, select an ensemble method, and click "Run Ensemble!" for a combined result.
   - 

## Technical Details
- **Backend**: Utilizes the `audio-separator` library with Roformer models.
- **Frontend**: Built with Gradio and a custom dark-themed UI.
- **Ensemble**: Supports various merging methods via the `ensemble_files` module.
- **CSS**: Custom styling for a modern, dark-themed design.

## Contributing
1. Fork the repository.
2. Create a branch for your changes:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes and open a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- The [audio-separator]([https://github.com/nomadkaraoke/python-audio-separator]) team.
- Gradio and the open-source community.
```
