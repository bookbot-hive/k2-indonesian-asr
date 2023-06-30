import librosa
import sherpa_ncnn
import os
import time
import gradio as gr
import numpy as np

from functools import lru_cache
from pathlib import Path
from huggingface_hub import Repository

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

language_to_models = {
    "id": ["bookbot/sherpa-ncnn-pruned-transducer-stateless7-streaming-id"],
}

language_choices = list(language_to_models.keys())

streaming_recognizer = None


def recognize(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    in_filename: str,
):
    recognizer = get_pretrained_model(
        repo_id,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )

    audio, sr = librosa.load(in_filename, sr=16_000)
    samples_per_read = int(0.32 * sr)
    recognized_text = ""

    for i in range(0, len(audio), samples_per_read):
        chunk = audio[i : i + samples_per_read]
        recognizer.accept_waveform(sr, chunk)
        transcript = recognizer.text
        if transcript:
            recognized_text = transcript

    tail_paddings = np.zeros(int(recognizer.sample_rate * 0.5), dtype=np.float32)
    recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)

    recognizer.input_finished()
    transcript = recognizer.text
    if transcript:
        recognized_text = transcript

    return recognized_text


@lru_cache(maxsize=10)
def get_pretrained_model(repo_id: str, decoding_method: str, num_active_paths: int):
    model_name = Path(repo_id.split("/")[-1])
    _ = Repository(
        local_dir=model_name,
        clone_from=repo_id,
        token=AUTH_TOKEN,
    )

    return sherpa_ncnn.Recognizer(
        tokens=str(model_name / "tokens.txt"),
        encoder_param=str(model_name / "encoder_jit_trace-pnnx.ncnn.param"),
        encoder_bin=str(model_name / "encoder_jit_trace-pnnx.ncnn.bin"),
        decoder_param=str(model_name / "decoder_jit_trace-pnnx.ncnn.param"),
        decoder_bin=str(model_name / "decoder_jit_trace-pnnx.ncnn.bin"),
        joiner_param=str(model_name / "joiner_jit_trace-pnnx.ncnn.param"),
        joiner_bin=str(model_name / "joiner_jit_trace-pnnx.ncnn.bin"),
        num_threads=os.cpu_count(),
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=30,
        rule2_min_trailing_silence=30,
        rule3_min_utterance_length=30,
    )


def process_uploaded_file(
    language: str,
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
    in_filename: str,
):
    return recognize(
        in_filename=in_filename,
        language=language,
        repo_id=repo_id,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )


def recognize_audio_from_mic(
    in_filename: str,
    state: str,
):
    audio, sr = librosa.load(in_filename, sr=16_000)
    streaming_recognizer.accept_waveform(sr, audio)
    time.sleep(0.32)
    transcript = streaming_recognizer.text
    if transcript:
        state = transcript
    return state, state


def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown.update(choices=choices, value=choices[0])
    raise ValueError(f"Unsupported language: {language}")


with gr.Blocks() as demo:
    gr.Markdown("# Automatic Speech Recognition with Next-gen Kaldi")

    language_radio = gr.Radio(
        label="Language", choices=language_choices, value=language_choices[0]
    )
    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][0],
    )

    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )

    decoding_method_radio = gr.Radio(
        label="Decoding method",
        choices=["greedy_search", "modified_beam_search"],
        value="greedy_search",
    )

    num_active_paths_slider = gr.Slider(
        minimum=1,
        value=4,
        step=1,
        label="Number of active paths for modified_beam_search",
    )

    with gr.Tab("File Upload"):
        uploaded_file = gr.Audio(
            source="upload",  # Choose between "microphone", "upload"
            type="filepath",
            label="Upload audio file",
        )
        uploaded_output = gr.Textbox(label="Recognized speech from uploaded file")
        with gr.Row():
            upload_button = gr.Button("Recognize audio")
            upload_clear_button = gr.ClearButton(
                components=[uploaded_file, uploaded_output]
            )

    with gr.Tab("Real-time Microphone Recognition"):
        if streaming_recognizer is None:
            streaming_recognizer = get_pretrained_model(
                model_dropdown.value,
                decoding_method_radio.value,
                num_active_paths_slider.value,
            )
            print("Model initialized!")

        state = gr.State(value="")
        mic_input_audio = gr.Audio(
            source="microphone",
            type="filepath",
            label="Upload audio file",
        )
        mic_text_output = gr.Textbox(label="Recognized speech from microphone")
        mic_input_audio.stream(
            fn=recognize_audio_from_mic,
            inputs=[mic_input_audio, state],
            outputs=[mic_text_output, state],
            show_progress=False,
        )
        with gr.Row():
            file_clear_button = gr.ClearButton(
                components=[mic_text_output, state]
            ).click(
                get_pretrained_model,
                inputs=[
                    model_dropdown,
                    decoding_method_radio,
                    num_active_paths_slider,
                ],
            )

    upload_button.click(
        process_uploaded_file,
        inputs=[
            language_radio,
            model_dropdown,
            decoding_method_radio,
            num_active_paths_slider,
            uploaded_file,
        ],
        outputs=uploaded_output,
    )


demo.launch(debug=True)
