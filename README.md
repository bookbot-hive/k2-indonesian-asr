# Kaldi 2.0 Indonesian ASR

<p align="center">
    <a href="https://github.com/bookbot-hive/k2-indonesian-asr/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/bookbot-hive/k2-indonesian-asr.svg?color=blue">
    </a>
    <a href="https://github.com/bookbot-hive/k2-indonesian-asr/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://discord.gg/gqwTPyPxa6">
        <img alt="chat on Discord" src="https://img.shields.io/discord/1001447685645148169?logo=discord">
    </a>
    <a href="https://huggingface.co/spaces/bookbot/k2-indonesian-asr">
        <img alt="HuggingFace Space" src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg">
    </a>
</p>

Indonesian speech/phoneme recognizer powered by Kaldi 2.0 (lhotse, icefall, sherpa). Trained on open source speech data. Deployable on Desktop (via Python/C++), web apps, iOS, and Android.

All models released here are trained on [icefall](https://github.com/bookbot-hive/icefall) (which runs on PyTorch) and are converted for deployment via [sherpa-ncnn](https://github.com/k2-fsa/sherpa-ncnn). Icefall is Kaldi 2.0 / Next-Gen Kaldi, and unifies the application of [k2](https://github.com/k2-fsa/k2) for finite state automata (FSA) and [lhotse](https://github.com/bookbot-hive/lhotse) (audio data-loading).

Through this repository, we aim to document and release our open source models for the public's use.

## Training Dataset

As of the time of writing, we use the following datasets to train our models:

- [Common Voice ID](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)
- [LibriVox Indonesia](https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia)
- [FLEURS ID](https://huggingface.co/datasets/google/fleurs)

Noticeably, these datasets only contain text annotations and do not contain phoneme annotations. We used [g2p ID](https://github.com/bookbot-kids/g2p_id) to phonemize those text annotations.

Moreover, LibriVox Indonesia's original annotation is written with old Indonesian Republican Spelling System (Edjaan Repoeblik). We pre-converted them into EYD (Ejaan yang Disempurnakan) via [Doeloe](https://github.com/bookbot-hive/Doeloe), before phonemizing them.


## Available Models

### Pruned Stateless Zipformer RNN-T Streaming ID (Phonemes)

| Model Format | Link                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Icefall      | [Pruned Stateless Zipformer RNN-T Streaming ID](https://huggingface.co/bookbot/pruned-transducer-stateless7-streaming-id)                         |
| Sherpa NCNN  | [Sherpa-ncnn Pruned Stateless Zipformer RNN-T Streaming ID](https://huggingface.co/bookbot/sherpa-ncnn-pruned-transducer-stateless7-streaming-id) |
| Sherpa ONNX  | TBA                                                                                                                                               |

**Results (PER)**

| Decoding             | LibriVox | FLEURS | Common Voice |
| -------------------- | :------: | :----: | :----------: |
| Greedy Search        |  4.87%   | 11.45% |    14.97%    |
| Modified Beam Search |  4.71%   | 11.25% |    14.31%    |
| Fast Beam Search     |  4.85%   | 12.55% |    14.89%    |

## Usage

There are [various ways to export and deploy](https://icefall.readthedocs.io/en/latest/model-export/index.html) these models for production. Sherpa (Kaldi 2.0's main deployment framework) also has various counterparts for running on NCNN and/or ONNX engines. Or, you can also directly use these models via icefall, but they require a working PyTorch installation and is unoptimized for production.

We will provide a few external links to [Sherpa](https://k2-fsa.github.io/sherpa/index.html)'s thorough documentation which you can follow. We will also provide usage examples for [Recognize a file](#example-recognize-a-file-python---sherpa-ncnn) and [Real-time recognition with a microphone](#example-real-time-recognition-with-a-microphone-python---sherpa-ncnn) in Python.

| Inference Framework | Platform | Language | Link                                                                 |
| ------------------- | -------- | -------- | -------------------------------------------------------------------- |
| Sherpa              | Desktop  | C++      | [Guide](https://k2-fsa.github.io/sherpa/cpp/installation/index.html) |
| Sherpa NCNN         | Desktop  | Python   | [Guide](https://k2-fsa.github.io/sherpa/ncnn/python/index.html)      |
| Sherpa NCNN         | Android  | Kotlin   | [Guide](https://k2-fsa.github.io/sherpa/ncnn/android/index.html)     |
| Sherpa NCNN         | iOS      | Swift    | [Guide](https://k2-fsa.github.io/sherpa/ncnn/ios/index.html)         |
| Sherpa ONNX         | Desktop  | Python   | [Guide](https://k2-fsa.github.io/sherpa/onnx/python/index.html)      |
| Sherpa ONNX         | Android  | Kotlin   | [Guide](https://k2-fsa.github.io/sherpa/onnx/android/index.html)     |
| Sherpa ONNX         | iOS      | Swift    | [Guide](https://k2-fsa.github.io/sherpa/onnx/ios/index.html)         |

## Example: Recognize a File (Python - Sherpa NCNN)

The following code is adapted from [this example](https://k2-fsa.github.io/sherpa/ncnn/python/index.html#recognize-a-file). View this example running in our [live demo](https://huggingface.co/spaces/bookbot/k2-indonesian-asr)!

```py
import wave
import numpy as np
import sherpa_ncnn

path = "./sherpa-ncnn-pruned-transducer-stateless7-streaming-id"

def main():
    recognizer = sherpa_ncnn.Recognizer(
        tokens=f"{path}/tokens.txt",
        encoder_param=f"{path}/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin=f"{path}/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param=f"{path}/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin=f"{path}/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param=f"{path}/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin=f"{path}/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )

    filename = ("path/to/your/audio.wav")
    with wave.open(filename) as f:
        assert f.getframerate() == recognizer.sample_rate, (
            f.getframerate(),
            recognizer.sample_rate,
        )
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / 32768

    recognizer.accept_waveform(recognizer.sample_rate, samples_float32)

    tail_paddings = np.zeros(int(recognizer.sample_rate * 0.5), dtype=np.float32)
    recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)

    recognizer.input_finished()
    print(recognizer.text)
```

## Example: Real-time Recognition with a Microphone (Python - Sherpa NCNN)

The following code is adapted from [this example](https://k2-fsa.github.io/sherpa/ncnn/python/index.html#real-time-recognition-with-a-microphone). View this example running in our [live demo](https://huggingface.co/spaces/bookbot/k2-indonesian-asr)!

```py
import sys
import sounddevice as sd
import sherpa_ncnn

path = "./sherpa-ncnn-pruned-transducer-stateless7-streaming-id"

def create_recognizer():
    recognizer = sherpa_ncnn.Recognizer(
        tokens=f"{path}/tokens.txt",
        encoder_param=f"{path}/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin=f"{path}/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param=f"{path}/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin=f"{path}/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param=f"{path}/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin=f"{path}/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    return recognizer


def main():
    print("Started! Please speak")
    recognizer = create_recognizer()
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)
            result = recognizer.text
            if last_result != result:
                last_result = result
                print(result)


if __name__ == "__main__":
    devices = sd.query_devices()
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')
    main()
```

## License

Our models and inference code are released with Apache-2.0 license. Common Voice and LibriVox Indonesia are released under Public Domain, [CC-0](https://creativecommons.org/share-your-work/public-domain/cc0/). FLEURS is licensed under the [Creative Commons license (CC-BY)](https://creativecommons.org/licenses/).

## References

```bibtex
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
```

```bibtex
@article{fleurs2022arxiv,
  title = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  url = {https://arxiv.org/abs/2205.12446},
  year = {2022},
}
```