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