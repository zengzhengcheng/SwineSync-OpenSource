from pathlib import Path

import numpy as np
import onnxruntime as ort

from models.ECGFeatureExtractor import ECGFeatureExtractor


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ECGModelInference:
    def __init__(self, model_paths):
        self.providers = self._get_optimal_providers()
        self.sessions = {}

        for name, path in model_paths.items():
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.sessions[name] = ort.InferenceSession(
                model_path.read_bytes(),
                providers=self.providers,
            )

        self._cache_io_names()

    def _get_optimal_providers(self):
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        elif "DmlExecutionProvider" in available_providers:
            providers.append("DmlExecutionProvider")
        elif "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _cache_io_names(self):
        self.io_names = {}
        for model_type, session in self.sessions.items():
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            self.io_names[model_type] = {
                "input_name": inputs[0].name,
                "output_names": [output.name for output in outputs],
            }

    def predict(self, model_type, input_data):
        session = self.sessions[model_type]
        io_info = self.io_names[model_type]
        self._validate_input_shape(model_type, input_data)
        return session.run(
            io_info["output_names"],
            {io_info["input_name"]: input_data.astype(np.float32)},
        )

    def _validate_input_shape(self, model_type, input_data):
        expected_shapes = {
            "tragan": (None, 3, 15360),
            "trabase": (None, 3, 15360),
            "traganbase": (None, 4, 15360),
        }
        actual_shape = input_data.shape
        expected_shape = expected_shapes[model_type]
        if actual_shape[1] != expected_shape[1] or actual_shape[2] != expected_shape[2]:
            raise ValueError(
                f"{model_type} expected {expected_shape}, got {actual_shape}"
            )

    def getout(self, input_data):
        input_data = input_data.astype(np.float32)
        gan_output1, gan_output2 = self.predict("tragan", input_data)
        clean_ecg = np.copy(gan_output1)
        base_output = self.predict("trabase", input_data)[0]

        wavelet_feats_np = ECGFeatureExtractor.extract_wavelet_features(clean_ecg)
        gan1_sigmoid = sigmoid(gan_output2)
        binary_gan1 = np.where(gan1_sigmoid > 0.5, 1, 0)

        ganbase_input = np.concatenate((clean_ecg, wavelet_feats_np, gan_output2), axis=1)
        ganbase_input = ganbase_input.astype(np.float32)
        ganbase_output = self.predict("traganbase", ganbase_input)[0]

        base_sigmoid = sigmoid(base_output)
        ganbase_sigmoid = sigmoid(ganbase_output)
        binary_base = np.where(base_sigmoid > 0.5, 1, 0)
        binary_ganbase = np.where(ganbase_sigmoid > 0.5, 1, 0)
        combined = binary_base | binary_gan1 | binary_ganbase
        return clean_ecg, combined
