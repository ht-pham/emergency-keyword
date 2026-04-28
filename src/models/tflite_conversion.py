import torch
import torch.onnx
import numpy as np

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def convert_pytorch_to_tflite(model, model_name="model", input_shape=(1, 12, 298)):
    """
    Convert PyTorch model to formats suitable for Arduino/inference
    
    Args:
        model: PyTorch model (CNN or CNN_LSTM)
        model_name: Name for output files
        input_shape: Input tensor shape (batch, channels, time_steps)
    """
    
    # Step 1: Save PyTorch model (native format)
    print(f"\n{'='*60}")
    print(f"Step 1: Saving {model_name} PyTorch model...")
    print(f"{'='*60}")
    model_path = f"{model_name}_pytorch.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved: {model_path}")
    print(f"  Format: PyTorch state dict")
    print(f"  Load with: model.load_state_dict(torch.load('{model_path}'))")
    
    # Step 2: Export to ONNX (cross-platform format)
    if ONNX_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"Step 2: Converting {model_name} PyTorch → ONNX...")
        print(f"{'='*60}")
        try:
            dummy_input = torch.randn(*input_shape)
            torch.onnx.export(
                model,
                dummy_input,
                f"{model_name}.onnx",
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            print(f"✓ Saved: {model_name}.onnx")
            print(f"  Format: ONNX (portable across frameworks)")
        except Exception as e:
            print(f"⚠️  ONNX export failed: {e}")
    else:
        print(f"\n⚠️  ONNX not available - skipping ONNX export")
    
    # Step 3: Try TFLite conversion (optional, requires TensorFlow with onnx-tf)
    print(f"\n{'='*60}")
    print(f"Step 3: TFLite Conversion Status")
    print(f"{'='*60}")
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available. Skipping TFLite conversion.")
    else:
        print("⚠️  TFLite conversion from PyTorch ONNX requires onnx-tf library.")
        print("    This has compatibility issues with TensorFlow 2.14+")
        print("")
        print("RECOMMENDED APPROACH FOR ARDUINO:")
        print(f"  1. Use the PyTorch model ({model_name}_pytorch.pth)")
        print("     - Run inference using PyTorch directly on a local machine")
        print("     - Export final predictions as C header files with quantized values")
        print("")
        print(f"  2. Use the ONNX model ({model_name}.onnx)")
        print("     - Convert using an online tool or dedicated converter")
        print("     - Links: https://onnx.ai/onnx/operators/")
        print("")
        print("  3. Use TensorFlow Lite alternatives:")
        print("     - Consider using TensorFlow model zoo pre-trained models directly")
        print("     - Or use quantization-aware training from the start")
    
    return None