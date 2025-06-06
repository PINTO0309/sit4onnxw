# sit4onnxw

Simple Inference Test for ONNX Runtime Web

https://github.com/PINTO0309/simple-onnx-processing-tools

## Overview

sit4onnxw is a comprehensive Python tool for benchmarking ONNX models using ONNX Runtime Web with support for CPU, WebGL, and WebGPU execution providers. This tool is inspired by [sit4onnx](https://github.com/PINTO0309/sit4onnx) but specifically designed to work with onnxruntime-web through browser automation.

**Key Differentiators:**
- **100% sit4onnx Compatible**: Same CLI interface and parameter specifications
- **Multi-Input Model Support**: Full support for complex models with multiple inputs/outputs
- **Dynamic Tensor Intelligence**: Automatic shape inference for dynamic dimensions
- **Robust Error Handling**: Categorized errors with actionable solutions
- **Browser Automation**: Leverages Selenium for reliable ONNX Runtime Web execution

## Features

### Core Functionality
- **Multiple Execution Providers**: CPU, WebGL, and WebGPU support with automatic fallback
- **Multi-Input Model Support**: Full support for models with multiple inputs and outputs
- **Dynamic Tensor Support**: Intelligent handling of dynamic dimensions with smart defaults
- **Model Format Support**: Both .onnx and .ort model formats

### Advanced Features
- **sit4onnx Compatible Interface**: Same parameter specification as original sit4onnx
- **Automatic Fallback**: WebGL/WebGPU failures automatically fallback to CPU
- **Smart Error Handling**: Categorized error messages with user-friendly suggestions
- **Performance Benchmarking**: Configurable test loops with detailed timing analysis
- **External Input Support**: Use your own numpy arrays as model inputs

### Usability
- **Flexible Shape Specification**: Multiple ways to specify input shapes
- **Comprehensive CLI**: Full command-line interface with all sit4onnx options
- **Python API**: Direct programmatic access for integration
- **Debug Mode**: Browser debugging support for troubleshooting

## Installation

```bash
pip install sit4onnxw
```

## Usage

### 1. Quick Start Examples

```bash
# Basic inference (CPU)
sit4onnxw --input_onnx_file_path model.onnx

# WebGL with automatic CPU fallback
sit4onnxw --input_onnx_file_path model.onnx --execution_provider webgl --fallback_to_cpu

# Dynamic tensor model with custom batch size
sit4onnxw --input_onnx_file_path dynamic_model.onnx --batch_size 4

# Multi-input model with fixed shapes (sit4onnx style)
sit4onnxw \
--input_onnx_file_path multi_input_model.onnx \
--fixed_shapes 1 64 112 200 \
--fixed_shapes 1 3 112 200 \
--execution_provider cpu

# Using external numpy input files
sit4onnxw \
--input_onnx_file_path multi_input_model.onnx \
--input_numpy_file_paths input1.npy \
--input_numpy_file_paths input2.npy \
--execution_provider cpu

# Performance benchmarking
sit4onnxw \
--input_onnx_file_path model.onnx \
--test_loop_count 100 \
--enable_profiling \
--output_numpy_file
```

### 2. CLI Usage

```bash
usage:
sit4onnxw [-h]
  --input_onnx_file_path INPUT_ONNX_FILE_PATH
  [--batch_size BATCH_SIZE]
  [--fixed_shapes FIXED_SHAPES]
  [--test_loop_count TEST_LOOP_COUNT]
  [--execution_provider {cpu,webgl,webgpu}]
  [--enable_profiling]
  [--output_numpy_file]
  [--numpy_seed NUMPY_SEED]
  [--input_numpy_file_paths INPUT_NUMPY_FILE_PATHS]
  [--input_names INPUT_NAMES]
  [--output_names OUTPUT_NAMES]
  [--ort_model_path ORT_MODEL_PATH]
  [--headless/--no-headless]
  [--timeout TIMEOUT]

optional arguments:
  -h, --help
    show this help message and exit
  --input_onnx_file_path INPUT_ONNX_FILE_PATH
    Path to ONNX model file
  --batch_size BATCH_SIZE
    Batch size for inference (default: 1)
  --fixed_shapes FIXED_SHAPES
    Input OPs with undefined shapes changed to specified shape. Can be specified multiple times for different input OPs.
  --test_loop_count TEST_LOOP_COUNT
    Number of times to run the test. Total execution time divided by test runs to show average inference time. (default: 10)
  --execution_provider {cpu,webgl,webgpu}
    ONNX Runtime Web Execution Provider. (default: cpu)
  --enable_profiling
    Outputs performance profiling result to a .json file
  --output_numpy_file
    Outputs the last inference result to an .npy file.
  --numpy_seed NUMPY_SEED
    Random seed for input data generation.
  --input_numpy_file_paths INPUT_NUMPY_FILE_PATHS
    Use external numpy.ndarray files for testing input data. Can specify multiple times.
  --debug
    Enable debug mode (keep browser open on error).
  --fallback_to_cpu
    Automatically fallback to CPU if other execution providers fail.
  --input_names INPUT_NAMES
    Input tensor names (comma-separated)
  --output_names OUTPUT_NAMES
    Output tensor names (comma-separated)
  --ort_model_path ORT_MODEL_PATH
    Path to ORT format model file
  --headless/--no-headless
    Run browser in headless mode (default: True)
  --timeout TIMEOUT
    Browser timeout in seconds (default: 60)
```

### 3. In-script Usage

```python
from sit4onnxw import inference

# Single input model
results = inference(
    input_onnx_file_path="mobilenetv2-12.onnx",
    execution_provider="webgl",
    test_loop_count=10
)

# Multi-input model with list of shapes (sit4onnx style)
results = inference(
    input_onnx_file_path="multi_input_model.onnx",
    fixed_shapes=[[1, 64, 112, 200], [1, 3, 112, 200]],
    execution_provider="cpu",
    test_loop_count=10
)

# Multi-input model with external numpy files
results = inference(
    input_onnx_file_path="multi_input_model.onnx",
    input_numpy_file_paths=["input1.npy", "input2.npy"],
    execution_provider="cpu",
    test_loop_count=10
)

# Legacy format support (backward compatibility)
results = inference(
    input_onnx_file_path="multi_input_model.onnx",
    fixed_shapes="1,64,112,200;1,3,112,200",  # semicolon-separated
    execution_provider="cpu"
)
```

## Model Support

### Input Types
- **Fixed Shape Models**: Models with static input dimensions
- **Dynamic Shape Models**: Models with dynamic dimensions (batch_size, seq_len, etc.)
- **Multi-Input Models**: Models with multiple input tensors
- **Mixed Models**: Combination of fixed and dynamic inputs

### Automatic Shape Inference
sit4onnxw automatically infers appropriate shapes for dynamic tensors:
- `batch_size`, `N`, `batch` → Uses --batch_size parameter (default: 1)
- `seq`, `sequence`, `seq_len`, `time` → Default: 90
- `features`, `hidden`, `embed`, `channels` → Default: 105
- Unknown dimensions by position → [batch_size, 90, 105, 1, ...]

### Tested Model Examples
- **Single Input Fixed**: [1, 90, 105] → Works perfectly
- **Single Input Dynamic**: [batch_size, seq, features] → Auto-inferred to [1, 90, 105]
- **Multi-Input Fixed**: [1, 64, 112, 200] + [1, 3, 112, 200] → 4 outputs
- **WebGL Large Models**: Works but may be slow (27+ seconds for large models)

## Performance Characteristics

| Execution Provider | Speed | Compatibility | Best Use Case |
|-------------------|-------|---------------|---------------|
| **CPU** | Fast (1-3s) | Excellent | Default choice, most reliable |
| **WebGL** | Variable | Good | When WebGL acceleration needed |
| **WebGPU** | Variable | Limited | Experimental, latest browsers |

## Error Handling

sit4onnxw provides intelligent error categorization:

- **Model Format**: Incompatible model format for execution provider
- **Input Tensor**: Shape/dimension mismatches with detailed error info
- **WebGL/WebGPU**: Provider-specific errors with fallback suggestions
- **WebAssembly**: WASM loading failures with troubleshooting hints

Use `--fallback_to_cpu` for automatic recovery from execution provider failures.

## Troubleshooting

### Common Issues
1. **WebGL Errors**: Use `--fallback_to_cpu` or switch to `--execution_provider cpu`
2. **Shape Mismatches**: Check model requirements with `--debug` mode
3. **Large Models**: Consider using CPU provider for better reliability
4. **Browser Issues**: Ensure Chrome/Chromium is installed and up-to-date

### Debug Mode
```bash
sit4onnxw --input_onnx_file_path model.onnx --debug
```
Keeps browser open for manual inspection when errors occur.

## Output Examples

### 1. Single Input Model (CPU)
```bash
$ sit4onnxw \
--input_onnx_file_path model_optimized_dynamic.onnx \
--execution_provider cpu \
--test_loop_count 5

sit4onnxw - Simple Inference Test for ONNX Runtime Web
Model: model_optimized_dynamic.onnx
Execution Provider: cpu
Batch Size: 1
Test Loop Count: 5
--------------------------------------------------
Model has 1 input(s):
  Input 0: input - shape: [batch_size, seq, features], type: 1
Input 'input': shape = [1, 90, 105], dtype = <class 'numpy.float32'>
Generated data shape: (1, 90, 105), size: 9450
Converting 'input': original shape=(1, 90, 105), size=9450
  Converted to list: length=1, type=<class 'list'>
Execution Provider: cpu
Test Loop Count: 5
Average Inference Time: 4.840 ms
Min Inference Time: 4.400 ms
Max Inference Time: 5.200 ms
--------------------------------------------------
Inference completed successfully!
Number of outputs: 1
Output 0: shape=(1, 2), dtype=float64
```

### 2. Multi-Input Model with Fixed Shapes
```bash
$ sit4onnxw \
--input_onnx_file_path model_multi_input_fix.onnx \
--fixed_shapes '1 64 112 200' \
--fixed_shapes '1 3 112 200' \
--test_loop_count 3

sit4onnxw - Simple Inference Test for ONNX Runtime Web
Model: model_multi_input_fix.onnx
Execution Provider: cpu
Batch Size: 1
Test Loop Count: 3
Fixed Shapes: [[1, 64, 112, 200], [1, 3, 112, 200]]
--------------------------------------------------
Model has 2 input(s):
  Input 0: feat - shape: [1, 64, 112, 200], type: 1
  Input 1: pc_dep - shape: [1, 3, 112, 200], type: 1
Applied fixed shape for input 0: [1, 64, 112, 200]
Input 'feat': shape = [1, 64, 112, 200], dtype = <class 'numpy.float32'>
Generated data shape: (1, 64, 112, 200), size: 1433600
Applied fixed shape for input 1: [1, 3, 112, 200]
Input 'pc_dep': shape = [1, 3, 112, 200], dtype = <class 'numpy.float32'>
Generated data shape: (1, 3, 112, 200), size: 67200
Execution Provider: cpu
Test Loop Count: 3
Average Inference Time: 1578.500 ms
Min Inference Time: 1568.000 ms
Max Inference Time: 1589.000 ms
--------------------------------------------------
Inference completed successfully!
Number of outputs: 4
Output 0: shape=(1, 3, 112, 200), dtype=float64
Output 1: shape=(1, 8, 112, 200), dtype=float64
Output 2: shape=(1, 1, 112, 200), dtype=float64
Output 3: shape=(1, 8, 112, 200), dtype=float64
```

### 3. WebGL with CPU Fallback
```bash
$ sit4onnxw \
--input_onnx_file_path model_optimized_dynamic.onnx \
--execution_provider webgl \
--fallback_to_cpu \
--test_loop_count 3

sit4onnxw - Simple Inference Test for ONNX Runtime Web
Model: model_optimized_dynamic.onnx
Execution Provider: webgl
Batch Size: 1
Test Loop Count: 3
--------------------------------------------------
Warning: webgl execution provider failed: Browser error: Error (Model Format): Model format incompatible with selected execution provider. Try CPU provider.
Falling back to cpu execution provider...
Model has 1 input(s):
  Input 0: input - shape: [batch_size, seq, features], type: 1
Input 'input': shape = [1, 90, 105], dtype = <class 'numpy.float32'>
Generated data shape: (1, 90, 105), size: 9450
Execution Provider: cpu
Test Loop Count: 3
Average Inference Time: 4.767 ms
Min Inference Time: 4.400 ms
Max Inference Time: 5.100 ms
--------------------------------------------------
Inference completed successfully!
Number of outputs: 1
Output 0: shape=(1, 2), dtype=float64
```

### 4. Error Case - Shape Mismatch
```bash
$ sit4onnxw \
--input_onnx_file_path model_multi_input_fix.onnx \
--fixed_shapes '1 32 112 200' \
--test_loop_count 1

sit4onnxw - Simple Inference Test for ONNX Runtime Web
Model: model_multi_input_fix.onnx
Execution Provider: cpu
Batch Size: 1
Test Loop Count: 1
Fixed Shapes: [[1, 32, 112, 200]]
--------------------------------------------------
Model has 2 input(s):
  Input 0: feat - shape: [1, 64, 112, 200], type: 1
  Input 1: pc_dep - shape: [1, 3, 112, 200], type: 1
Applied fixed shape for input 0: [1, 32, 112, 200]
Warning: Single fixed shape provided but this is input 1. Using defaults.
Error: Browser error: Error (Unknown): failed to call OrtRun(). ERROR_CODE: 2, ERROR_MESSAGE: Got invalid dimensions for input: feat for the following indices index: 1 Got: 32 Expected: 64 Please fix either the inputs/outputs or the model.
```

### 5. Dynamic Tensor with Batch Size
```bash
$ sit4onnxw \
--input_onnx_file_path model_optimized_dynamic.onnx \
--batch_size 4 \
--test_loop_count 2

sit4onnxw - Simple Inference Test for ONNX Runtime Web
Model: model_optimized_dynamic.onnx
Execution Provider: cpu
Batch Size: 4
Test Loop Count: 2
--------------------------------------------------
Model has 1 input(s):
  Input 0: input - shape: [batch_size, seq, features], type: 1
Input 'input': shape = [4, 90, 105], dtype = <class 'numpy.float32'>
Generated data shape: (4, 90, 105), size: 37800
Execution Provider: cpu
Test Loop Count: 2
Average Inference Time: 16.967 ms
Min Inference Time: 16.900 ms
Max Inference Time: 17.000 ms
--------------------------------------------------
Inference completed successfully!
Number of outputs: 1
Output 0: shape=(4, 2), dtype=float64
```

### 6. Using External Numpy Files
```bash
$ sit4onnxw \
--input_onnx_file_path model_multi_input_fix.onnx \
--input_numpy_file_paths test_feat.npy \
--input_numpy_file_paths test_pc_dep.npy \
--test_loop_count 1

sit4onnxw - Simple Inference Test for ONNX Runtime Web
Model: model_multi_input_fix.onnx
Execution Provider: cpu
Batch Size: 1
Test Loop Count: 1
--------------------------------------------------
Model has 2 input(s):
  Input 0: feat - shape: [1, 64, 112, 200], type: 1
  Input 1: pc_dep - shape: [1, 3, 112, 200], type: 1
Converting 'feat': original shape=(1, 64, 112, 200), size=1433600
Converting 'pc_dep': original shape=(1, 3, 112, 200), size=67200
Execution Provider: cpu
Test Loop Count: 1
Average Inference Time: 1561.200 ms
Min Inference Time: 1561.200 ms
Max Inference Time: 1561.200 ms
--------------------------------------------------
Inference completed successfully!
Number of outputs: 4
Output 0: shape=(1, 3, 112, 200), dtype=float64
Output 1: shape=(1, 8, 112, 200), dtype=float64
Output 2: shape=(1, 1, 112, 200), dtype=float64
Output 3: shape=(1, 8, 112, 200), dtype=float64
```

## Requirements

- Python 3.10+
- Chrome or Chromium browser
- WebDriver compatible browser setup
- Chrome WebDriver (automatically managed via webdriver-manager)

## License

MIT License