#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import tempfile
import numpy as np
from typing import List, Optional, Union, Dict, Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import onnx


def inference(
    input_onnx_file_path: str,
    batch_size: Optional[int] = 1,
    fixed_shapes: Optional[Union[List[List[int]], List[int], str]] = None,
    test_loop_count: Optional[int] = 10,
    execution_provider: Optional[str] = 'cpu',
    enable_profiling: Optional[bool] = False,
    output_numpy_file: Optional[bool] = False,
    numpy_seed: Optional[int] = None,
    input_numpy_file_paths: Optional[List[str]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    ort_model_path: Optional[str] = None,
    headless: Optional[bool] = True,
    timeout: Optional[int] = 60,
) -> List[np.ndarray]:
    """
    Perform inference on ONNX model using ONNX Runtime Web through browser automation.

    Args:
        input_onnx_file_path: Path to ONNX model file
        batch_size: Batch size for inference
        fixed_shapes: Fixed input shapes
        test_loop_count: Number of test iterations
        execution_provider: Execution provider ('cpu', 'webgl', 'webgpu')
        enable_profiling: Enable performance profiling
        output_numpy_file: Save output to numpy file
        numpy_seed: Random seed for input generation
        input_numpy_file_paths: Paths to input numpy files
        input_names: Input tensor names
        output_names: Output tensor names
        ort_model_path: Path to ORT format model
        headless: Run browser in headless mode
        timeout: Browser timeout in seconds

    Returns:
        List of output numpy arrays
    """

    # Validate inputs
    if not os.path.exists(input_onnx_file_path):
        raise FileNotFoundError(f"ONNX model file not found: {input_onnx_file_path}")

    if execution_provider not in ['cpu', 'webgl', 'webgpu']:
        raise ValueError(f"Unsupported execution provider: {execution_provider}")

    # Load ONNX model to get input/output info
    model = onnx.load(input_onnx_file_path)
    model_inputs = model.graph.input
    model_outputs = model.graph.output
    
    # Debug: print model input information
    print(f"Model has {len(model_inputs)} input(s):")
    for i, input_info in enumerate(model_inputs):
        shape_str = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape_str.append(str(dim.dim_value))
            elif dim.dim_param:
                shape_str.append(f"{dim.dim_param}")
            else:
                shape_str.append("?")
        print(f"  Input {i}: {input_info.name} - shape: [{', '.join(shape_str)}], type: {input_info.type.tensor_type.elem_type}")

    # Prepare input data
    input_data = _prepare_input_data(
        model_inputs,
        batch_size,
        fixed_shapes,
        input_numpy_file_paths,
        numpy_seed
    )

    # Create HTML test page
    html_content = _create_test_html(
        input_onnx_file_path,
        input_data,
        execution_provider,
        test_loop_count,
        enable_profiling,
        ort_model_path
    )

    # Setup Chrome driver
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Write HTML to temp file and load in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_html_path = f.name

        driver.get(f"file://{temp_html_path}")

        # Wait for inference to complete
        wait = WebDriverWait(driver, timeout)
        
        # Wait for either results or error status
        def check_completion(driver):
            status_element = driver.find_element(By.ID, "status")
            status_text = status_element.text
            if status_text in ["Completed", "Error"] or status_text.startswith("Error"):
                return True
            return False
        
        wait.until(check_completion)
        
        # Check for errors first
        status_element = driver.find_element(By.ID, "status")
        if status_element.text.startswith("Error"):
            # Get browser console logs for debugging
            try:
                logs = driver.get_log('browser')
                error_logs = [log for log in logs if log['level'] == 'SEVERE']
                if error_logs:
                    error_msg = f"Browser error: {status_element.text}\nConsole errors: {error_logs[0]['message']}"
                else:
                    error_msg = f"Browser error: {status_element.text}"
            except:
                error_msg = f"Browser error: {status_element.text}"
            raise RuntimeError(error_msg)
        
        # Get results
        results_element = driver.find_element(By.ID, "results")
        results_json = results_element.get_attribute("data-results")
        
        if not results_json or results_json.strip() == "":
            raise RuntimeError("No results data received from browser")
        
        try:
            results = json.loads(results_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse results JSON: {e}\nReceived data: {results_json[:200]}...")

        # Process outputs
        outputs = []
        for output_data in results['outputs']:
            output_array = np.array(output_data['data']).reshape(output_data['shape'])
            outputs.append(output_array)

        # Print benchmark results
        if 'benchmark' in results:
            benchmark = results['benchmark']
            print(f"Execution Provider: {execution_provider}")
            print(f"Test Loop Count: {test_loop_count}")
            print(f"Average Inference Time: {benchmark['avg_time']:.3f} ms")
            print(f"Min Inference Time: {benchmark['min_time']:.3f} ms")
            print(f"Max Inference Time: {benchmark['max_time']:.3f} ms")

        # Save outputs if requested
        if output_numpy_file:
            for i, output in enumerate(outputs):
                output_filename = f"output_{i}.npy"
                np.save(output_filename, output)
                print(f"Output saved to: {output_filename}")

        return outputs

    finally:
        driver.quit()
        # Clean up temp file
        if 'temp_html_path' in locals():
            os.unlink(temp_html_path)


def _prepare_input_data(
    model_inputs,
    batch_size: int,
    fixed_shapes: Optional[Union[List[List[int]], List[int], str]],
    input_numpy_file_paths: Optional[List[str]],
    numpy_seed: Optional[int]
) -> Dict[str, np.ndarray]:
    """Prepare input data for inference."""

    if numpy_seed is not None:
        np.random.seed(numpy_seed)

    input_data = {}

    for i, input_info in enumerate(model_inputs):
        input_name = input_info.name

        # Use provided numpy file if available
        if input_numpy_file_paths and i < len(input_numpy_file_paths):
            input_data[input_name] = np.load(input_numpy_file_paths[i])
            continue

        # Get input shape
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                # Dynamic dimension
                if dim.dim_param == 'batch_size' or dim.dim_param.lower() in ['n', 'batch']:
                    shape.append(batch_size)
                elif dim.dim_param.lower() in ['seq', 'sequence', 'seq_len', 'time']:
                    # Sequence dimension - use reasonable default
                    shape.append(90)
                elif dim.dim_param.lower() in ['features', 'hidden', 'embed', 'channels']:
                    # Feature dimension - use reasonable default
                    shape.append(105)
                else:
                    # For other dynamic dimensions, use 1 as default
                    shape.append(1)
            else:
                # Unknown dimension, use better defaults based on position
                if len(shape) == 0:
                    # First dimension - likely batch
                    shape.append(batch_size)
                elif len(shape) == 1:
                    # Second dimension - likely sequence
                    shape.append(90)
                elif len(shape) == 2:
                    # Third dimension - likely features
                    shape.append(105)
                else:
                    # Other dimensions
                    shape.append(1)
        
        # Handle fixed_shapes - support multiple inputs and formats
        if fixed_shapes:
            if isinstance(fixed_shapes, str):
                # Legacy string format (semicolon-separated)
                if ';' in fixed_shapes:
                    # Multiple input specification: "1,3,224,224;1,10"
                    input_shapes = fixed_shapes.split(';')
                    if i < len(input_shapes):
                        shape = [int(x.strip()) for x in input_shapes[i].split(',')]
                        print(f"Applied fixed shape for input {i}: {shape}")
                    else:
                        print(f"Warning: No fixed shape specified for input {i}, using defaults")
                else:
                    # Single input specification: "1,3,224,224"
                    if i == 0:
                        shape = [int(x.strip()) for x in fixed_shapes.split(',')]
                        print(f"Applied fixed shape for input {i}: {shape}")
                    else:
                        print(f"Warning: Single fixed shape provided but this is input {i}. Using defaults.")
            elif isinstance(fixed_shapes, list):
                # Check if this is a list of lists (sit4onnx style) or single list
                if len(fixed_shapes) > 0 and isinstance(fixed_shapes[0], list):
                    # List of lists format from CLI: [[1,3,224,224], [1,10]]
                    if i < len(fixed_shapes):
                        shape = fixed_shapes[i]
                        print(f"Applied fixed shape for input {i}: {shape}")
                    else:
                        print(f"Warning: No fixed shape specified for input {i}, using defaults")
                else:
                    # Single list format from Python API: [1,3,224,224]
                    if i == 0:
                        if len(fixed_shapes) == len(shape):
                            shape = fixed_shapes
                            print(f"Applied fixed shape for input {i}: {shape}")
                        else:
                            print(f"Warning: fixed_shapes length ({len(fixed_shapes)}) doesn't match input dimensions ({len(shape)})")
                            print(f"Using original shape: {shape}")
                    else:
                        print(f"Warning: Single list fixed shape provided but this is input {i}. Using defaults.")
        
        # Apply batch_size to the first dimension if it's dynamic
        if len(shape) > 0 and batch_size > 1:
            # Check if first dimension was dynamic and replace with batch_size
            original_dim = model_inputs[i].type.tensor_type.shape.dim[0]
            if original_dim.dim_param or not original_dim.dim_value:
                shape[0] = batch_size
        
        # Generate random input data
        dtype_map = {
            1: np.float32,   # FLOAT
            7: np.int64,     # INT64
            6: np.int32,     # INT32
        }

        elem_type = input_info.type.tensor_type.elem_type
        dtype = dtype_map.get(elem_type, np.float32)
        
        print(f"Input '{input_name}': shape = {shape}, dtype = {dtype}")

        if dtype == np.float32:
            input_data[input_name] = np.random.randn(*shape).astype(dtype)
        else:
            input_data[input_name] = np.random.randint(0, 10, shape).astype(dtype)
        
        print(f"Generated data shape: {input_data[input_name].shape}, size: {input_data[input_name].size}")

    return input_data


def _create_test_html(
    model_path: str,
    input_data: Dict[str, np.ndarray],
    execution_provider: str,
    test_loop_count: int,
    enable_profiling: bool = False,  # Currently unused but kept for future use
    ort_model_path: Optional[str] = None
) -> str:
    """Create HTML page for ONNX Runtime Web testing."""

    # Convert numpy arrays to lists for JSON serialization
    input_data_json = {}
    for name, array in input_data.items():
        data_list = array.tolist()
        print(f"Converting '{name}': original shape={array.shape}, size={array.size}")
        print(f"  Converted to list: length={len(data_list)}, type={type(data_list)}")
        input_data_json[name] = {
            'data': data_list,
            'shape': list(array.shape),
            'dtype': str(array.dtype)
        }

    # Read model file as base64
    import base64
    
    # Determine which model path to use and what format it is
    actual_model_path = ort_model_path if ort_model_path else model_path
    is_ort_format = actual_model_path.endswith('.ort') if ort_model_path else False
    
    with open(actual_model_path, 'rb') as f:
        model_data = base64.b64encode(f.read()).decode('utf-8')

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ONNX Runtime Web Test</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
</head>
<body>
    <div id="status">Loading...</div>
    <div id="results" data-results=""></div>

    <script>
        // Set WASM paths for proper loading
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
        
        // Configure WebGL/WebGPU settings
        if ('{execution_provider}' === 'webgl') {{
            ort.env.webgl.contextId = 'webgl2';
            ort.env.webgl.matmulMaxBatchSize = 16;
            ort.env.webgl.pack = true;
        }}
        
        if ('{execution_provider}' === 'webgpu') {{
            ort.env.webgpu.validateInputContent = false;
        }}
        
        async function runInference() {{
            try {{
                console.log('Starting inference with execution provider: {execution_provider}');
                
                // Set execution provider
                const providers = [];
                let useWebGLFallback = false;
                
                if ('{execution_provider}' === 'webgl') {{
                    providers.push({{
                        name: 'webgl',
                        // Additional WebGL options that might help with compatibility
                        preferredLayout: 'NHWC'
                    }});
                    useWebGLFallback = true;
                }} else if ('{execution_provider}' === 'webgpu') {{
                    providers.push('webgpu');
                }}
                providers.push('cpu');  // fallback

                // Load model
                const modelData = atob('{model_data}');
                const modelBuffer = new Uint8Array(modelData.length);
                for (let i = 0; i < modelData.length; i++) {{
                    modelBuffer[i] = modelData.charCodeAt(i);
                }}

                // Check if this is an ORT format model
                const isOrtFormat = {str(is_ort_format).lower()};
                
                let session;
                let actualProvider = '{execution_provider}';
                
                try {{
                    // Try to create session with requested providers
                    if (isOrtFormat) {{
                        // For ORT format, use the buffer directly with ORT format loading
                        session = await ort.InferenceSession.create(modelBuffer, {{
                            executionProviders: providers,
                            // Explicitly specify this is ORT format
                            graphOptimizationLevel: 'all'
                        }});
                    }} else {{
                        // For regular ONNX format
                        session = await ort.InferenceSession.create(modelBuffer, {{
                            executionProviders: providers
                        }});
                    }}
                }} catch (providerError) {{
                    console.warn('Failed to create session with requested provider, trying CPU only:', providerError);
                    
                    // If WebGL/WebGPU fails, try CPU only
                    if (useWebGLFallback || '{execution_provider}' === 'webgpu') {{
                        try {{
                            session = await ort.InferenceSession.create(modelBuffer, {{
                                executionProviders: ['cpu']
                            }});
                            actualProvider = 'cpu (fallback)';
                            console.log('Successfully fell back to CPU provider');
                        }} catch (cpuError) {{
                            throw new Error(`Failed to load model with any provider. Original error: ${{providerError.message}}, CPU error: ${{cpuError.message}}`);
                        }}
                    }} else {{
                        throw providerError;
                    }}
                }}
                
                console.log(`Session created successfully with provider: ${{actualProvider}}`);
                console.log('Input names:', session.inputNames);
                console.log('Output names:', session.outputNames);

                // Prepare input data
                const inputData = {json.dumps(input_data_json)};
                const feeds = {{}};

                for (const [name, data] of Object.entries(inputData)) {{
                    // Flatten nested arrays properly
                    function flattenDeep(arr) {{
                        return arr.reduce((acc, val) => Array.isArray(val) ? acc.concat(flattenDeep(val)) : acc.concat(val), []);
                    }}
                    
                    const flatData = flattenDeep(data.data);
                    
                    console.log(`Processing input '${{name}}': shape=${{data.shape}}, dtype=${{data.dtype}}, data_length=${{flatData.length}}`);
                    
                    // Calculate expected size
                    const expectedSize = data.shape.reduce((a, b) => a * b, 1);
                    const actualSize = flatData.length;
                    
                    if (expectedSize !== actualSize) {{
                        throw new Error(`Tensor '${{name}}' size mismatch: expected ${{expectedSize}}, got ${{actualSize}}`);
                    }}
                    
                    let typedArray;
                    let tensorType;
                    if (data.dtype.includes('float32')) {{
                        typedArray = new Float32Array(flatData);
                        tensorType = 'float32';
                    }} else if (data.dtype.includes('int64')) {{
                        typedArray = new BigInt64Array(flatData.map(x => BigInt(x)));
                        tensorType = 'int64';
                    }} else if (data.dtype.includes('int32')) {{
                        typedArray = new Int32Array(flatData);
                        tensorType = 'int32';
                    }} else {{
                        typedArray = new Float32Array(flatData);
                        tensorType = 'float32';
                    }}
                    
                    feeds[name] = new ort.Tensor(tensorType, typedArray, data.shape);
                }}

                // Warm up
                await session.run(feeds);

                // Benchmark
                const times = [];
                const testLoopCount = {test_loop_count};

                document.getElementById('status').textContent = 'Running benchmark...';

                for (let i = 0; i < testLoopCount; i++) {{
                    const start = performance.now();
                    const results = await session.run(feeds);
                    const end = performance.now();
                    times.push(end - start);

                    if (i === 0) {{
                        // Store first result
                        window.firstResult = results;
                    }}
                }}

                // Calculate statistics
                const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
                const minTime = Math.min(...times);
                const maxTime = Math.max(...times);

                // Prepare outputs
                const outputs = [];
                for (const [name, tensor] of Object.entries(window.firstResult)) {{
                    outputs.push({{
                        name: name,
                        data: Array.from(tensor.data),
                        shape: tensor.dims,
                        dtype: tensor.type
                    }});
                }}

                const result = {{
                    outputs: outputs,
                    benchmark: {{
                        avg_time: avgTime,
                        min_time: minTime,
                        max_time: maxTime,
                        test_count: testLoopCount,
                        execution_provider: actualProvider
                    }}
                }};

                document.getElementById('results').setAttribute('data-results', JSON.stringify(result));
                document.getElementById('status').textContent = 'Completed';

            }} catch (error) {{
                console.error('Error details:', error);
                console.error('Error stack:', error.stack);
                const errorMessage = error.message || error.toString();
                
                // Categorize errors for better user understanding
                let errorCategory = 'Unknown';
                let userFriendlyMessage = errorMessage;
                
                if (errorMessage.includes('WebGL')) {{
                    errorCategory = 'WebGL';
                    userFriendlyMessage = 'WebGL execution provider not supported or failed. Try CPU provider instead.';
                }} else if (errorMessage.includes('WebGPU')) {{
                    errorCategory = 'WebGPU';
                    userFriendlyMessage = 'WebGPU execution provider not supported or failed. Try CPU provider instead.';
                }} else if (errorMessage.includes('irVersion') || errorMessage.includes('loadFromOrtFormat')) {{
                    errorCategory = 'Model Format';
                    userFriendlyMessage = 'Model format incompatible with selected execution provider. Try CPU provider.';
                }} else if (errorMessage.includes('Tensor')) {{
                    errorCategory = 'Input Tensor';
                    userFriendlyMessage = errorMessage; // Keep original tensor error messages
                }} else if (errorMessage.includes('WASM') || errorMessage.includes('WebAssembly')) {{
                    errorCategory = 'WebAssembly';
                    userFriendlyMessage = 'WebAssembly loading failed. Check network connection and browser support.';
                }}
                
                document.getElementById('status').textContent = `Error (${{errorCategory}}): ${{userFriendlyMessage}}`;
                
                // Provide detailed error information for debugging
                const errorResult = {{
                    error: errorMessage,
                    errorType: error.constructor.name,
                    errorCategory: errorCategory,
                    userFriendlyMessage: userFriendlyMessage,
                    stack: error.stack,
                    executionProvider: '{execution_provider}',
                    browserInfo: navigator.userAgent,
                    outputs: []
                }};
                document.getElementById('results').setAttribute('data-results', JSON.stringify(errorResult));
            }}
        }}

        // Start inference when page loads
        window.addEventListener('load', runInference);
    </script>
</body>
</html>
    """

    return html_template