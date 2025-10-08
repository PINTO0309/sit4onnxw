#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import sys
from typing import Optional
from .web_inference import inference


class FixedShapesOption(click.Option):
    """
    Custom click option that collects all subsequent non-option tokens
    (including negative integers) for --fixed_shapes, allowing users to
    specify shapes without quoting them.
    """

    def add_to_parser(self, parser, ctx):
        super().add_to_parser(parser, ctx)

        def _wrap_option(option_parser):
            original_process = option_parser.process

            def process(value, state):
                collected = []
                if value is not None:
                    collected.append(value)
                while state.rargs:
                    next_arg = state.rargs[0]
                    if next_arg == '--':
                        break
                    if next_arg.startswith('-') and not self._looks_like_int(next_arg):
                        break
                    collected.append(state.rargs.pop(0))
                joined = ' '.join(collected)
                return original_process(joined, state)

            option_parser.process = process

        for opt in self.opts:
            option = parser._long_opt.get(opt)
            if option:
                _wrap_option(option)
        for opt in self.secondary_opts:
            option = parser._short_opt.get(opt)
            if option:
                _wrap_option(option)

    @staticmethod
    def _looks_like_int(token: str) -> bool:
        if token in {'', '-', '+'}:
            return False
        stripped = token.lstrip('+-')
        return stripped.isdigit()


@click.command()
@click.option(
    '--input_onnx_file_path',
    '-if',
    type=click.Path(exists=True),
    required=True,
    help='Input onnx file path.'
)
@click.option(
    '--batch_size',
    '-b',
    type=int,
    default=1,
    help='Value to be substituted if input batch size is undefined. Ignored if input dimensions are static or other inputs specified.'
)
@click.option(
    '--fixed_shapes',
    '-fs',
    type=str,
    multiple=True,
    cls=FixedShapesOption,
    help='Input OPs with undefined shapes changed to specified shape. Can be specified multiple times for different input OPs.'
)
@click.option(
    '--test_loop_count',
    '-tlc',
    type=int,
    default=10,
    help='Number of times to run the test. Total execution time divided by test runs to show average inference time.'
)
@click.option(
    '--execution_provider',
    '-ep',
    type=click.Choice(['cpu', 'webgl', 'webgpu']),
    default='cpu',
    help='ONNX Runtime Web Execution Provider.'
)
@click.option(
    '--enable_profiling',
    '-prof',
    is_flag=True,
    help='Outputs performance profiling result to a .json file'
)
@click.option(
    '--output_numpy_file',
    '-ofp',
    is_flag=True,
    help='Outputs the last inference result to an .npy file.'
)
@click.option(
    '--numpy_seed',
    '-seed',
    type=int,
    help='Random seed for input data generation.'
)
@click.option(
    '--input_numpy_file_paths',
    '-ifp',
    type=str,
    multiple=True,
    help='Use external numpy.ndarray files for testing input data. Can specify multiple times.'
)
@click.option(
    '--input_names',
    '-in',
    type=str,
    help='Input tensor names (comma-separated).'
)
@click.option(
    '--output_names',
    '-on',
    type=str,
    help='Output tensor names (comma-separated).'
)
@click.option(
    '--ort_model_path',
    '-ort',
    type=click.Path(exists=True),
    help='Path to ORT format model file.'
)
@click.option(
    '--headless/--no-headless',
    default=True,
    help='Run browser in headless mode.'
)
@click.option(
    '--timeout',
    '-t',
    type=int,
    default=60,
    help='Browser timeout in seconds.'
)
@click.option(
    '--debug',
    '-d',
    is_flag=True,
    help='Enable debug mode (keep browser open on error).'
)
@click.option(
    '--fallback_to_cpu',
    '-fallback',
    is_flag=True,
    help='Automatically fallback to CPU if other execution providers fail.'
)
def main(
    input_onnx_file_path: str,
    batch_size: int,
    fixed_shapes: tuple,
    test_loop_count: int,
    execution_provider: str,
    enable_profiling: bool,
    output_numpy_file: bool,
    numpy_seed: Optional[int],
    input_numpy_file_paths: tuple,
    input_names: Optional[str],
    output_names: Optional[str],
    ort_model_path: Optional[str],
    headless: bool,
    timeout: int,
    debug: bool,
    fallback_to_cpu: bool,
):
    """
    Simple Inference Test for ONNX Runtime Web

    Benchmark ONNX models using ONNX Runtime Web with support for CPU, WebGL, and WebGPU execution providers.
    """

    try:
        # Process multiple parameter specifications (sit4onnx style)
        fixed_shapes_list = None
        if fixed_shapes:
            # Convert tuple to list of lists, parsing each shape specification
            # Example: ("1 3 224 224", "1 10") -> [[1,3,224,224], [1,10]]
            fixed_shapes_list = []
            for shape_spec in fixed_shapes:
                shape = [int(x.strip()) for x in shape_spec.split()]
                fixed_shapes_list.append(shape)

        input_numpy_file_paths_list = None
        if input_numpy_file_paths:
            # Convert tuple to list
            input_numpy_file_paths_list = list(input_numpy_file_paths)

        input_names_list = None
        if input_names:
            input_names_list = [x.strip() for x in input_names.split(',')]

        output_names_list = None
        if output_names:
            output_names_list = [x.strip() for x in output_names.split(',')]

        # Display configuration
        click.echo(f"sit4onnxw - Simple Inference Test for ONNX Runtime Web")
        click.echo(f"Model: {input_onnx_file_path}")
        click.echo(f"Execution Provider: {execution_provider}")
        click.echo(f"Batch Size: {batch_size}")
        click.echo(f"Test Loop Count: {test_loop_count}")
        if fixed_shapes_list:
            click.echo(f"Fixed Shapes: {fixed_shapes_list}")
        if ort_model_path:
            click.echo(f"ORT Model: {ort_model_path}")
        click.echo("-" * 50)

        # Run inference with fallback support
        original_provider = execution_provider
        providers_to_try = [execution_provider]
        
        # Add fallback if enabled
        if fallback_to_cpu and execution_provider != 'cpu':
            providers_to_try.append('cpu')
        
        results = None
        last_error = None
        
        for provider in providers_to_try:
            try:
                if provider != original_provider:
                    click.echo(f"Falling back to {provider} execution provider...")
                
                results = inference(
                    input_onnx_file_path=input_onnx_file_path,
                    batch_size=batch_size,
                    fixed_shapes=fixed_shapes_list,
                    test_loop_count=test_loop_count,
                    execution_provider=provider,
                    enable_profiling=enable_profiling,
                    output_numpy_file=output_numpy_file,
                    numpy_seed=numpy_seed,
                    input_numpy_file_paths=input_numpy_file_paths_list,
                    input_names=input_names_list,
                    output_names=output_names_list,
                    ort_model_path=ort_model_path,
                    headless=headless and not debug,
                    timeout=timeout,
                )
                break  # Success, exit loop
                
            except Exception as e:
                last_error = e
                if provider == providers_to_try[-1]:
                    # This was the last provider to try
                    raise e
                else:
                    click.echo(f"Warning: {provider} execution provider failed: {e}")
        
        if results is None:
            raise last_error

        click.echo("-" * 50)
        click.echo(f"Inference completed successfully!")
        click.echo(f"Number of outputs: {len(results)}")

        for i, output in enumerate(results):
            click.echo(f"Output {i}: shape={output.shape}, dtype={output.dtype}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
