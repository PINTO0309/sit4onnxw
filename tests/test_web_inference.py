#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
import onnx
from onnx import helper, TensorProto

from sit4onnxw.web_inference import _prepare_input_data, _create_test_html


class TestWebInference(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple ONNX model for testing
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, 1000]
        )

        # Create a simple identity node
        node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['output']
        )

        # Create the graph
        graph = helper.make_graph(
            [node],
            'test_model',
            [input_tensor],
            [output_tensor]
        )

        # Create the model
        self.test_model = helper.make_model(graph)

        # Save to temporary file
        self.temp_model_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.onnx'
        )
        onnx.save(self.test_model, self.temp_model_file.name)
        self.temp_model_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_model_file.name)

    def test_prepare_input_data_with_fixed_shapes(self):
        """Test input data preparation with fixed shapes."""
        model_inputs = self.test_model.graph.input

        input_data = _prepare_input_data(
            model_inputs=model_inputs,
            batch_size=2,
            fixed_shapes=[2, 3, 224, 224],
            input_numpy_file_paths=None,
            numpy_seed=42
        )

        self.assertIn('input', input_data)
        self.assertEqual(input_data['input'].shape, (2, 3, 224, 224))
        self.assertEqual(input_data['input'].dtype, np.float32)

    def test_prepare_input_data_with_numpy_file(self):
        """Test input data preparation with numpy file."""
        # Create test numpy file
        test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        temp_numpy_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.npy'
        )
        np.save(temp_numpy_file.name, test_data)
        temp_numpy_file.close()

        try:
            model_inputs = self.test_model.graph.input

            input_data = _prepare_input_data(
                model_inputs=model_inputs,
                batch_size=1,
                fixed_shapes=None,
                input_numpy_file_paths=[temp_numpy_file.name],
                numpy_seed=None
            )

            self.assertIn('input', input_data)
            np.testing.assert_array_equal(input_data['input'], test_data)

        finally:
            os.unlink(temp_numpy_file.name)

    def test_create_test_html(self):
        """Test HTML test page creation."""
        input_data = {
            'input': np.random.randn(1, 3, 224, 224).astype(np.float32)
        }

        html_content = _create_test_html(
            model_path=self.temp_model_file.name,
            input_data=input_data,
            execution_provider='webgl',
            test_loop_count=5,
            enable_profiling=True
        )

        # Check that HTML contains expected elements
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('onnxruntime-web', html_content)
        self.assertIn('webgl', html_content)
        self.assertIn('runInference', html_content)
        self.assertIn('atob', html_content)  # Check for base64 decoding function

    def test_invalid_execution_provider(self):
        """Test error handling for invalid execution provider."""
        from sit4onnxw.web_inference import inference

        with self.assertRaises(ValueError) as context:
            inference(
                input_onnx_file_path=self.temp_model_file.name,
                execution_provider='invalid_provider'
            )

        self.assertIn('Unsupported execution provider', str(context.exception))

    def test_nonexistent_model_file(self):
        """Test error handling for nonexistent model file."""
        from sit4onnxw.web_inference import inference

        with self.assertRaises(FileNotFoundError):
            inference(
                input_onnx_file_path='/nonexistent/model.onnx'
            )


if __name__ == '__main__':
    unittest.main()