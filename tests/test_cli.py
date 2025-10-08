from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from sit4onnxw.cli import main as cli_main


def test_cli_accepts_unquoted_fixed_shapes():
    runner = CliRunner()
    model_path = Path(__file__).resolve().parent.parent / 'model_optimized_fix.onnx'
    captured = {}

    def fake_inference(**kwargs):
        captured['fixed_shapes'] = kwargs.get('fixed_shapes')
        return [SimpleNamespace(shape=(1,), dtype='float32')]

    with patch('sit4onnxw.cli.inference', fake_inference):
        result = runner.invoke(
            cli_main,
            [
                '--input_onnx_file_path',
                str(model_path),
                '--execution_provider',
                'cpu',
                '--fixed_shapes',
                '1',
                '3',
                '480',
                '640',
                '--test_loop_count',
                '1',
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured['fixed_shapes'] == [[1, 3, 480, 640]]
    assert 'Error' not in result.output
