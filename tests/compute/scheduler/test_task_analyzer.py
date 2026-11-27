
"""Тесты анализатора вычислительных задач LAIE."""

import pytest

from services.compute.scheduler import (
    TaskType,
    create_default_registry,
)
from services.compute.scheduler.task_analyzer import (
    TaskAnalyzer,
    TaskParameters,
)


@pytest.fixture
def analyzer():
    """Возвращает анализатор со стандартным реестром."""
    return TaskAnalyzer(create_default_registry())


def test_analyzer_rejects_invalid_registry():
    with pytest.raises(TypeError):
        TaskAnalyzer("invalid")


@pytest.mark.parametrize("multiplier", [0.5, 0, -1])
def test_analyzer_rejects_invalid_workspace_multiplier(multiplier):
    with pytest.raises(ValueError):
        TaskAnalyzer(
            create_default_registry(),
            workspace_multiplier=multiplier,
        )


def test_analyzer_rejects_non_numeric_multiplier():
    with pytest.raises(TypeError):
        TaskAnalyzer(
            create_default_registry(),
            workspace_multiplier="1.5",
        )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("sample_count", 0, ValueError),
        ("sample_count", -1, ValueError),
        ("sample_count", 1.5, TypeError),
        ("sample_count", True, TypeError),
        ("feature_count", "10", TypeError),
    ],
)
def test_task_parameters_validate_values(field, value, error):
    with pytest.raises(error):
        TaskParameters(**{field: value})


def test_task_parameters_accept_none():
    parameters = TaskParameters()

    assert parameters.sample_count is None
    assert parameters.matrix_m is None


def test_analyzer_rejects_invalid_parameters_object(analyzer):
    with pytest.raises(TypeError):
        analyzer.analyze(
            TaskType.MATRIX_MULTIPLICATION,
            {"matrix_m": 10},
        )


def test_matrix_multiplication_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=10,
            matrix_k=20,
            matrix_n=30,
        ),
    )

    assert result.item_count == 6000
    assert result.estimated_operations == 12000
    assert result.estimated_input_bytes == (10 * 20 + 20 * 30) * 8
    assert result.estimated_output_bytes == 10 * 30 * 8
    assert result.complexity_model == "O(M*K*N)"


def test_matrix_analysis_requires_all_dimensions(analyzer):
    with pytest.raises(ValueError, match="matrix_k"):
        analyzer.analyze(
            TaskType.MATRIX_MULTIPLICATION,
            TaskParameters(matrix_m=10, matrix_n=20),
        )


def test_statistical_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.STATISTICAL_ANALYSIS,
        TaskParameters(
            sample_count=100,
            feature_count=5,
        ),
    )

    assert result.item_count == 500
    assert result.estimated_operations == 5000
    assert result.estimated_input_bytes == 100 * 5 * 8
    assert result.estimated_output_bytes == 5 * 8


def test_combination_generation_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.COMBINATION_GENERATION,
        TaskParameters(
            candidate_count=1000,
            item_width=8,
        ),
    )

    assert result.item_count == 1000
    assert result.estimated_operations == 8000
    assert result.estimated_output_bytes == 1000 * 8 * 8


def test_monte_carlo_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.MONTE_CARLO,
        TaskParameters(
            iterations=10000,
            operations_per_iteration=200,
            state_size=16,
        ),
    )

    assert result.item_count == 10000
    assert result.estimated_operations == 2_000_000
    assert result.estimated_input_bytes == 16 * 8
    assert result.estimated_output_bytes == 10000 * 8


def test_genetic_optimization_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.GENETIC_OPTIMIZATION,
        TaskParameters(
            population_size=100,
            generations=50,
            genome_size=20,
        ),
    )

    assert result.item_count == 5000
    assert result.estimated_operations == 100_000
    assert result.estimated_input_bytes == 100 * 20 * 8


def test_model_training_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.MODEL_TRAINING,
        TaskParameters(
            sample_count=1000,
            feature_count=20,
            epochs=5,
            batch_size=100,
        ),
    )

    assert result.item_count == 5000
    assert result.estimated_operations == 200_000
    assert result.estimated_input_bytes == 1000 * 20 * 4
    assert result.estimated_output_bytes == 20 * 4
    assert result.estimated_workspace_bytes >= 100 * 20 * 4


def test_model_training_defaults_batch_size_to_sample_count(analyzer):
    result = analyzer.analyze(
        TaskType.MODEL_TRAINING,
        TaskParameters(
            sample_count=100,
            feature_count=10,
            epochs=2,
        ),
    )

    assert result.estimated_workspace_bytes >= 100 * 10 * 4


def test_hypothesis_testing_analysis(analyzer):
    result = analyzer.analyze(
        TaskType.HYPOTHESIS_TESTING,
        TaskParameters(
            sample_count=500,
            hypothesis_count=20,
        ),
    )

    assert result.item_count == 20
    assert result.estimated_operations == 100_000
    assert result.estimated_input_bytes == 500 * 8
    assert result.estimated_output_bytes == 20 * 8


def test_total_memory_equals_components(analyzer):
    result = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=10,
            matrix_k=10,
            matrix_n=10,
        ),
    )

    assert result.estimated_memory_bytes == (
        result.estimated_input_bytes
        + result.estimated_output_bytes
        + result.estimated_workspace_bytes
    )


def test_workspace_multiplier_changes_memory_estimate():
    registry = create_default_registry()

    normal = TaskAnalyzer(
        registry,
        workspace_multiplier=1.5,
    )
    conservative = TaskAnalyzer(
        registry,
        workspace_multiplier=3.0,
    )

    parameters = TaskParameters(
        matrix_m=100,
        matrix_k=100,
        matrix_n=100,
    )

    normal_result = normal.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        parameters,
    )
    conservative_result = conservative.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        parameters,
    )

    assert (
        conservative_result.estimated_workspace_bytes
        > normal_result.estimated_workspace_bytes
    )


def test_analysis_preserves_task_definition(analyzer):
    result = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=10,
            matrix_k=10,
            matrix_n=10,
        ),
    )

    assert result.task_type == TaskType.MATRIX_MULTIPLICATION
    assert result.parallelizable is True
    assert result.supports_batching is True
    assert result.precision.value == "float64"


def test_analysis_result_is_immutable(analyzer):
    result = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=10,
            matrix_k=10,
            matrix_n=10,
        ),
    )

    with pytest.raises(AttributeError):
        result.item_count = 999


def test_analysis_can_be_serialized(analyzer):
    result = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=10,
            matrix_k=20,
            matrix_n=30,
        ),
    )

    data = result.as_dict()

    assert data["task_type"] == "matrix_multiplication"
    assert data["workload"] == "gpu_parallel"
    assert "estimated_memory_bytes" in data
    assert isinstance(data["supported_backends"], list)
    assert isinstance(data["assumptions"], list)


def test_cuda_analysis_preserves_cpu_fallback(analyzer):
    result = analyzer.analyze(
        TaskType.MONTE_CARLO,
        TaskParameters(
            iterations=1000,
            operations_per_iteration=100,
            state_size=10,
        ),
    )

    backends = {backend.value for backend in result.supported_backends}

    assert backends == {"cpu", "cuda"}


def test_combination_generation_is_cpu_only(analyzer):
    result = analyzer.analyze(
        TaskType.COMBINATION_GENERATION,
        TaskParameters(
            candidate_count=100,
            item_width=8,
        ),
    )

    assert [backend.value for backend in result.supported_backends] == [
        "cpu"
    ]


def test_workspace_is_never_negative(analyzer):
    result = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=2,
            matrix_k=2,
            matrix_n=2,
        ),
    )

    assert result.estimated_workspace_bytes >= 0
    assert result.estimated_memory_bytes > 0
