import pytest
from services.probability import (
    affordable_ticket_count,
    exact_match_probability,
    full_combination_probability,
)


def test_full_combination_probability_is_exact():
    assert full_combination_probability() == pytest.approx(1 / 503880)


def test_hypergeometric_probabilities_sum_to_one():
    total = sum(
        exact_match_probability(pool_size=20, draw_count=8, ticket_count=8, matches=k)
        for k in range(9)
    )
    assert total == pytest.approx(1.0)


def test_affordable_ticket_count():
    assert affordable_ticket_count(budget=5000, ticket_price=400) == (12, 200)
    assert affordable_ticket_count(budget=5000, ticket_price=300) == (16, 200)


def test_invalid_price_raises():
    with pytest.raises(ValueError):
        affordable_ticket_count(budget=5000, ticket_price=0)
