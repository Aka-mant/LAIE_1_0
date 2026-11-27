from math import comb


def exact_match_probability(
    *, pool_size: int, draw_count: int, ticket_count: int, matches: int
) -> float:
    """Гипергеометрическая вероятность ровно matches совпадений."""
    if not (0 <= draw_count <= pool_size):
        raise ValueError("draw_count должен быть от 0 до pool_size")
    if not (0 <= ticket_count <= pool_size):
        raise ValueError("ticket_count должен быть от 0 до pool_size")
    if not (0 <= matches <= min(draw_count, ticket_count)):
        return 0.0
    misses = draw_count - matches
    if misses > pool_size - ticket_count:
        return 0.0
    return (
        comb(ticket_count, matches)
        * comb(pool_size - ticket_count, misses)
        / comb(pool_size, draw_count)
    )


def full_combination_probability() -> float:
    """Вероятность полного совпадения для 8/20 + 1/4."""
    return 1.0 / (comb(20, 8) * 4)


def affordable_ticket_count(*, budget: int, ticket_price: int) -> tuple[int, int]:
    if budget < 0 or ticket_price <= 0:
        raise ValueError("Некорректный бюджет или цена")
    count = budget // ticket_price
    return count, budget - count * ticket_price
