# LAIE 1.0 — Local Autonomous Edition

Автономная локальная исследовательская система для анализа истории лотереи
«Рапидо Про», проверки статистических гипотез и формирования портфелей в рамках
лимита 5 000 рублей на тираж.

## Архитектура

- Windows x64; Django web UI served on `127.0.0.1` only.
- Local PostgreSQL is the authoritative database.
- CPU is always supported; optional NVIDIA CUDA acceleration is provided through PyTorch.
- No cloud workers, remote compute, external AI APIs or cloud database.
- Internet is not required during normal runtime after installation and local data loading.
- Ticket purchase is manual only.

## Important scientific limitation

Лотерея является случайной игрой. Исторические частоты и найденные закономерности
не доказывают предсказуемость будущих тиражей. Любая модель сравнивается с равномерным
случайным базовым алгоритмом на независимых временных выборках. Система не обещает
прибыль и не должна маркировать гипотезу как преимущество без статистически устойчивых
вневыборочных доказательств.

## Setup

См. `docs/POWERSHELL_SETUP.md`.

## Инженерный процесс разработки

Роли команды, порядок разработки, независимого тестирования, аудита и приёмки описаны в [docs/ENGINEERING_WORKFLOW.md](docs/ENGINEERING_WORKFLOW.md).

## GPU

См. `docs/LOCAL_GPU_ARCHITECTURE.md`. Состояние оборудования доступно по адресу
`http://127.0.0.1:8000/api/v1/compute/status/` после запуска Django.

## Status

Это стартовый каркас, а не законченная система. До промышленного использования
необходимо реализовать оставшиеся доменные приложения, импорт истории, полный
экспериментальный контур, интерфейс и пройти тестирование на реальной локальной БД.
