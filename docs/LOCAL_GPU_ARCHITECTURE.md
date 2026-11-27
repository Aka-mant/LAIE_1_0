# LAIE Local Autonomous Edition — GPU architecture

## Обязательные архитектурные ограничения

- Django запускается только на `127.0.0.1`; адреса LAN, `0.0.0.0` и внешние DNS-имена запрещены.
- PostgreSQL, исходные данные, результаты экспериментов, модели и журналы хранятся локально.
- PostgreSQL host ограничен `127.0.0.1`, `::1` или `localhost` проверкой при загрузке Django settings.
- Нет Cloud Worker, Cloud Jobs, удалённых вычислительных endpoints, внешних AI API или удалённой БД.
- Интернет не нужен для штатного выполнения после установки зависимостей и локальной загрузки истории.
- Исторические результаты обновляются только по инициативе пользователя.
- Покупка билетов остаётся ручной операцией.

## NVIDIA / CUDA

`services.compute.engine.ComputeEngine` загружает PyTorch лениво. Если PyTorch/CUDA
не установлены или драйвер недоступен, приложение остаётся работоспособным на CPU.

Для крупных задач движок выбирает CUDA при достижении
`LAIE_COMPUTE_MIN_GPU_ITEMS` и наличии свободной VRAM сверх резерва
`LAIE_GPU_MEMORY_RESERVE_MB`. При ошибке CUDA в демонстрационной операции
матричного умножения выполняется повтор на CPU.

Текущая конфигурация целевой машины: NVIDIA GeForce RTX 3070, 8 GB VRAM.
Фактические модель, версия драйвера и доступная память должны подтверждаться
локальным выводом `nvidia-smi`, а не предполагаться по спецификации.

## Уже существующие компоненты

В репозитории уже присутствуют следующие элементы вычислительного контура:

- `services/compute/engine.py` — базовый CPU/CUDA selector и матричная операция с CPU fallback;
- `services/compute/benchmark.py` — benchmark CPU/CUDA матричного умножения, прогрев, синхронизация CUDA и сравнение точности с float64;
- `services/compute/end_to_end_benchmark.py` — сквозное измерение с учётом передачи CPU RAM → GPU VRAM → CPU RAM;
- `services/compute/scheduler/` — адаптивная обработка матричных пакетов, decision/execution engines, анализ задач, оценка памяти, hardware monitor, PyTorch adapter и реестр типов задач;
- `apps/compute/management/commands/benchmark_compute.py` и `benchmark_end_to_end.py` — локальные команды запуска бенчмарков;
- `apps/compute/views.py` — диагностический endpoint `GET /api/v1/compute/status/`.

Наличие этих модулей не доказывает, что все доменные задачи LAIE уже используют
единый execution pipeline. Необходимо проверить контракты scheduler, конкурентность,
bounded retries, OOM, CPU fallback и фактическую интеграцию каждого workload.

## Установка CUDA PyTorch

Основной `requirements.txt` намеренно не устанавливает PyTorch: он содержит
CPU-совместимый набор зависимостей. CUDA-сборка устанавливается отдельно по
официальному селектору https://pytorch.org/get-started/locally/ после проверки
версии NVIDIA-драйвера. Не фиксировать произвольную версию CUDA Toolkit.

## Диагностика

`GET /api/v1/compute/status/` сообщает наличие PyTorch, доступность CUDA,
модель GPU и VRAM, если эти сведения доступны драйверу.

Scheduler hardware monitor дополнительно собирает имя CPU, число логических ядер,
общую и доступную RAM, имя CUDA-устройства и свободную/общую VRAM.

## Параметры

- `LAIE_COMPUTE_MIN_GPU_ITEMS`: минимальный размер задачи для автоматического выбора GPU.
- `LAIE_GPU_MEMORY_RESERVE_MB`: резерв VRAM, который не отдаётся вычислительной задаче.

Эти значения — стартовые эвристики, не измеренная модель производительности.
Решение о выборе устройства не должно считаться доказательством ускорения:
необходим локальный benchmark CPU/GPU на репрезентативных задачах LAIE.

## Текущие ограничения

- Benchmark-компоненты существуют, но основной benchmark матричного умножения
  не включает transfer time в измеряемый интервал; для этого есть отдельный
  end-to-end benchmark.
- Scheduler содержит adaptive matrix service и memory guard, но его устойчивость
  к конкурентным GPU-заданиям, отмене, повторным OOM и восстановлению требует отдельной проверки.
- Не доказана интеграция GPU с каждым математическим/ML/optimization pipeline.
- В просмотренном HardwareSnapshot есть CPU/RAM и CUDA/VRAM; телеметрия
  температуры и загрузки GPU не обнаружена.
- Compute API предоставляет JSON-диагностику, но полноценная Compute Dashboard
  с очередью, прогрессом, benchmark history и ошибками не подтверждена.
- PyTorch/CUDA runtime проверяется на целевой машине; успешная установка пакета
  сама по себе не доказывает доступность GPU.
