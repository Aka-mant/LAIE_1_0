# Развёртывание LAIE в Windows PowerShell

## Требования

- Windows 10/11 x64.
- Python 3.13 x64.
- PostgreSQL 16+ установлен на этом же компьютере и слушает только loopback.
- Для GPU-ускорения: совместимый NVIDIA-драйвер и CUDA-сборка PyTorch (необязательно).

## Установка

```powershell
# Перейти в каталог проекта
Set-Location C:\Projects\LAIE

# Виртуальное окружение
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Если PowerShell блокирует активацию, разрешить только для текущего процесса
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Установить основное CPU-окружение
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Локальная конфигурация

Файлы `.env` и `.env.example` не хранятся в Git. Создайте локальный `.env` вручную:

```powershell
@'
DJANGO_SECRET_KEY=replace-with-a-long-random-secret
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost,::1
DB_NAME=laie
DB_USER=laie
DB_PASSWORD=replace-with-a-strong-local-password
DB_HOST=127.0.0.1
DB_PORT=5432
LAIE_BUDGET_LIMIT=5000.00
LAIE_COMPUTE_MIN_GPU_ITEMS=100000
LAIE_GPU_MEMORY_RESERVE_MB=1536
'@ | Set-Content -Encoding UTF8 .env

notepad .env
```

**Важно:** `DB_HOST` принимает только `127.0.0.1`, `::1` или `localhost`.
`DJANGO_ALLOWED_HOSTS` также принимает только loopback-хосты. Внешние адреса,
имена DNS, wildcard и LAN-адреса блокируются при загрузке настроек.

## Локальная PostgreSQL

Создайте роль и базу через локальный `psql` от имени администратора:

```sql
CREATE ROLE laie LOGIN PASSWORD 'replace-with-a-strong-local-password';
CREATE DATABASE laie OWNER laie ENCODING 'UTF8';
```

Проверьте конфигурацию PostgreSQL: сервер должен принимать соединения только
через локальный интерфейс. Не открывайте порт 5432 в локальную сеть или Интернет.

## Проверка и запуск

```powershell
python manage.py check
python manage.py migrate
python manage.py createsuperuser
pytest
python manage.py runserver 127.0.0.1:8000
```

Интерфейс доступен только на этом компьютере: http://127.0.0.1:8000/

Не запускайте Django с адресом `0.0.0.0` или IP локальной сети.

## NVIDIA GPU (необязательно)

Основной `requirements.txt` обеспечивает CPU-режим. CUDA-зависимости устанавливаются
отдельно после проверки модели GPU, версии драйвера и доступной сборки PyTorch.

1. Выполните `nvidia-smi` и сохраните модель GPU и версию драйвера.
2. Откройте официальный установщик PyTorch:
   https://pytorch.org/get-started/locally/
3. Выберите Windows, Pip, Python и поддерживаемую CUDA-сборку.
4. Выполните сгенерированную команду внутри активного `.venv`.
5. Проверьте:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

Если CUDA или PyTorch недоступны, LAIE должна продолжить работу на CPU.
Не устанавливайте CUDA Toolkit наугад и не копируйте DLL вручную.

## Автономный режим

После установки, загрузки нужных исторических данных и зависимостей LAIE не требует
Интернета для локальных вычислений, обучения и анализа. Обновление истории выполняется
отдельным действием пользователя. Внешние ИИ API, облачные вычисления, удалённые
workers и автоматическая покупка билетов не используются.
