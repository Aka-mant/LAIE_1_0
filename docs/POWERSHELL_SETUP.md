# Развёртывание LAIE в Windows PowerShell

Требования: Python 3.13 x64, PostgreSQL 16+, Git (необязательно).

```powershell
# 1. Создать каталог проекта
New-Item -ItemType Directory -Force C:\Projects\LAIE | Out-Null
Set-Location C:\Projects\LAIE

# 2. Распаковать архив LAIE_1_0.zip в этот каталог, если он скачан отдельно

# 3. Создать виртуальное окружение
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Если PowerShell блокирует активацию, используйте разовый обход только для текущего процесса:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 4. Установить зависимости
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Создать конфигурацию
Copy-Item .env.example .env
notepad .env

# 6. Создать БД и пользователя в PostgreSQL (выполнить в psql от администратора)
# CREATE ROLE laie LOGIN PASSWORD 'replace-with-strong-password';
# CREATE DATABASE laie OWNER laie ENCODING 'UTF8';

# 7. Проверить подключение, создать миграции
python manage.py check
python manage.py makemigrations lotteries draws
python manage.py migrate

# 8. Создать администратора
python manage.py createsuperuser

# 9. Тесты и сервер
pytest
python manage.py runserver 127.0.0.1:8000
```

Не открывайте PostgreSQL в Интернет. Облачный контур в промышленной версии должен обращаться к локальному координатору через исходящее HTTPS-соединение с аутентификацией и подписью заданий.
