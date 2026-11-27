import os
from pathlib import Path

from dotenv import load_dotenv

from config.local_only import parse_local_allowed_hosts, require_loopback_host

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "unsafe-dev-key-change-before-deploy")
DEBUG = os.getenv("DJANGO_DEBUG", "False").lower() == "true"

# Автономная установка не принимает запросы с LAN или внешних доменов.
ALLOWED_HOSTS = parse_local_allowed_hosts(
    os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,::1")
)

INSTALLED_APPS = [
    "django.contrib.admin", "django.contrib.auth", "django.contrib.contenttypes",
    "django.contrib.sessions", "django.contrib.messages", "django.contrib.staticfiles",
    "rest_framework", "apps.lotteries", "apps.draws", "apps.research",
    "apps.experiments", "apps.portfolios", "apps.compute", "apps.api",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "config.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"], "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
    ]},
}]
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# PostgreSQL обязана работать на этом же компьютере.
_db_host = require_loopback_host(
    os.getenv("DB_HOST", "127.0.0.1"),
    "DB_HOST",
)
DATABASES = {"default": {
    "ENGINE": "django.db.backends.postgresql",
    "NAME": os.getenv("DB_NAME", "laie"),
    "USER": os.getenv("DB_USER", "laie"),
    "PASSWORD": os.getenv("DB_PASSWORD", ""),
    "HOST": _db_host,
    "PORT": os.getenv("DB_PORT", "5432"),
    "CONN_MAX_AGE": 60,
}}

LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
}
LAIE_BUDGET_LIMIT = os.getenv("LAIE_BUDGET_LIMIT", "5000.00")
LAIE_COMPUTE_MIN_GPU_ITEMS = int(os.getenv("LAIE_COMPUTE_MIN_GPU_ITEMS", "100000"))
LAIE_GPU_MEMORY_RESERVE_MB = int(os.getenv("LAIE_GPU_MEMORY_RESERVE_MB", "1536"))
LAIE_GPU_MEMORY_RESERVE_RATIO = float(
    os.getenv("LAIE_GPU_MEMORY_RESERVE_RATIO", "0.30")
)
