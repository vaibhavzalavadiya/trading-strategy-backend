import os
import dj_database_url
from .settings import *

# Production settings
DEBUG = False
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')

# Allowed hosts
ALLOWED_HOSTS = [
    os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'my-backend.onrender.com'),
    'localhost',
    '127.0.0.1',
]

# Database
DATABASES = {
    'default': dj_database_url.parse(os.environ.get('DATABASE_URL'))
}

# Static files with Whitenoise
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# CORS settings for production
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = [
    "https://trading-strategy-backtest.vercel.app/",  # Replace with your Vercel domain
]

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'