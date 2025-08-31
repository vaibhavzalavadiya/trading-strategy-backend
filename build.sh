#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --noinput --settings=config.settings_prod
python manage.py migrate --settings=config.settings_prod