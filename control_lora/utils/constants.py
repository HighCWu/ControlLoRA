import os

from appdirs import user_cache_dir


HOME_PATH = os.environ.get('CONTROL_LORA_HOME', user_cache_dir("control_lora"))
os.makedirs(HOME_PATH, exist_ok=True)
