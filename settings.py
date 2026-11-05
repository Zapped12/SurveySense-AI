"""Read server configuration without accepting settings from survey or UI data."""

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _get_value(name, default):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return default


def get_setting(name, default=""):
    """Read text settings; a nonempty environment value takes priority."""
    value = _get_value(name, default)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text in your environment or Streamlit secrets.")
    return value.strip()


def live_api_enabled():
    """Fail closed: only boolean true or the text 'true' enables paid requests."""
    value = _get_value("ENABLE_LIVE_API", False)
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")
