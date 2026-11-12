"""Earth Engine initialization using gcloud Application Default Credentials (ADC).

`earthengine authenticate` is blocked for this account (it requests the Drive scope),
and a stale ~/.config/earthengine/credentials file makes a plain ee.Initialize() fail.
So we load ADC explicitly and pass it to ee.Initialize.

One-time setup:
    gcloud auth application-default login \
        --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform
"""

from __future__ import annotations

import os

import ee
import google.auth
from dotenv import find_dotenv, load_dotenv

EE_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]


def init_ee(project: str | None = None) -> str:
    """Initialize Earth Engine with ADC credentials and return the project used.

    The project comes from the argument, else the EE_PROJECT environment variable.
    A .env file in the current directory or a parent is loaded first, without
    overriding variables already set in the environment.
    """
    load_dotenv(find_dotenv(usecwd=True))
    project = project or os.environ.get("EE_PROJECT")
    if not project:
        raise ValueError("Set EE_PROJECT in your environment or .env, or pass project=")
    credentials, _ = google.auth.default(scopes=EE_SCOPES)
    ee.Initialize(credentials=credentials, project=project)
    return project
