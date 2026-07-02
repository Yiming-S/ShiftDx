# ShiftDx dashboard container.
# Runs the dashboard on the pre-built CSVs in data/. The heavy build pipeline
# (MOABB/MNE + CrossPython) is intentionally NOT installed here — rebuild
# datasets on a workstation and bake the resulting CSVs into data/.
FROM python:3.13-slim

WORKDIR /app

# System deps occasionally needed by scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install only the dashboard runtime requirements (skip the git-based da4bci /
# crosspython lines and the MOABB/MNE build deps to keep the image small).
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    "streamlit>=1.36,<2" "plotly>=5.18,<7" "pandas>=2.0,<3" "numpy>=1.24,<3" \
    "scipy>=1.10,<2" "scikit-learn>=1.3,<2" "statsmodels>=0.14,<1" \
    "matplotlib>=3.7,<4" "POT>=0.9,<1"

# Optional: enable the DA Lab pages by also installing DA4BCI.
# RUN pip install --no-cache-dir "git+https://github.com/Yiming-S/DA4BCI-Python.git"

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
