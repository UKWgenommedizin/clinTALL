FROM python:3.10-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# set workdir
WORKDIR /app

# copy requirements
COPY clinTall_requirements.txt /app/requirements.txt

# RUN pip install torch torchvision torchaudio 
# install python deps
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY clinTall.py /app/clinTall.py

# default command (override with docker run args)
CMD ["python", "clinTall.py"]

