FROM python:3.13-slim
# versões fixas para reproduzir os resultados (torch só CPU, bem menor)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy==2.5.5 gymnasium==1.4.0 matplotlib==3.11.2 pandas==3.1.1 \
    stable-baselines3==2.9.2 typst==0.15.2
ENV MPLCONFIGDIR=/tmp/matplotlib MPLBACKEND=Agg PYTHONUNBUFFERED=1
WORKDIR /trabalho
