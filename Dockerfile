# Dockerfile para Video Maker API v2
FROM python:3.11-slim

# Instalar dependências do sistema necessárias para ffmpeg e processamento de vídeo
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar requirements.txt primeiro (para cache de layers do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Criar diretórios necessários
RUN mkdir -p /app/assets /app/tmp /app/videos

# Copiar arquivos da aplicação
COPY server.py .
COPY video_maker.py .

# Copiar assets (fontes e ícones)
COPY anton.ttf assets/
COPY arial.ttf assets/
COPY noto.ttf assets/
COPY noto_hindi.ttf assets/
COPY icon_volume.png assets/

# Expor porta da API
EXPOSE 8000

# Variáveis de ambiente padrão
ENV WORK_DIR=/app
ENV CUDA=0
ENV PYTHONUNBUFFERED=1

# Comando para iniciar o servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
