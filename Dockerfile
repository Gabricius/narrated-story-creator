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

# Copiar assets obrigatórios (fontes)
COPY anton.ttf assets/
COPY arial.ttf assets/
COPY noto.ttf assets/
COPY noto_hindi.ttf assets/

# Copiar ícone se existir, caso contrário criar um placeholder
# Isso evita que o build falhe se o arquivo não existir
COPY icon_volume.png assets/ || true

# Criar ícone placeholder se não existir
RUN python3 << 'EOF'
import os
from pathlib import Path

icon_path = Path('/app/assets/icon_volume.png')

if not icon_path.exists():
    print("Warning: icon_volume.png not found, creating placeholder...")
    try:
        from PIL import Image, ImageDraw
        
        # Criar um ícone de volume simples
        img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Desenhar um alto-falante simples
        # Corpo do alto-falante
        draw.rectangle([100, 200, 200, 312], fill='white', outline='black', width=3)
        # Cone
        draw.polygon([(200, 200), (300, 150), (300, 362), (200, 312)], fill='white', outline='black')
        # Ondas sonoras
        for i in range(3):
            offset = 50 + (i * 40)
            draw.arc([300+offset, 200-offset, 400+offset, 312+offset], 
                    start=-45, end=45, fill='white', width=8)
        
        img.save(icon_path)
        print(f"Created placeholder icon at {icon_path}")
    except Exception as e:
        print(f"Could not create icon: {e}")
        print("Continuing without icon - application may have issues")
else:
    print(f"Found icon_volume.png at {icon_path}")
EOF

# Expor porta da API
EXPOSE 8000

# Variáveis de ambiente padrão
ENV WORK_DIR=/app
ENV CUDA=0
ENV PYTHONUNBUFFERED=1

# Comando para iniciar o servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
