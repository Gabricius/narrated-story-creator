# Dockerfile para Video Maker API v2
FROM python:3.11-slim

# Instalar dependências do sistema necessárias para ffmpeg e processamento de vídeo
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar requirements.txt primeiro (para cache de layers do Docker)
COPY requirements.txt .

# Copiar rclone para acessar Google Drive
COPY --from=rclone/rclone:latest /usr/local/bin/rclone /usr/bin/rclone

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Criar diretórios necessários
RUN mkdir -p /app/assets /app/tmp /app/videos

# Copiar arquivos da aplicação (estes SÃO obrigatórios)
COPY server.py .
COPY video_maker.py .

# Script para obter assets (fontes e ícone)
# Este script baixa os arquivos automaticamente
RUN python3 << 'EOF'
import os
import urllib.request
from pathlib import Path
import shutil

assets_dir = Path('/app/assets')
assets_dir.mkdir(exist_ok=True)

# Definir fontes necessárias com URLs alternativas mais confiáveis
fonts = {
    'anton.ttf': 'https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf',
    'arial.ttf': 'https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf',
}

print("📥 Downloading required fonts...")
for filename, url in fonts.items():
    target = assets_dir / filename
    if target.exists():
        print(f"✓ {filename} already exists")
    else:
        try:
            print(f"⬇️  Downloading {filename} from {url}...")
            # Add headers to avoid issues with GitHub
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(str(target), 'wb') as out_file:
                    out_file.write(response.read())
            
            # Verify file was downloaded
            if target.exists() and target.stat().st_size > 0:
                print(f"✓ {filename} downloaded successfully ({target.stat().st_size} bytes)")
            else:
                print(f"⚠️  {filename} download failed - file is empty or missing")
        except Exception as e:
            print(f"⚠️  Could not download {filename}: {e}")

# Use Arial as fallback for Noto fonts (Arial supports most characters)
arial_path = assets_dir / 'arial.ttf'
noto_path = assets_dir / 'noto.ttf'
noto_hindi_path = assets_dir / 'noto_hindi.ttf'

if arial_path.exists() and arial_path.stat().st_size > 0:
    print("📋 Ensuring Noto fonts exist (using Arial as base)...")
    
    # SEMPRE copiar Arial como noto.ttf (sem condição)
    shutil.copy(str(arial_path), str(noto_path))
    print(f"✓ Arial → noto.ttf ({noto_path.stat().st_size} bytes)")
    
    # SEMPRE copiar Arial como noto_hindi.ttf (sem condição)
    shutil.copy(str(arial_path), str(noto_hindi_path))
    print(f"✓ Arial → noto_hindi.ttf ({noto_hindi_path.stat().st_size} bytes)")
else:
    print("❌ CRITICAL ERROR: Arial font not available!")
    raise Exception("Arial font is required but missing")

# Criar ícone de volume
icon_path = assets_dir / 'icon_volume.png'
if not icon_path.exists():
    print("🎨 Creating volume icon...")
    try:
        from PIL import Image, ImageDraw
        
        # Criar um ícone de volume simples
        img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Desenhar alto-falante
        # Corpo
        draw.rectangle([100, 200, 200, 312], fill='white', outline='black', width=4)
        # Cone
        draw.polygon([(200, 200), (300, 150), (300, 362), (200, 312)], fill='white', outline='black')
        
        # Ondas sonoras
        for i in range(3):
            offset = 50 + (i * 40)
            draw.arc([300+offset, 200-offset, 400+offset, 312+offset], 
                    start=-45, end=45, fill='white', width=10)
        
        img.save(str(icon_path))
        print(f"✓ Volume icon created at {icon_path}")
    except Exception as e:
        print(f"⚠️  Could not create icon: {e}")
else:
    print(f"✓ icon_volume.png already exists")

# Final verification
print("\n📊 Final asset check:")
for filename in ['anton.ttf', 'arial.ttf', 'noto.ttf', 'noto_hindi.ttf', 'icon_volume.png']:
    file_path = assets_dir / filename
    if file_path.exists():
        size = file_path.stat().st_size
        print(f"  ✓ {filename} ({size:,} bytes)")
    else:
        print(f"  ❌ {filename} MISSING!")

print("✅ Asset download complete!")
EOF

# Verificar se os assets foram criados
RUN ls -lah /app/assets/

# Expor porta da API
EXPOSE 8000

# Variáveis de ambiente padrão
ENV WORK_DIR=/app
ENV CUDA=0
ENV PYTHONUNBUFFERED=1

# Comando para iniciar o servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
