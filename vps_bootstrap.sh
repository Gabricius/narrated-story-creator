#!/bin/bash
# ═══════════════════════════════════════════════
# VPS Bootstrap — configura uma VPS nova para o pipeline
# 
# Uso (a partir de qualquer máquina):
#   ssh root@NOVA_VPS 'bash -s' < vps_bootstrap.sh
#
# Ou direto na VPS:
#   curl -sL https://raw.githubusercontent.com/SEU_USER/SEU_REPO/main/vps_bootstrap.sh | bash
#
# O que faz:
#   1. Instala dependências (rclone, python libs)
#   2. Baixa youtube_upload.py do GitHub (sempre a versão mais recente)
#   3. Configura /root/.env (pede as vars interativamente)
#   4. Testa conexão com Supabase
# ═══════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════"
echo "  VPS Bootstrap — Pipeline Setup"
echo "═══════════════════════════════════════"

# ── 1. Dependências ──
echo ""
echo "[1/4] Instalando dependências..."

# Python packages
pip3 install --quiet --break-system-packages \
    google-api-python-client \
    google-auth-oauthlib \
    google-auth-httplib2 \
    requests \
    2>/dev/null || pip3 install --quiet \
    google-api-python-client \
    google-auth-oauthlib \
    google-auth-httplib2 \
    requests

# Rclone
if ! command -v rclone &>/dev/null; then
    echo "  Instalando rclone..."
    curl -s https://rclone.org/install.sh | bash
else
    echo "  rclone já instalado: $(rclone version | head -1)"
fi

echo "  ✓ Dependências OK"

# ── 2. Download youtube_upload.py ──
echo ""
echo "[2/4] Baixando youtube_upload.py..."

# EDITE ESTA URL para apontar pro seu repositório:
UPLOAD_SCRIPT_URL="${UPLOAD_SCRIPT_URL:-https://raw.githubusercontent.com/SEU_USER/SEU_REPO/main/youtube_upload.py}"

# Se a URL ainda é o placeholder, tenta copiar do diretório atual
if [[ "$UPLOAD_SCRIPT_URL" == *"SEU_USER"* ]]; then
    if [[ -f "./youtube_upload.py" ]]; then
        cp ./youtube_upload.py /root/youtube_upload.py
        echo "  ✓ Copiado do diretório atual"
    else
        echo "  ⚠️  Configure UPLOAD_SCRIPT_URL no script ou coloque youtube_upload.py no diretório atual"
        echo "  Pulando download..."
    fi
else
    curl -sL "$UPLOAD_SCRIPT_URL" -o /root/youtube_upload.py
    echo "  ✓ Baixado de $UPLOAD_SCRIPT_URL"
fi

chmod +x /root/youtube_upload.py 2>/dev/null || true

# ── 3. Configurar .env ──
echo ""
echo "[3/4] Configurando /root/.env..."

if [[ -f /root/.env ]]; then
    echo "  /root/.env já existe:"
    grep -v "KEY\|SECRET\|TOKEN" /root/.env | head -5
    read -p "  Sobrescrever? (s/N): " overwrite
    if [[ "$overwrite" != "s" && "$overwrite" != "S" ]]; then
        echo "  Mantendo .env existente"
    else
        WRITE_ENV=true
    fi
else
    WRITE_ENV=true
fi

if [[ "$WRITE_ENV" == "true" ]]; then
    # Check if vars are passed via environment first
    if [[ -n "$SUPABASE_URL" && -n "$SUPABASE_ANON_KEY" ]]; then
        echo "  Usando variáveis de ambiente existentes"
    else
        echo ""
        read -p "  SUPABASE_URL (ex: https://xxx.supabase.co): " SUPABASE_URL
        read -p "  SUPABASE_ANON_KEY (eyJ...): " SUPABASE_ANON_KEY
    fi
    
    cat > /root/.env << EOF
SUPABASE_URL=$SUPABASE_URL
SUPABASE_ANON_KEY=$SUPABASE_ANON_KEY
EOF
    chmod 600 /root/.env
    echo "  ✓ /root/.env criado (chmod 600)"
fi

# ── 4. Testar conexão ──
echo ""
echo "[4/4] Testando conexão com Supabase..."

source /root/.env 2>/dev/null || true

if [[ -n "$SUPABASE_URL" && -n "$SUPABASE_ANON_KEY" ]]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "apikey: $SUPABASE_ANON_KEY" \
        -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
        "$SUPABASE_URL/rest/v1/youtube_credentials?select=credential_name&limit=1")
    
    if [[ "$HTTP_CODE" == "200" ]]; then
        COUNT=$(curl -s \
            -H "apikey: $SUPABASE_ANON_KEY" \
            -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
            "$SUPABASE_URL/rest/v1/youtube_credentials?select=credential_name" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
        echo "  ✓ Supabase OK — $COUNT credencial(is) encontrada(s)"
    else
        echo "  ⚠️  Supabase respondeu HTTP $HTTP_CODE — verifique URL e chave"
    fi
else
    echo "  ⚠️  Variáveis não configuradas, pulando teste"
fi

# ── Resumo ──
echo ""
echo "═══════════════════════════════════════"
echo "  Setup completo!"
echo "═══════════════════════════════════════"
echo ""
echo "  Arquivos:"
echo "    /root/youtube_upload.py  — script de upload"
echo "    /root/.env               — credenciais Supabase"
echo ""
echo "  Testar manualmente:"
echo "    python3 /root/youtube_upload.py --json '{\"channel_credential\":\"nora-scott\",\"title\":\"Teste\"}'"
echo ""
echo "  Atualizar script:"
echo "    curl -sL \$UPLOAD_SCRIPT_URL -o /root/youtube_upload.py"
echo ""
