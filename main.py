import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from supabase import create_client, Client

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Buscador Inteligente de Empresas")

# CORS – en producción limitar a tu dominio de Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Cargar variables de entorno ----------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ---------- Validaciones rápidas ----------
if not all([SERPAPI_KEY, MISTRAL_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    logger.error("Faltan variables de entorno obligatorias (SERPAPI_KEY, MISTRAL_API_KEY, SUPABASE_URL, SUPABASE_KEY).")

# ---------- Configurar Mistral (compatible con OpenAI) ----------
mistral = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)
MODELO_MISTRAL = "mistral-small-latest"   # modelo gratuito y eficiente

# ---------- Cliente Supabase ----------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CACHE_TTL_HOURS = 24
MAX_RESULTADOS = 5

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------

def buscar_empresas(query: str, num: int = MAX_RESULTADOS) -> list[dict]:
    """Usa SerpAPI para buscar en Google. Retorna lista con titulo, link, snippet."""
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "num": num,
            "hl": "es",          # resultados en español
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        empresas = []
        for item in data.get("organic_results", []):
            empresas.append({
                "titulo": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        logger.info(f"SerpAPI OK: {len(empresas)} resultados para '{query}'")
        return empresas
    except Exception as e:
        logger.error(f"Error en SerpAPI: {e}")
        return []

def extraer_texto_visible(url: str) -> str:
    """Obtiene texto visible de una web, máximo 3000 caracteres."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        texto = soup.get_text(separator=" ", strip=True)
        return texto[:3000]
    except Exception as e:
        logger.warning(f"No se pudo obtener texto de {url}: {e}")
        return ""

def extraer_datos_empresa_mistral(texto_web: str, snippet: str) -> dict:
    """
    Envía el texto y snippet a Mistral para extraer datos estructurados.
    Retorna un diccionario con los campos solicitados.
    """
    prompt = f"""
Eres un asistente que extrae información de empresas. A partir del siguiente contenido web y snippet de búsqueda, devuelve EXCLUSIVAMENTE un JSON válido (sin comentarios ni markdown) con los campos:
- nombre_empresa (string)
- descripcion (string, máximo 2 frases)
- productos (lista de strings, máximo 5)
- contacto (objeto con: email, telefono, direccion; si no se encuentra, poner null)
- emails_adicionales (lista de strings con otros correos electrónicos encontrados, sin incluir el email principal de contacto; si no hay, lista vacía [])
- redes_sociales (objeto con: linkedin, facebook, instagram, twitter; cada uno debe ser la URL completa incluyendo https://; si no se encuentra, null)
- anio_fundacion (número o null si no se encuentra)
- numero_empleados (número o null si no se encuentra)
- certificaciones (lista de strings, vacía si no hay)
- pagina_web (string, URL completa incluyendo https://)

Snippet de Google: {snippet}

Contenido de la página:
{texto_web[:3000]}

JSON:
"""
    try:
        response = mistral.chat.completions.create(
            model=MODELO_MISTRAL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
        texto_respuesta = response.choices[0].message.content.strip()
        # Limpiar backticks si los hubiera
        if texto_respuesta.startswith("```json"):
            texto_respuesta = texto_respuesta[7:]
        if texto_respuesta.endswith("```"):
            texto_respuesta = texto_respuesta[:-3]
        datos = json.loads(texto_respuesta)
        logger.info("Extracción Mistral exitosa")
        return datos
    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando JSON de Mistral: {e} | Respuesta: {texto_respuesta[:200]}")
        return {
            "nombre_empresa": None,
            "descripcion": None,
            "productos": [],
            "contacto": {"email": None, "telefono": None, "direccion": None},
            "emails_adicionales": [],
            "redes_sociales": {"linkedin": None, "facebook": None, "instagram": None, "twitter": None},
            "anio_fundacion": None,
            "numero_empleados": None,
            "certificaciones": [],
            "pagina_web": None,
            "error_extraccion": str(e)
        }
    except Exception as e:
        logger.error(f"Error inesperado en Mistral: {e}")
        return {
            "nombre_empresa": None,
            "descripcion": None,
            "productos": [],
            "contacto": {"email": None, "telefono": None, "direccion": None},
            "emails_adicionales": [],
            "redes_sociales": {"linkedin": None, "facebook": None, "instagram": None, "twitter": None},
            "anio_fundacion": None,
            "numero_empleados": None,
            "certificaciones": [],
            "pagina_web": None,
            "error_extraccion": str(e)
        }

def obtener_de_cache(query: str) -> Optional[list]:
    """Retorna lista de empresas si está en caché y no ha expirado, o None."""
    try:
        resp = supabase.table("empresas_cache").select("*").eq("query", query).execute()
        if resp.data:
            entry = resp.data[0]
            created = datetime.fromisoformat(entry["created_at"])
            if datetime.utcnow() - created < timedelta(hours=CACHE_TTL_HOURS):
                logger.info(f"Cache hit para '{query}'")
                return json.loads(entry["resultados"])
        return None
    except Exception as e:
        logger.error(f"Error consultando caché: {e}")
        return None

def guardar_en_cache(query: str, resultados: list) -> None:
    """Guarda los resultados en Supabase (upsert)."""
    try:
        supabase.table("empresas_cache").upsert({
            "query": query,
            "resultados": json.dumps(resultados, ensure_ascii=False),
            "created_at": datetime.utcnow().isoformat()
        }, on_conflict="query").execute()
        logger.info(f"Cache guardado para '{query}'")
    except Exception as e:
        logger.error(f"Error guardando en caché: {e}")

# ------------------------------------------------------------
# Endpoint principal
# ------------------------------------------------------------
@app.get("/buscar")
async def buscar(query: str = Query(..., description="Frase de búsqueda")):
    # 1. Intentar cache
    cache = obtener_de_cache(query)
    if cache:
        return {"empresas": cache, "fuente": "cache"}

    # 2. Buscar empresas con SerpAPI
    empresas_basicas = buscar_empresas(query, num=MAX_RESULTADOS)
    if not empresas_basicas:
        raise HTTPException(status_code=404, detail="No se encontraron resultados o hubo un error en la búsqueda.")

    # 3. Extraer datos con Mistral
    resultados = []
    for emp in empresas_basicas:
        texto = extraer_texto_visible(emp["link"])
        datos = extraer_datos_empresa_mistral(texto, emp["snippet"])
        datos["pagina_web"] = emp.get("link", datos.get("pagina_web"))
        resultados.append(datos)

    # 4. Guardar en caché
    guardar_en_cache(query, resultados)

    return {"empresas": resultados, "fuente": "fresh"}

@app.get("/")
async def raiz():
    return {"mensaje": "API funcionando. Use /buscar?query=..."}
