# Acciones-Bursatiles

Escáner automático de acciones de EE.UU. que se ejecuta con **GitHub Actions** después de cada cierre del
mercado. Analiza el S&P 500 y el Nasdaq-100, dibuja gráficos con las medias móviles **MA(7), MA(25) y MA(99)**,
lee las noticias de cada acción, ordena las de mayor potencial de subida, hace seguimiento de cada compra
(con sus razones y por qué es importante) y entrena una **red neuronal** que aprende a detectar buenos
puntos de compra.

> ⚠️ Proyecto educativo y experimental. **No es asesoría financiera.**

## ¿Qué hace cada día?

| Paso | Detalle |
|---|---|
| 🔎 **Escaneo** | Descarga 5 años de precios diarios de ~520 acciones (S&P 500 + Nasdaq-100, desde Wikipedia) y de los índices SPY, QQQ, DIA e IWM. |
| 📊 **Gráficos** | Velas japonesas con **MA(7)**, **MA(25)**, **MA(99)** y volumen para las mejores oportunidades, las acciones en seguimiento y los índices. |
| 📰 **Noticias** | Titulares de Yahoo Finance y Google News (y Finnhub si das una clave). Cada titular recibe un sentimiento de −1 a +1 con un vocabulario financiero (*beats*, *upgrade*, *downgrade*, *lawsuit*…); los recientes pesan más. |
| 🚀 **Potencial de subida** | Puntaje de 0 a 100 que combina análisis técnico (medias, cruces, retrocesos, máximos de 52 semanas), momentum (RSI, MACD, volumen, fuerza frente al S&P 500), noticias, precio objetivo de analistas y, cuando esté lista, la red neuronal. |
| 💼 **Seguimiento** | Una cartera simulada de $100.000 compra las mejores oportunidades y guarda **las razones de cada compra** y **por qué la empresa es importante**. Cada día revisa la tesis, escribe una bitácora, avisa con alertas y vende por stop-loss, objetivo o ruptura de tendencia. También sigue **tus compras reales**. |
| 🧠 **Red neuronal** | Entrena una sesión, predice y se autoevalúa. Cuando aprende lo suficiente, sus recomendaciones se suman al puntaje. |

## Dónde ver los resultados

Cada ejecución publica los reportes en la rama **[`reportes`](../../tree/reportes)**:

- [`README.md`](../../blob/reportes/README.md): panel principal (mercado, top de oportunidades, estado de la red y de la cartera).
- [`oportunidades.md`](../../blob/reportes/oportunidades.md): gráfico, señales, riesgos, noticias y analistas de cada acción.
- [`seguimiento.md`](../../blob/reportes/seguimiento.md): razones de compra, importancia, tesis, alertas y bitácora de cada posición.
- [`red_neuronal.md`](../../blob/reportes/red_neuronal.md): cómo aprende, criterios de madurez, curvas de aprendizaje y predicciones.
- [`datos/escaneo.csv`](../../blob/reportes/datos/escaneo.csv): todas las acciones analizadas con sus puntajes.

La rama `reportes` siempre tiene un único commit con la última versión. Así el repositorio no crece con cada
gráfico. El estado (cartera, pesos de la red, historial) viaja en esa misma rama y además se guarda como
artefacto de cada ejecución durante 30 días.

## Puesta en marcha

1. **Lleva este código a la rama principal** (`main`). GitHub solo ejecuta los workflows programados desde
   la rama por defecto.
2. **Primera ejecución:** pestaña **Actions** → *Escaneo del mercado y gráficos* → **Run workflow**. Tarda
   unos 10 minutos. Después corre solo de lunes a viernes a las 22:30 UTC, y los sábados hace una sesión
   larga de entrenamiento de la red.
3. Si el workflow falla al publicar con un error 403: **Settings → Actions → General → Workflow
   permissions → Read and write permissions**.
4. *(Opcional)* **GitHub Pages**: **Settings → Pages → Deploy from a branch → `reportes` / `(root)`** para
   ver los reportes como página web.
5. *(Opcional)* Más noticias: crea una clave gratuita en [finnhub.io](https://finnhub.io) y guárdala como
   secreto `FINNHUB_API_KEY` (**Settings → Secrets and variables → Actions**).

## Tus propias acciones

Escribe tus compras reales en [`config/mis_acciones.yaml`](config/mis_acciones.yaml):

```yaml
posiciones:
  - ticker: MSFT
    fecha_compra: 2026-09-15
    precio_compra: 430.50
    acciones: 5
    razones:
      - "Rebotó en la MA(99) con volumen alto"
      - "Crecimiento de Azure por encima del 30%"
    importancia: "Líder en nube e inteligencia artificial; base de mi cartera a largo plazo"
```

El reporte de seguimiento muestra cada una con su gráfico, ganancia o pérdida, estado de la tesis (🟢 vigente,
🟡 debilitada, 🔴 rota), alertas (por ejemplo, "el precio cayó bajo la MA(99)") y noticias. Si dejas las razones
vacías, el sistema las completa con su análisis.

## La red neuronal

**Cómo aprende: el maestro compra en el mínimo y vende en el máximo.** Con el historial de precios, un
"maestro" que ya conoce lo que pasó después marca como buen punto de compra cada día en que, comprando al
cierre, el precio subió al menos **8%** en los siguientes **20 días** sin caer antes más de **3%**. La red solo
ve la información disponible ese día: distancia a las MA(7)/MA(25)/MA(99), pendientes, retornos, RSI, MACD,
volatilidad, volumen, rango de 52 semanas, velas y el estado del S&P 500 (31 variables). Con cientos de miles de
ejemplos aprende a reconocer esos momentos.

- Perceptrón multicapa 31 → 128 → 64 → 32 → 1, escrito en NumPy (Adam, Leaky ReLU, parada temprana).
- **Aprendizaje lento y continuo:** cada ejecución continúa desde los pesos guardados; nunca empieza de cero.
- **Validación honesta:** los meses más recientes se reservan (con un hueco para que no haya filtraciones) y,
  además, cada día se registran sus 10 favoritas, que se comprueban 20 días hábiles después con lo que pasó
  realmente.

**¿Cuándo está lista?** Solo cuando cumple **todos** estos criterios (configurables):

| Criterio | Mínimo |
|---|---|
| Sesiones de entrenamiento | 20 |
| AUC en validación (media de 5 sesiones) | 0.60 |
| Lift del 5% mejor valorado | 1.4× |
| Ventaja de AUC sobre una regla simple (elegir las más volátiles) | +0.02 |
| Predicciones evaluadas en vivo | 60 |
| Precisión en vivo frente a la tasa base | 1.2× |

Como cada predicción tarda 20 días hábiles en comprobarse, la red necesita **al menos 5 o 6 semanas**. Puede
tardar mucho más, o no llegar nunca si no encuentra un patrón real. Mientras tanto sus predicciones aparecen
como *experimentales* y no influyen en el puntaje. Al cumplir los criterios se activa sola.

## Configuración

Todo se ajusta en [`config/config.yaml`](config/config.yaml): universo de acciones, filtros de precio y volumen,
número de oportunidades, pesos del puntaje, reglas de la cartera simulada (stop-loss, objetivo, tamaño de
posición) y parámetros de la red (maestro, arquitectura, tiempo de entrenamiento, criterios de madurez).

## Uso local

```bash
pip install -r requirements-dev.txt
python -m bolsa escanear                       # escaneo completo (necesita internet)
python -m bolsa escanear --max-tickers 50      # prueba rápida con 50 acciones
python -m bolsa entrenar --minutos 30          # sesión larga de la red neuronal
python -m bolsa escanear --offline             # datos sintéticos, sin internet
python -m pytest -q                            # pruebas
```

Los reportes quedan en la carpeta `salida/`.

## Estructura

```
bolsa/
  universo.py       lista de acciones (Wikipedia, con respaldo local)
  datos.py          precios, ficha de la empresa y noticias (Yahoo) + datos sintéticos
  indicadores.py    MA(7), MA(25), MA(99), RSI, MACD, ATR, Bollinger
  noticias.py       noticias y sentimiento
  puntuacion.py     puntaje de potencial de subida
  cartera.py        seguimiento: razones, importancia, alertas, cartera simulada
  graficos.py       gráficos PNG
  reporte.py        páginas Markdown
  principal.py      orquestación del escaneo y del entrenamiento
  red/
    maestro.py         etiquetas: comprar en el mínimo, vender en el máximo
    caracteristicas.py variables de entrada
    neuronal.py        perceptrón multicapa en NumPy
    entrenador.py      entrenamiento incremental, evaluación en vivo y madurez
config/             configuración, tus acciones y lista de respaldo
tests/              pruebas automáticas
.github/workflows/  escaneo diario + entrenamiento semanal, y pruebas
```

## Limitaciones

- Los datos vienen de fuentes gratuitas (Yahoo Finance, Google News, Wikipedia), que pueden fallar, cambiar de
  formato o limitar peticiones. El sistema usa respaldos y sigue adelante con lo que consiga.
- El sentimiento se calcula con un léxico, no con un modelo de lenguaje: entiende titulares simples, no la ironía
  ni el contexto.
- La cartera simulada opera al precio de cierre, sin comisiones ni deslizamiento.
- Rendimientos pasados no garantizan resultados futuros.
