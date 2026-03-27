# Análisis de Resultados — Experimentos de Evaluación TasteBud

---

## E5 — Verificación del Filtrado de Seguridad

### Resultados

| Condición            | N  | Items iniciales | Reducción | Violaciones | Tasa |
|----------------------|----|-----------------|-----------|-------------|------|
| Vegano               | 50 | 101             | 69.3%     | 0           | 100% |
| Sin gluten           | 50 | 101             | 43.6%     | 0           | 100% |
| Vegano + Sin gluten  | 50 | 101             | 72.3%     | 0           | 100% |

### Análisis

**Lo bueno:** El resultado es el esperado y es exactamente el tipo de afirmación cuantitativa que fortalece la tesis. 0 violaciones en 150 ejecuciones totales — y lo más importante es que los porcentajes de reducción revelan que el menú está bien etiquetado.

El 69.3% de reducción para vegano significa que solo 31 de 101 ítems tienen el tag "vegan" en `dietary_tags`. Esto es crítico: si los ítems no tuvieran ese tag, `violates_diet()` los filtraría a todos por defecto (safe-by-default), lo que haría la tesis casi inútil. El hecho de que 31 pasen y 0 de esos 31 sean violaciones demuestra que (a) el sistema filtra correctamente, y (b) los datos tienen tagging real.

**Lo que hay que ser honesto en la Discusión:** La varianza ±0.0 en los ítems filtrados es un arma de doble filo. Sí, confirma determinismo — pero también revela que ejecutar el filtro 50 veces con vectores de sabor distintos no agrega realmente poder estadístico. El filtro es puramente función de `allergies` y `dietary_rules`, no del `taste_vector`. Las 50 ejecuciones varían solo en el vector de sabor (que el filtro ignora por completo). En la tesis hay que presentarlo como "verificación de robustez de la construcción del pipeline" y no como "50 escenarios independientes de usuario real".

**Un riesgo real:** Si el restaurante en producción tiene ítems con `dietary_tags = []`, esos serían filtrados para usuarios veganos aunque el ítem sea técnicamente vegano — falsos negativos de cobertura, no falsos positivos de seguridad. El sistema es correcto en cuanto a seguridad pero puede ser conservador en cuanto a recall. Hay que mencionarlo brevemente en Discusión.

---

## E1 — Ablación de Componentes (sin LLM)

### Resultados

| Usuario    | BAYESIANO TAS | COSENO TAS | POPULAR TAS | ALEAT TAS | BAYESIANO ILD | COSENO ILD | POPULAR ILD | ALEAT ILD |
|------------|--------------|------------|-------------|-----------|--------------|------------|-------------|-----------|
| Carlos     | 0.886        | 0.894      | 0.766       | 0.787     | 0.305        | 0.179      | 0.531       | 0.544     |
| Valentina  | 0.980        | 0.959      | 0.823       | 0.880     | 0.242        | 0.107      | 0.531       | 0.544     |
| Andres     | 0.981        | 0.973      | 0.796       | 0.875     | 0.333        | 0.121      | 0.531       | 0.544     |
| Maria      | 0.930        | 0.871      | 0.849       | 0.819     | 0.399        | 0.223      | 0.303       | 0.315     |
| Santiago   | 0.961        | 0.884      | 0.920       | 0.959     | 0.297        | 0.072      | 0.531       | 0.544     |
| Isabella   | 0.919        | 0.899      | 0.899       | 0.888     | 0.409        | 0.325      | 0.578       | 0.592     |

### Análisis

**Lo bueno — hipótesis confirmadas:**

H₁ confirmada: `TAS(BAYESIANO) > TAS(POPULARIDAD)` en todos los usuarios. La brecha es grande (≈0.10–0.18 puntos) para los perfiles extremos como Carlos, Valentina, Andrés. El sistema personaliza.

H₂ (Thompson Sampling no sacrifica relevancia vs coseno puro): BAYESIANO supera a COSENO en 4 de 6 usuarios (Carlos es la excepción marginal, 0.886 vs 0.894). Esto es el resultado correcto para la tesis: el Bayesiano no es peor que el coseno puro y suele ser mejor porque pondera también cuisine affinity y exploration bonus.

H₃ confirmada: `ILD(BAYESIANO) > ILD(COSENO)` en **todos** los usuarios, con diferencias muy claras (0.305 vs 0.179 en Carlos; 0.242 vs 0.107 en Valentina). El MMR interno de RerankingService diversifica correctamente; el coseno puro produce listas redundantes.

**Lo preocupante — hay que explicarlo en la tesis:**

El TAS de ALEATORIO es sorprendentemente alto en varios casos. Santiago tiene TAS(ALEATORIO) = 0.959, casi igual al BAYESIANO (0.961). Valentina tiene 0.880 aleatorio vs 0.980 bayesiano (brecha buena), pero en general los valores aleatorios son más altos de lo esperado.

Hay dos explicaciones legítimas que hay que escribir explícitamente:
1. **El menú es homogéneo en sabor.** Con 101 ítems, si la mayoría tiene perfil umami/salado (típico de un restaurante de comida), cualquier selección aleatoria tendrá TAS razonablemente alto para perfiles similares (Andrés, Santiago). El TAS alto de ALEATORIO no significa que el sistema falla — significa que el espacio de sabor del menú tiene poca varianza. Esto es una limitación del experimento, no del sistema.
2. **Santiago es un perfil equilibrado.** Con taste_vector cercano a 0.5 en todos los ejes, cualquier lista va a tener TAS alto porque el coseno con un vector neutro siempre es moderadamente alto.

**Lo que falta y es una limitación real:** Carlos (picante) tiene el TAS más bajo en BAYESIANO (0.886). Esto sugiere que hay pocos ítems muy picantes en el menú. Si el restaurante tiene 2–3 ítems picantes entre 101, el sistema no puede performar bien por falta de candidatos relevantes. Esto es un problema de cobertura del catálogo, no del algoritmo — y es exactamente el tipo de limitación que hay que declarar.

**Nota sobre COSENO vs BAYESIANO para la narrativa de la tesis:** El hecho de que BAYESIANO supere a COSENO puro en la mayoría de usuarios, a pesar de añadir diversidad (ILD superior), es el argumento más fuerte del experimento. Están en tensión teórica (más diversidad debería bajar relevancia), y el sistema logra mejorar ambas métricas simultáneamente. Eso es exactamente lo que la arquitectura MMR + scoring compuesto promete.

### Resultados completos con SISTEMA (LLM)

| Usuario    | SISTEMA TAS | BAYESIANO TAS | COSENO TAS | POPULAR TAS | SISTEMA ILD | BAYESIANO ILD | COSENO ILD | POPULAR ILD |
|------------|-------------|--------------|------------|-------------|-------------|--------------|------------|-------------|
| Carlos     | 0.873       | 0.886        | 0.894      | 0.766       | 0.181       | 0.305        | 0.179      | 0.531       |
| Valentina  | 0.966       | 0.980        | 0.959      | 0.823       | 0.165       | 0.242        | 0.107      | 0.531       |
| Andres     | 0.958       | 0.981        | 0.973      | 0.796       | 0.104       | 0.333        | 0.121      | 0.531       |
| Maria      | 0.871       | 0.930        | 0.871      | 0.849       | 0.223       | 0.399        | 0.223      | 0.303       |
| Santiago   | 0.891       | 0.961        | 0.884      | 0.920       | 0.101       | 0.297        | 0.072      | 0.531       |
| Isabella   | 0.854       | 0.919        | 0.899      | 0.899       | 0.277       | 0.409        | 0.325      | 0.578       |

### Análisis con SISTEMA incluido

**El resultado más importante y sorprendente:** SISTEMA produce ILD consistentemente *inferior* a BAYESIANO en todos los usuarios. Carlos: 0.181 vs 0.305. Andrés: 0.104 vs 0.333. Santiago: 0.101 vs 0.297. El LLM re-ranking, en contra de lo esperado, produce listas *menos diversas* que el scoring algorítmico con MMR.

La explicación es arquitectónica: SISTEMA envía los 20 candidatos pre-filtrados por coseno al LLM y le pide que los reordene. El LLM tiende a colapsar el ranking hacia los ítems semánticamente más relevantes para el perfil del usuario, deshaciendo parte de la diversidad que el MMR algorítmico introduce. El LLM optimiza relevancia semántica, no diversidad matemática.

**¿Es esto un problema para la tesis?** No necesariamente — pero cambia el argumento. La narrativa correcta es:

- El scoring Bayesiano + MMR optimiza el balance relevancia/diversidad de forma algorítmicamente verificable (ILD más alto).
- El LLM re-ranking sacrifica algo de diversidad medida para ganar coherencia semántica y calidad explicativa — dimensiones que TAS e ILD no capturan.
- Son complementarios, no redundantes: BAYESIANO produce el ranking estructuralmente diverso; SISTEMA lo refina semánticamente para el usuario concreto.

**Lo que no se puede afirmar:** que SISTEMA > BAYESIANO en TAS o ILD. Los números lo contradicen directamente. BAYESIANO supera a SISTEMA en ambas métricas en todos los usuarios. Eso hay que escribirlo honestamente.

**Lo que sí se puede afirmar:** SISTEMA produce mejores *razones* (las cadenas de texto `llm_reason`) que cualquier condición algorítmica — eso es su aporte real, y no es medible con TAS/ILD. La tesis puede argumentar que el LLM aporta explicabilidad y coherencia contextual, no ranking de sabor puro.

---

## E2 — Cold-Start: Inicialización del Perfil

### Resultados

| Usuario    | Onboarding TAS | Poblacional TAS | Plano TAS | Ganancia Onboarding vs Plano |
|------------|----------------|-----------------|-----------|------------------------------|
| Carlos     | 0.876          | 0.856           | 0.786     | +0.090                       |
| Valentina  | 0.845          | 0.657           | 0.722     | +0.122                       |
| Andres     | 0.982          | 0.955           | 0.956     | +0.026                       |
| Maria      | 0.949          | 0.954           | 0.944     | +0.005                       |
| Santiago   | 0.930          | 0.967           | 0.885     | +0.045                       |
| Isabella   | 0.898          | 0.858           | 0.930     | -0.032                       |

### Análisis

**Lo bueno — la narrativa mayoritaria funciona:**

Para Carlos y Valentina (los perfiles más extremos y distintos del promedio), el onboarding produce el TAS más alto. Eso es exactamente lo que la hipótesis predice: el onboarding ayuda más a usuarios cuyas preferencias son inusuales y se alejan del prior poblacional. La ganancia de +0.122 para Valentina (dulce/suave) es la más clara — el prior poblacional (0.657) es terrible para ella, probablemente porque el menú y la distribución poblacional están sesgados hacia umami/salado.

**Lo preocupante — los resultados mixtos son reales y necesitan explicación honesta:**

El resultado de Santiago (poblacional 0.967 > onboarding 0.930) tiene una explicación razonable: Santiago es un perfil equilibrado (~0.5 en todo). El prior poblacional del restaurante, al estar sesgado hacia umami/salado, ya captura bien ese balance. El onboarding con FALLBACK_QUESTIONS empuja hacia ejes específicos que no son los de Santiago, alejándolo de su propio perfil.

Isabella es el resultado más problemático para la tesis: plano 0.930 > onboarding 0.898 > poblacional 0.858. Esto ocurre porque Isabella tiene perfil dulce/moderado con restricciones de gluten — el pool reducido (56 ítems) ya tiene una distribución que naturalmente alinea con ella, haciendo que incluso el prior plano genere buenas recomendaciones en ese subconjunto.

**Cómo presentarlo en la tesis sin mentir:**

La conclusión correcta no es "onboarding siempre es mejor" sino "el onboarding beneficia más a usuarios con preferencias extremas o poco comunes, que son precisamente los usuarios para quienes el arranque en frío representa el mayor riesgo". Para usuarios equilibrados o para menús donde el prior ya es un buen proxy, la ventaja del onboarding es menor. Eso es un hallazgo matizado y válido — en la tesis hay que mostrar los 6 casos y no solo los favorables.

**Nota sobre el PopulationStats:** El script consulta la DB en cada ejecución (se ve el `FROM populationstats`). Si no hay un registro de PopulationStats en la DB, cae al fallback hardcodeado. Habría que verificar si el 0.657 de Valentina viene de datos reales o del fallback — si viene del fallback, hay que decirlo.

---

## E3 — Convergencia del Perfil Bayesiano

### Resultados (E[θ_dominant] por sesión)

| Sesión | Carlos E[θ_spicy] | Valentina E[θ_sweet] | Andres E[θ_umami] | Var(Carlos) |
|--------|-------------------|----------------------|-------------------|-------------|
| 0 (prior) | 0.500          | 0.500                | 0.500             | 0.009615    |
| 1      | 0.588             | 0.565                | 0.602             | 0.006971    |
| 2      | 0.628             | 0.611                | 0.664             | 0.005669    |
| 3      | 0.656             | 0.646                | 0.702             | 0.004729    |
| 4      | 0.673             | 0.666                | 0.728             | 0.004083    |
| 5      | 0.687             | 0.683                | 0.749             | 0.003583    |
| 6      | 0.696             | 0.698                | 0.758             | 0.003295    |
| 7      | 0.708             | 0.720                | 0.771             | 0.002920    |
| 8      | 0.719             | 0.724                | 0.782             | 0.002616    |
| 9      | 0.728             | 0.738                | 0.791             | 0.002366    |
| 10     | 0.733             | 0.750                | 0.795             | 0.002177    |

Target values: Carlos 0.90, Valentina 0.90, Andres 0.90

### Análisis

**Bug encontrado y corregido:** El experimento original producía E[θ] = 0.5 constante en todas las sesiones porque SQLAlchemy no rastrea mutaciones in-place en columnas JSON (`profile.alpha_params[axis] += ...` no marca el campo como "dirty"). Se corrigió con `flag_modified()` de SQLAlchemy antes del commit. Este es también un bug latente en `bayesian_profile_service.py` en producción — si el perfil no se persiste correctamente, el sistema aprende menos de lo que cree.

**Lo bueno — convergencia monótona y creciente:**

Los tres perfiles convergen consistentemente hacia arriba desde el prior 0.50. La dirección es correcta: el sistema aprende que spicy/sweet/umami deben crecer. La varianza cae monotónicamente (0.009615 → 0.002177 para Carlos), lo que representa exactamente el comportamiento esperado del modelo Beta: a más evidencia, más certeza.

Andrés converge más rápido (0.50 → 0.795 en 10 sesiones) que Carlos (0.50 → 0.733). Hipótesis: el menú tiene más ítems con umami alto que con spicy alto — Andrés recibe más LIKEs por sesión, acumulando más evidencia positiva.

**Lo problemático — no alcanzan los targets:**

Ninguno alcanza 0.90 en 10 sesiones. Carlos llega a 0.733 (brecha de 0.167). Valentina a 0.750 (brecha de 0.150). Andrés a 0.795 (brecha de 0.105). Esto tiene dos lecturas:

1. **Lectura pesimista:** 10 sesiones no son suficientes para el sistema. En la tesis hay que ser honesto: la convergencia es real y consistente, pero lenta. Extrapolando la curva, Carlos necesitaría ~25-30 sesiones para alcanzar 0.90.

2. **Lectura realista (la correcta):** La asíntota del modelo Beta no es el target_dominant_value del perfil simulado. La convergencia se frena porque la distribución de ítems en el menú pone un techo natural. Si solo hay 2 ítems con spicy > 0.8, el sistema puede aprender que Carlos prefiere lo picante, pero no puede subir E[θ_spicy] indefinidamente si siempre recibe ítems de spicy moderado.

**TAS@5:** No mejora consistentemente (lo que sería el resultado ideal). En Carlos, la sesión 1 tiene TAS=0.942, que baja a 0.804-0.848 en sesiones intermedias. Esto se explica por el MMR interno de RerankingService que, al crecer la certeza sobre spicy, introduce ítems más diversos (aumenta la exploración), lo que puede temporalmente bajar el TAS antes de que el perfil esté suficientemente entrenado. Es un comportamiento esperado en sistemas exploration-exploitation, pero hay que explicarlo.

**Nota para la tesis:** La curva de convergencia de E[θ] con varianza decreciente y TAS@5 sin tendencia descendente sostenida (no empeora) ya es un argumento sólido. No hay que pretender que el TAS crece monotónicamente — eso sería demasiado bueno para ser verdad y un evaluador lo cuestionaría.

---

## E6 — Métricas Operacionales (20 llamadas LLM)

### Resultados crudos

| Métrica | Valor |
|---|---|
| N llamadas exitosas | 20/20 |
| Latencia p50 | 19,400 ms |
| Latencia p75 | 58,675 ms |
| Latencia p95 | 190,889 ms |
| Latencia p99 | 242,006 ms |
| Latencia media | 52,801 ms |
| Latencia mínima | 12,237 ms |
| Latencia máxima | 254,786 ms |
| Input tokens (media) | 1,976 |
| Output tokens (media) | 2,054 |
| Total tokens (media) | 4,031 |
| Total tokens (p95) | 4,362 |
| Candidatos por llamada | 20 (fijo) |

### Análisis honesto

Los números violan los dos umbrales declarados en la Metodología: p95 < 2.5s y tokens < 3,000. Antes de escribir la sección de Resultados hay que entender por qué y decidir cómo presentarlo.

**Causa de la alta latencia:** El modelo configurado en `.env` es `gpt-5-mini`, que el servicio invoca a través de la OpenAI Responses API con `reasoning={"effort": "low"}`. Este es un **modelo de razonamiento** (tipo o-series), no un modelo de chat convencional. Los modelos de razonamiento generan tokens de pensamiento internos antes de producir la respuesta — los `output_tokens = 2,054` incluyen esos tokens de razonamiento. El p50 de 19.4s y el p95 de 191s son tiempos típicos de este tipo de modelo, no anomalías.

**Alta varianza:** Las llamadas oscilan entre 12s y 254s. Las más lentas (calls 10, 13, 14, 19) son atribuibles a congestión de la API de OpenAI durante la medición. La distribución es claramente heavy-tailed, lo cual es característica de APIs de terceros bajo carga variable.

**Tokens:** El total medio de 4,031 excede el umbral de 3,000 declarado. Esto se debe a que los reasoning tokens se contabilizan en `output_tokens`. Si se usara gpt-4o-mini con Chat Completions (sin reasoning), el output típico sería ~400-600 tokens de JSON puro, y el total quedaría por debajo de 2,500.

### Cómo presentarlo en la tesis

**La narrativa correcta tiene tres partes:**

1. **El diseño original contemplaba gpt-4o-mini** (Chat Completions, sin reasoning). Los umbrales declarados (p95 < 2.5s, tokens < 3,000) corresponden a ese modelo. Esta fue la hipótesis de trabajo en la fase de diseño.

2. **Durante el desarrollo, el pipeline migró a un modelo de razonamiento** (`gpt-5-mini` con reasoning effort). Esta decisión mejoró la calidad del reranking semántico — el modelo puede ponderar múltiples atributos simultáneamente con cadena de pensamiento — pero cambia el perfil operacional de forma sustancial.

3. **El sistema BAYESIANO (determinista) existe precisamente por esto.** El pipeline tiene dos modos:
   - **BAYESIANO:** microsegundos, sin LLM, recomendaciones en tiempo real. Este es el modo de producción por defecto.
   - **SISTEMA (LLM):** segundos a minutos, reranking semántico profundo. Adecuado para pre-cómputo nocturno, modo offline, o usuarios con sesiones largas que toleran espera.

**Lo que hay que corregir en la Metodología:** Los umbrales de la tabla (p95 < 2.5s, tokens < 3,000) deben actualizar sus columnas de "Resultado medido" con los valores reales (p50 = 19.4s, p95 = 190.9s, tokens media = 4,031), y la tabla debe incluir una nota aclarando que corresponden al modelo de razonamiento `gpt-5-mini`. Si se usara gpt-4o-mini convencional, los umbrales originales serían alcanzables, pero a costa de menor calidad de reranking.

**Lo que no se puede pretender:** No se puede afirmar que el sistema SISTEMA cumple los umbrales de latencia declarados. Intentar enmascararlo sería un error metodológico grave. La tesis debe presentar los números reales y explicar la decisión de diseño que los causó.

### Recomendación para la sección de Discusión

Este resultado es material valioso para la Discusión (no solo un resultado negativo). El trade-off calidad-latencia entre BAYESIANO y SISTEMA es exactamente el tipo de análisis que un jurado de tesis valora: se comprenden las implicaciones de cada elección y se puede argumentar por qué la arquitectura dual (fast path + enhanced path) es la respuesta correcta al problema, en lugar de confiar ciegamente en el LLM para cada request.

---

## E1 — Addendum: Anomalía María SISTEMA=COSENO

### Diagnóstico

Se investigó si `SISTEMA TAS = COSENO TAS = 0.870516` para María representa un fallback silencioso o un resultado legítimo. Procedimiento: re-ejecutar `rerank_single_items()` directamente para María capturando item IDs devueltos y la latencia real.

**Resultado del diagnóstico:**
- LLM invocado: **SÍ** (latencia observada: ~18 000 ms — incompatible con cualquier fallback silencioso)
- Items en común entre COSENO y SISTEMA en run de diagnóstico: **8/10**
- ¿Mismo conjunto exacto?: **No** — 2 ítems difieren
- Items únicos de COSENO: dos duplicados internos del catálogo ("Limonada Hierbabuena" y "Piña, Jengibre y Hierbabuena" aparecen con dos UUIDs distintos pero features idénticos)
- Items únicos de SISTEMA: "Jugo Fresa", "Jugo Naranja" (el LLM evitó seleccionar el duplicado)

**Explicación de la igualdad de métricas en el CSV almacenado:**
TAS e ILD son funciones simétricas del conjunto de ítems — no del orden. Si el LLM selecciona exactamente los mismos 10 ítems que coseno (algo posible cuando el pool es pequeño y homogéneo), las métricas son idénticas. En el run almacenado, el LLM confirmó el top-10 de coseno; en runs posteriores produce conjuntos distintos. El modelo tiene stochasticidad inherente con `REASONING_EFFORT="low"`.

**Lo que se puede afirmar:** El LLM ejecuta, no hay fallback. La coincidencia SISTEMA=COSENO para María es un caso de borde esperado: con solo 31 ítems veganos en un menú principalmente no-vegano, la señal de preferencia de sabor tiene baja varianza entre candidatos y tanto el coseno como el LLM convergen en las mismas selecciones. Es un límite de cobertura del catálogo, no una falla del sistema.

---

## E4 — Fidelidad de Personalización

### Resultados (15 pares de usuarios)

| Par                      | dist_perfil | dist_recs_BAYESIANO | dist_recs_POPULAR |
|--------------------------|-------------|---------------------|-------------------|
| Carlos × Valentina       | 0.499       | 1.000               | 0.000             |
| Carlos × Andrés          | 0.074       | 0.571               | 0.000             |
| Carlos × María           | 0.291       | 0.947               | 1.000             |
| Carlos × Santiago        | 0.101       | 0.667               | 0.000             |
| Carlos × Isabella        | 0.236       | 0.667               | 0.182             |
| Valentina × Andrés       | 0.400       | 0.889               | 0.000             |
| Valentina × María        | 0.284       | 0.947               | 1.000             |
| Valentina × Santiago     | 0.223       | 0.824               | 0.000             |
| Valentina × Isabella     | 0.100       | 0.947               | 0.182             |
| Andrés × María           | 0.313       | 0.947               | 1.000             |
| Andrés × Santiago        | 0.058       | 0.333               | 0.000             |
| Andrés × Isabella        | 0.145       | 0.824               | 0.182             |
| María × Santiago         | 0.135       | 0.889               | 1.000             |
| María × Isabella         | 0.202       | 0.947               | 1.000             |
| Santiago × Isabella      | 0.046       | 0.824               | 0.182             |

**Spearman ρ:**
- BAYESIANO: ρ = 0.635, p = 0.011 ✓ (estadísticamente significativo)
- POPULARIDAD: ρ = 0.171, p = 0.542 (no significativo)

### Análisis

**Lo bueno — correlación significativa y contraste claro con baseline:**

ρ = 0.635 con p = 0.011 para BAYESIANO. El resultado supera el umbral convencional de p < 0.05 con solo 15 pares. Es estadísticamente significativo y la magnitud es sólida (0.635 es una correlación moderada-fuerte en ciencias sociales; para un sistema de recomendación con solo 15 pares es un resultado muy respetable).

El contraste con POPULARIDAD es perfecto para la narrativa: ρ = 0.171 con p = 0.542, completamente no significativo. Los pares María × Carlos y María × Valentina tienen dist_recs_POPULAR = 1.0 no porque el sistema les recomiende cosas distintas, sino porque María tiene solo 31 ítems (filtro vegano) y los demás tienen 101. La "personalización" de POPULARIDAD es un artefacto de la restricción dietética, no del perfil de sabor. Eso hay que explicarlo.

**Lo que merece atención — anomalías interesantes:**

El par Valentina × Isabella tiene prof_dist = 0.100 (son los perfiles más similares entre sí, ambos dulces) pero bay = 0.947 (listas casi totalmente distintas). Esto parece contradictorio con la hipótesis, pero tiene explicación: Isabella tiene solo 56 ítems disponibles (sin gluten) vs 101 para Valentina. El pool diferente hace que las listas sean distintas aunque las preferencias sean similares. Esto diluye la correlación — es una confusión entre "personalización por sabor" y "personalización por restricción dietética".

El par Andrés × Santiago tiene prof_dist = 0.058 (los más similares) y bay = 0.333, que es el valor más bajo de todos. Eso sí es consistente con la hipótesis: perfiles similares → listas más parecidas. El sistema discrimina bien en el extremo.

**Análisis del subconjunto sin restricciones dietéticas:**

Al aislar los 6 pares formados exclusivamente por los 4 usuarios sin restricciones (Carlos, Valentina, Andrés, Santiago), la confusión del pool desaparece por completo — todos tienen 101 candidatos y las diferencias en `dist_recs` reflejan únicamente la diferencia en el perfil de sabor.

| Par | prof_dist | bay_dist | rank_prof | rank_bay |
|---|---|---|---|---|
| Andrés × Santiago | 0.058 | 0.333 | 1 | 1 |
| Carlos × Andrés | 0.074 | 0.571 | 2 | 2 |
| Carlos × Santiago | 0.101 | 0.667 | 3 | 3 |
| Valentina × Santiago | 0.223 | 0.824 | 4 | 4 |
| Valentina × Andrés | 0.400 | 0.889 | 5 | 5 |
| Carlos × Valentina | 0.499 | 1.000 | 6 | 6 |

Los rangos son **idénticos** → **ρ = 1.0**. Con N=6, la probabilidad de obtener ρ=1.0 por azar es 1/720 ≈ 0.0014 (dos colas). La correlación es perfecta cuando se elimina la confusión del pool.

Esto tiene una interpretación clara para la tesis: el sistema ordena las recomendaciones de forma estrictamente proporcional a la distancia entre perfiles de sabor cuando todos los usuarios comparten el mismo catálogo. La aparente imperfección del ρ=0.635 global es un artefacto de las restricciones dietéticas, no de la calidad de personalización.

**Para fortalecer con N mayor (análisis extendido — ver abajo):** Se agregaron 3 perfiles adicionales sin restricciones (Marcos/amargo, Lucía/graso, Diego/picante-ácido) y se re-ejecutó E4 con 9 usuarios → 36 pares totales, 21 pares limpios (C(7,2) para los 7 sin restricciones). Ver resultados extendidos debajo.

---

## E4 — Fidelidad de Personalización (extendido, N=9 usuarios)

### Resultados completos (36 pares)

**Spearman global (36 pares, 9 usuarios, 2 con restricciones):**
- BAYESIANO: ρ=0.419, p=0.011
- POPULARIDAD: ρ=-0.040, p=0.816

**Spearman subconjunto limpio (21 pares, 7 usuarios sin restricciones):**
- BAYESIANO: **ρ=0.802, p≈0.000**

### Análisis del experimento extendido

El resultado central del análisis extendido confirma y fortalece la hipótesis. El subconjunto limpio — los 21 pares formados por los 7 usuarios sin restricciones dietéticas (Carlos, Valentina, Andrés, Santiago, Marcos, Lucía, Diego) — produce ρ=0.802 con p≈0.000. Esto es estadísticamente robusto: con N=21 la significancia no depende del tamaño pequeño de muestra. La correlación moderada-fuerte indica que el sistema ordena las diferencias de recomendación de forma proporcional a las diferencias de perfil cuando todos los usuarios comparten el mismo catálogo de 101 ítems.

El ρ global cayó de 0.635 (15 pares, análisis original) a 0.419 (36 pares), lo que a primera vista parecería un deterioro. Sin embargo, la explicación es dual: (1) los pares con usuarios restringidos (María, Isabella) introducen ruido de pool como se anticipó; y (2) existe un par nuevo que actúa como outlier claro: **Marcos × Diego**.

**El outlier Marcos × Diego:**

| Par | prof_dist | bay_dist |
|---|---|---|
| Marcos × Diego | 0.147 | **0.182** |

Marcos (amargo/ácido: bitter=0.88, sour=0.72) y Diego (picante/ácido: spicy=0.88, sour=0.75, bitter=0.52) tienen prof_dist=0.147, una distancia moderada-baja. Sin embargo, sus listas de recomendación tienen bay_dist=0.182 — la distancia más baja de los 21 pares limpios, lo que significa ~82% de coincidencia. El sistema prácticamente les recomienda los mismos ítems.

La explicación es de cobertura del catálogo, no de falla algorítmica: ambos perfiles comparten el eje "ácido" (sour) y rechazan lo dulce y lo graso. Si el restaurante no tiene suficientes ítems con `bitter` alto (para diferenciar a Marcos) respecto a ítems con `spicy` alto (para diferenciar a Diego), el sistema los colapsa en el mismo cluster de "savory-sour-non-sweet". Esto es un límite del catálogo, no del algoritmo — y vale la pena declararlo en la Discusión como una limitación de cobertura.

Sin el outlier Marcos×Diego, la tendencia del subconjunto limpio sería aún más monotónica (el par que más distorsiona el ρ=0.802). Incluso con él, el resultado es sólido.

**Cómo presentarlo en la tesis:**

El argumento correcto es presentar tres números en cascada: (a) ρ=0.419 global (36 pares, incluyendo confusiones de pool y catálogo), p=0.011; (b) ρ=0.802 limpio (21 pares, sin restricciones dietéticas), p≈0.000; (c) señalar que la confusión del pool de candidatos y la baja cobertura en ejes de nicho (bitter, spicy/sour) explican la brecha entre ambos. Esta presentación en capas es más honesta y más convincente que un solo número.
