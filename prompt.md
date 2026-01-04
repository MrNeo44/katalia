
-- PROMPT para poder tener las metricas. 

Analiza TODO el proyecto usando el servidor MCP “katalia-metrics” y entrégame un ranking final.

REPO_PATH:
C:\Users\DiegoGaticaPizarro\Documents\repo-cns\ms-parametros-soapex

Objetivo:
- Calcular métricas (LOC/SLOC/MI/CC) para TODOS los archivos .py del repo (sin tests).
- Guardar resultados en cache (use_cache=true).
- Evitar timeouts: procesa en batches de 5 archivos y con max_workers=1.
- NO uses Git churn (churn_mode="none" siempre).
- Al final, genera un ranking Top 20 usando el tool rank_python_files.

Instrucciones exactas (hazlo tú sin preguntarme):
1) Llama:
   katalia_metrics_ping
   y verifica radon_available. Si radon_available=false, sigue igual pero usa use_radon=false.
2) Llama:
   katalia_metrics_list_python_files con:
     repo_path=REPO_PATH
     include_tests=false
   Obtén la lista completa “files”.
3) Divide la lista en batches de tamaño 5.
4) Para cada batch, llama:
   katalia_metrics_analyze_python_batch con:
     repo_path=REPO_PATH
     files=<batch>
     include_tests=false
     churn_mode="none"
     use_cache=true
     force_recompute=false
     require_radon=false
     use_radon=<true si radon_available, si no false>
     summary_only=true
     max_workers=1
     max_file_bytes=1000000
   Si algún batch falla/timeoutea, reinténtalo automáticamente dividiéndolo a la mitad.
   Si sigue fallando, bájalo a batches de 1 archivo y continúa. No te detengas.
5) Cuando termines todos los batches, llama:
   katalia_metrics_rank_python_files con:
     repo_path=REPO_PATH
     top_k=20
     mi_threshold=65
     include_errors=false
6) Respóndeme SOLO con:
   - Total de archivos .py encontrados, analizados OK y fallidos.
   - Tabla Top 20: rank, path, score, cc_max, MI, LOC, reason.
   - Top 5: 1–2 líneas por archivo con sugerencia de refactor (sin escribir código, solo idea).

Importante:
- No pegues el JSON completo de cada batch.
- No uses Git.
- Si un archivo es demasiado grande y queda con error file_too_large, sigue igual.

--------------------------------------------

Usa el MCP "katalia-metrics" SOLO para medir métricas del repo (sin refactor).

REPO:

C:\Users\DiegoGaticaPizarro\Documents\repo-cns\ms-parametros-soapex

1) ping()

2) list_python_files(repo_path=REPO, include_tests=false)

3) procesar en batches de 5 usando analyze_python_batch(... use_cache=true, max_workers=1, churn_mode="none", use_radon=<según radon_available>)

4) al final export_metrics_json(repo_path=REPO) y rank_python_files(repo_path=REPO, top_k=20, mi_threshold=65)

5) imprime:

- totales (.py, ok, fallidos)

- rutas de metrics_python.json y hotspots_python.json

- tabla top 20: rank | path | score | cc_max | MI | LOC | reason

Sin pegar JSON completos.