"""Render measured offline evidence into the existing Spanish audit directory."""
import json
from pathlib import Path
from audio_diag.report import to_markdown

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/phase4'
VAULT = Path('/home/cancio/Documentos/Obsidian/Personal/ProdMusic/_Plan_Maestro/_Historial_Auditorias')


def load(name):
    return json.loads((DATA / (name + '.json')).read_text())


def main():
    manifest, comparison, controls = load('manifest'), load('opus-comparison'), load('controlled-checks')
    assert manifest['opus_count'] == comparison['count'] == 7
    assert manifest['wav_count'] == 3 and all(controls['assertions'].values())
    lines = ['# Fase 4 — Incremento offline B/D/E', '',
             '**EN CURSO. Toda aceptación humana sigue PENDIENTE.** Sin escucha, Gemini, descargas EBU ni interacción con OMNi. Revisión independiente del incremento pendiente; el hijo de revisión previo fue interrumpido y no produjo un dictamen utilizable.', '',
             '## B2/B3 — Siete perfiles individuales', '',
             'Normalización previa algebraica a potencia espectral total unitaria: P_banda/P_total cancela g². Prueba con ganancias 0,0001; 0,1; 1; 13. No es igualación LUFS. Dispersión min/max por banda, sin promedio ni sonido objetivo; extremos pueden proceder de temas distintos. Ningún valor absoluto Opus de LUFS/TP/crest se usa como objetivo.', '',
             '| ID | Archivo exacto | SHA-256 |', '|---|---|---|']
    for p in manifest['opus_mapping']:
        lines.append(f"| {p['id']} | `{p['path']}` | `{p['sha256']}` |")
    lines += ['', 'Metadatos locales, URL de procedencia y hash de cada `.info.json`: `data/phase4/manifest.json`. No se consultaron servicios externos. Identidad de versión por escucha NO VERIFICADA.', '',
              '| Banda | Mínimo dB relativo | Máximo dB relativo | N |', '|---|---:|---:|---:|']
    for name, d in comparison['dispersion'].items():
        lines.append(f"| {name} | {d['min_db']:.4f} | {d['max_db']:.4f} | {d['known_count']} |")
    lines += ['', '## B5 — Tres WAV de proyecto, no objetivos de acabado', '',
              '| Archivo / SHA | LUFS integrado | TP dBTP | Crest global heurístico dB |', '|---|---:|---:|---:|']
    for i in range(1, 4):
        r = load(f'wav-{i}')
        lines.append(f"| `{r['file']['path']}` / `{r['file']['sha256']}` | {r['level1']['loudness']['integrated_lufs']:.4f} | {r['level1']['peaks']['true_peak_dbtp']:.4f} | {r['level2']['crest_factor']['global_db']:.4f} |")
    lines += ['', 'Perfiles individuales y crest por banda completos: `wav-1.json`, `wav-2.json`, `wav-3.json`; resumen `wav-profiles.json`. Se conservan individualmente, no se convierten en rango de mastering.', '',
              '## D1/D2 y E1', '',
              'Plantilla y tabla de evidencia/contraindicaciones: [[INFORME_PRODUCCION]]. Ejemplo completo atado a hash: [[Fase4_Produccion_Master]]. B1 y B4 siguen pendientes en [[Fase4_Escucha_Referencias]].', '',
              '## E2 — Controles independientes', '',
              'Fragmento de 24 s desde segundo 30; originales de Música abiertos solo en lectura. WAV generados únicamente en `/home/cancio/.hermes/cache/scratch/opencode/phase4-bde-v2/`. Recetas y argv exactos: `data/phase4/controlled-checks.json`. Cada variante parte del limpio, no de otra degradación.', '',
              '- Corte: highpass 250 Hz, 2 polos; se espera menos sub relativo.',
              '- Fase: DFT rectangular del fragmento entero, invertir únicamente bins R <120 Hz, iDFT; L y bins superiores se preservan. Límites: periodicidad implícita, ringing en extremos/corte; no es crossover de producción. Readback float32 frente a transformación exacta: error máximo 5,82e-8; L idéntico. Prueba sintética con 60 Hz + 2 kHz verifica polaridad y conservación de agudos.',
              '- Aspereza: EQ 3200 Hz, ancho 1,3 octavas, +15 dB; se espera aumento de banda media amplia, no diagnóstico específico de aspereza percibida.',
              '- Clipping: multiplicar por 8 y limitar explícitamente a ±1; se esperan más runs consecutivos.',
              '- Sección: volumen cero de 8 a 12 s; interior 8,1–11,9 s cero con energía original no nula, exterior antes de 7,9 y después de 12,1 idéntico; pista localizada en 8,128 s, 10,65 beats. El limpio ya tenía un break en 9,568 s: no basta comparar diccionarios.',
              '- Break intencional: mismo audio que sección silenciada y mismo SHA, intención declarada; heurístico, no puerta de fallo. Ningún algoritmo puede inferir intención de ondas idénticas.', '',
              '### Fallo conservado y corrección', '',
              '`controlled-checks-v1-failed.json` conserva el fallo original. La resta de un lowpass causal introduce rotación de fase, no inversión exacta: mono loss −0,3863 → −3,1183 dB, insuficiente para el criterio original >3 dB. Se corrigió la síntesis, no el detector ni el umbral. Con inversión exacta: −0,3863 → −10,4358 dB. Los WAV v1 permanecen en scratch `phase4-bde/`.', '',
              'Los filtros independientes tienen efectos colaterales: la inversión produce 5 runs ≥2 sobre escala completa, el EQ 13637 y el corte 7 muestras aisladas sobre escala completa. Son PCM float sin clamp salvo el control clipping; no atribuir todos los cambios a una única causa perceptiva. No se ha demostrado utilidad musical completa.', '',
              '| Aserción observable | Resultado |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in controls['assertions'].items()]
    lines += ['', '## Verificación y pendientes', '',
              '- `python3 -m scripts.phase4_offline`: PASS, 7 Opus + 3 WAV + máster completo + 7 análisis de excerpt; SHA antes/después coincidentes. JSON estricto sin NaN/inf.',
              '- `python3 -m unittest discover` con timeout 600 s: 57 tests PASS (última ejecución 229,949 s).',
              '- `python3 -m compileall -q audio_diag scripts tests`: PASS. Whitespace: `git diff --check` y `git diff --no-index --check /dev/null FILE` sobre 28 archivos (incluye untracked): PASS. JSON estricto: PASS. Type/lint no configurados; ruff, mypy y pyright no disponibles, NO EJECUTADOS.',
              '- Pendientes: B1/B4 escucha y anotaciones; coherencia/aceptación humana; revisión independiente; C1/C2 Gemini y dataset EBU sin autorización. Fase NO cerrada.', '',
              'Todos los JSON de medición son locales y git-ignored bajo `/home/cancio/Code/active/audio-diag/data/phase4/`.']
    (VAULT / 'Fase4_BDE_Informe.md').write_text('\n'.join(lines) + '\n')
    r = load('master')
    header = ['# Ejemplo de informe de producción — master existente', '',
              '**Coherencia musical NO VERIFICADA. Aceptación humana PENDIENTE.**', '',
              '- Escucha: ninguna; medición offline del archivo completo (198,418753 s).',
              '- Versión medida: archivo existente, audio-diag 0.1.0; SHA indicado abajo. No se conoce versión del proyecto que lo originó.',
              '- BPM: 158 declarado por ruta/contrato y usado como parámetro; no verificado por escucha ni estimado.',
              '- Estructura/compases: DESCONOCIDOS; tonalidad NO DETERMINABLE del máster.',
              '- Fuentes/samples y cadenas reales de dispositivos: DESCONOCIDOS hasta readback; no se accedió a OMNi.',
              '- Decisiones intencionadas del máster: no declaradas; no se inventa intención para silencios o contrastes.',
              '- Opinión LLM: ninguna crítica musical realizada. Hipótesis de corrección requieren evidencia y escucha: [[INFORME_PRODUCCION]].',
              '- Evidencia: `data/phase4/master.json`; comando `python3 -m scripts.phase4_offline` → `analyze(MASTER, bpm=158)`.',
              '- Puertas: LUFS fuera de −8 ±2; TP supera −0,3 en solo 0,00447 dB (margen menor que diferencias habituales entre implementaciones; no afirmar defecto audible). Clipping 0.',
              '- Cola bajo −60 dBFS: 9,738 s, medida, intención desconocida; no etiquetarla como fallo musical. LRA y crest no prueban punch ni coherencia.',
              '- Limitaciones: solo estéreo, heurísticas de arreglo no son defectos, nivel 3 no determinable; sin stems ni readback ni escucha. No se corrigió ni re-renderizó el proyecto.', '',
              '## Medición íntegra (evidencia, no opinión)', '']
    (VAULT / 'Fase4_Produccion_Master.md').write_text('\n'.join(header) + to_markdown(r))


if __name__ == '__main__':
    main()
