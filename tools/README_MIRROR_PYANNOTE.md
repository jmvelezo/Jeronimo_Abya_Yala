# Community-1 — mirror verificable de Jerónimo Abya Yala

## Objetivo

Jerónimo instala `pyannote/speaker-diarization-community-1` como una copia local,
versionada y verificable. El uso posterior del modelo es local: no requiere token
ni conexión. El modo Automático sigue este orden:

1. copia local ya validada;
2. mirror de Jerónimo en Google Drive;
3. repositorio oficial de Hugging Face, fijado a la misma revisión, como contingencia.

La credencial de Hugging Face no forma parte del ZIP del mirror ni de su
descriptor. Google Drive sólo aloja el artefacto binario; Jerónimo confía en el
SHA-256 fijado dentro de `pyannote_mirror_release.json`, no en el nombre del
archivo ni en un checksum descargado desde el mismo Drive.

## Fase 1 — obtener y congelar el release oficial

En una PC autorizada, desde la raíz de Jerónimo:

```powershell
.\.venv\Scripts\python.exe .\tools\preparar_mirror_pyannote.py
```

Se consulta exclusivamente el repositorio upstream oficial, se congela su commit,
se verifica que siga declarando `cc-by-4.0`, se preservan README/atribuciones y se
generan en `tools/_mirror_output`:

- `jeronimo-pyannote-community-1-<revision>.zip`
- `...zip.sha256`
- `...manifest.json`
- `...release-template.json`

Subir **sólo el ZIP canónico** a Google Drive. Conservar los otros tres archivos
como registro de mantenimiento.

## Fase 2 — fijar Google Drive como fuente primaria

En Drive, compartir el ZIP como **Cualquier persona que tenga el enlace → Lector**.
Luego ejecutar:

```powershell
.\.venv\Scripts\python.exe .\tools\configurar_mirror_pyannote_drive.py --drive-url "PEGA_AQUI_EL_ENLACE_COMPARTIDO"
```

La herramienta vuelve a calcular el SHA-256 local, lee la revisión desde el
manifiesto del ZIP y genera en la raíz:

`pyannote_mirror_release.json`

Ese descriptor contiene únicamente metadatos públicos del artefacto: revisión,
SHA-256, tamaño, nombre y Google Drive file ID. No contiene contraseñas ni tokens.

Al recompilar la BASE, `build/build_portable.ps1` copia automáticamente ese
descriptor junto al ejecutable.

## Instalación en una PC del equipo

Antes de descargar el runtime pesado de diarización, Jerónimo intenta preparar
Community-1. Si Drive responde, verifica tamaño + SHA-256 del ZIP, extrae en un
directorio temporal, verifica los hashes internos del manifiesto y recién entonces
instala en:

`runtime/models/pyannote/community-1`

Si Drive falla, devuelve HTML, entrega bytes incompletos o un SHA incorrecto, el
archivo se descarta y se intenta el repositorio oficial de Hugging Face. Cuando
existe un release de mirror configurado, ese respaldo solicita la **misma revisión
congelada**, no `main`.

## Regla de mantenimiento

Nunca reemplazar en Drive los bytes de un release ya publicado conservando el
mismo descriptor. Para una versión nueva del upstream:

1. volver a ejecutar Fase 1;
2. auditar el nuevo manifiesto/licencias;
3. subir un ZIP nuevo;
4. ejecutar Fase 2 con el enlace nuevo;
5. recompilar Jerónimo.

La siguiente fase del proyecto añadirá la contingencia de diarización alternativa
cuando ni el mirror ni el upstream oficial estén disponibles. Esa contingencia no
forma parte de esta Fase 2 para no cambiar simultáneamente el algoritmo de
separación de hablantes.
