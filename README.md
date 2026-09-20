<p align="center">
  <img src="assets/brand/jeronimo_mark.png" alt="Jerónimo Abya Yala" width="150">
</p>

<h1 align="center">Jerónimo Abya Yala</h1>

<p align="center">
  <strong>Transcripción, diarización y revisión de entrevistas con prioridad en el procesamiento local y el control de los datos.</strong>
</p>

<p align="center">
  <img src="assets/textures/territorio_hero.png" alt="Territorio Vivo — identidad visual de Jerónimo Abya Yala" width="900">
</p>

## ¿Qué es?

**Jerónimo Abya Yala** es una aplicación de escritorio orientada al trabajo con entrevistas y materiales de investigación cualitativa. Nació como herramienta de apoyo para procesos de desgrabación, revisión y organización de entrevistas, con especial atención a la **soberanía tecnológica**, la **privacidad** y la posibilidad de mantener los datos en el equipo del usuario.

El proyecto fue desarrollado para acompañar el trabajo con entrevistas del equipo **Jóvenes y Escuela Secundaria**, vinculado al Instituto de Investigaciones en Ciencias de la Educación (IICE), Facultad de Filosofía y Letras, Universidad de Buenos Aires. Esta referencia describe el contexto de origen del proyecto y no implica que se trate de un producto oficial de la Universidad.

## Funciones principales

- Transcripción local de audio y video.
- Diarización de hablantes mediante **WhisperX + pyannote**.
- Perfil local sin hablantes mediante **faster-whisper**.
- Procesamiento de texto con modelos locales mediante **Ollama**.
- Uso opcional de OpenAI o APIs compatibles cuando el usuario decide trabajar con servicios externos.
- Editor de entrevistas sincronizado con audio para revisar y corregir transcripciones.
- Exportación y revisión de resultados desde una interfaz unificada.
- Diagnóstico del equipo y gestión de modelos locales.
- Interfaz visual **Territorio Vivo**, desarrollada específicamente para el proyecto.

## Filosofía de trabajo

Jerónimo intenta que el procesamiento sensible pueda realizarse **localmente**. Esto implica trasladar parte del costo computacional al equipo del usuario: los modelos pueden requerir varios gigabytes de almacenamiento, memoria y capacidad de procesamiento, y una GPU NVIDIA compatible puede acelerar de forma importante determinadas tareas.

Los servicios externos son opcionales. Cuando una función implica que audio o texto salga del equipo, la aplicación lo informa y requiere una configuración explícita del usuario.

## Estado actual

**Versión 0.9.3 — pre-release para Windows 10/11 x64.**

El proyecto continúa en desarrollo y validación. Antes de trabajar con material único o sensible se recomienda conservar siempre una copia de los archivos originales.

## Nota del desarrollador

> La decisión de priorizar herramientas locales no busca solamente independencia técnica. También intenta que quien trabaja con entrevistas pueda conocer dónde están sus archivos, qué modelo los procesa y cuándo un dato sale de su computadora. Esa elección tiene un costo: exige más recursos del equipo y obliga a cuidar compatibilidad, rendimiento y almacenamiento. Jerónimo Abya Yala nace de ese equilibrio entre utilidad, autonomía y responsabilidad sobre los datos.
>
> Si encontrás un problema o una situación que pueda mejorar, documentarla ayuda directamente al desarrollo del proyecto.
>
> **Con cariño, José Manuel.**

## Desarrollo asistido por IA

El desarrollo de Jerónimo Abya Yala contó con **ChatGPT, de OpenAI, como herramienta de apoyo** durante distintas etapas de análisis, planificación, generación y revisión de código, diseño de pruebas y documentación técnica.

ChatGPT fue utilizado como asistente de desarrollo. Las decisiones de diseño, selección de componentes, integración, validación, pruebas y publicación del proyecto corresponden al desarrollador.

## Autor y proyecto

**José Manuel · Chaos Reigns**  
GitHub: [@jmvelezo](https://github.com/jmvelezo)

---

*Jerónimo Abya Yala — tecnología local, datos bajo tu control.*
