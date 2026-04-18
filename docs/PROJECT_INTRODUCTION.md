# Introducción y Marco Teórico del Proyecto SignSpeak

## 1. Contexto y Planteamiento del Problema

El Lenguaje de Señas Mexicano (LSM) constituye el medio principal de comunicación para miles de personas en México. Sin embargo, existe una barrera comunicativa significativa entre la comunidad sorda y la población oyente que no domina este lenguaje. Esta brecha lingüística limita el acceso a servicios esenciales, oportunidades educativas y la interacción social cotidiana, generando una necesidad imperativa de herramientas tecnológicas que actúen como puentes de comunicación efectivos y accesibles.

## 2. Descripción del Sistema

SignSpeak se define como un sistema de traducción automatizada en tiempo real diseñado para interpretar el Lenguaje de Señas Mexicano y convertirlo a texto en español. El proyecto integra disciplinas avanzadas de las ciencias de la computación, específicamente la Visión Artificial (Computer Vision) y el Aprendizaje Profundo (Deep Learning), para capturar, analizar y decodificar los gestos manuales y corporales que componen la estructura gramatical del LSM.

## 3. Fundamentos Tecnológicos: Visión Artificial y Extracción de Características

El núcleo perceptivo del sistema se fundamenta en el uso de **MediaPipe**, un framework desarrollado por Google para el procesamiento de datos multimodales. SignSpeak emplea específicamente el modelo de detección holística (Holistic Tracking), el cual permite la localización simultánea de puntos de referencia (landmarks) en las manos, el rostro y la postura corporal. Esta tecnología permite abstraer la información visual del video en vectores numéricos estructurados, eliminando el ruido del fondo y centrando el análisis únicamente en la biomecánica del usuario, lo que resulta esencial para la interpretación precisa de señas que dependen tanto de la configuración manual como de la expresión no manual.

## 4. Fundamentos de Procesamiento: Redes Neuronales y Datos Secuenciales

Para la interpretación semántica de los datos extraídos, el sistema implementa modelos de redes neuronales basados en **TensorFlow** y **Keras**. Dado que el lenguaje de señas es intrínsecamente temporal (donde el significado depende de la secuencia de movimientos a lo largo del tiempo), se utilizan arquitecturas especializadas en datos secuenciales, como las redes **LSTM (Long Short-Term Memory)**. Estas redes son capaces de aprender dependencias a largo plazo, permitiendo al sistema distinguir entre gestos estáticos (como las letras del alfabeto) y dinámicos (palabras o frases completas), proporcionando una traducción que respeta el flujo natural del lenguaje.

## 5. Arquitectura de Software

Desde una perspectiva teórica de ingeniería de software, SignSpeak adopta una **arquitectura de microservicios** para garantizar la escalabilidad, mantenibilidad y desacoplamiento de sus componentes. El sistema se descompone en unidades funcionales autónomas:

- **API Gateway:** Actúa como punto de entrada unificado y gestor de tráfico.
- **Vision/ML Service:** Dedicado exclusivamente a la inferencia de modelos de inteligencia artificial.
- **Translation Service:** Orquestador de la lógica de negocio y gestión de estados.
- **Storage Service:** Responsable de la persistencia de datos multimedia.

Esta distribución permite que cada módulo evolucione independientemente y asegura que el sistema pueda manejar cargas de trabajo intensivas mediante la comunicación asíncrona, implementada a través de colas de mensajes (RabbitMQ).

## 6. Conclusión

En síntesis, SignSpeak representa la convergencia de la inteligencia artificial moderna con la necesidad social de inclusión. Al transformar señales visuales complejas en información textual comprensible, el proyecto establece un marco tecnológico sólido para la accesibilidad universal, demostrando cómo la ingeniería de software avanzada puede aplicarse directamente a la resolución de problemáticas sociales tangibles.
