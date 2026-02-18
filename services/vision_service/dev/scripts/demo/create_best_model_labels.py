import pickle
from pathlib import Path

# Lista PROPORCIONADA POR EL USUARIO (Orden estricto)
LABELS = [
    'hospital', 'si', 'duro', 'lunes', 'perro', 'cansado', 'ayer', 'yo', 'nosotros',
    'beber', 'ambulancia', 'infeccion', 'ojo', 'no', 'pregunta', 'duda', 'bien',
    'mal', 'suave', 'normal', 'frio', 'caliente', 'mejor', 'peor', 'estresado',
    'rapido', 'lento', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado',
    'domingo', 'gato', 'camaron', 'pollo', 'abeja', 'confundido', 'ahora', 'hoy',
    'manana', 'nunca', 'siempre', 'diario', 'mama', 'papa', 'esposo', 'esposa',
    'hijo', 'hija', 'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
    'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '10', 'como', 'cuantos', 'para_que', 'por_que', 'cocinar',
    'recibir', 'estudiar', 'interpretar', 'ir', 'no_ver', 'dormir', 'pelear',
    'trabajar', 'descansar', 'comer', 'correr', 'caminar', 'jarabe', 'virus',
    'aborto', 'accidente', 'doctor', 'enfermera', 'enfermero', 'paciente',
    'enfermo', 'terapia', 'pastillas', 'inyeccion', 'contagiar', 'revisar',
    'calentura', 'cancer', 'infarto', 'lesion', 'embarazo', 'sangre', 'gripa',
    'garganta', 'tos', 'debil', 'huesos', 'farmacia', 'emergencia', 'inflamacion',
    'analisis', 'coronavirus', 'cita', 'dolor', 'fractura', 'urgencia', 'orina',
    'popo', 'mareo', 'vomito', 'convulciones', 'gases' ,'diarrea', 'moco', 'sed',
    'nariz', 'oreja', 'boca', 'cuello', 'hombro', 'espalda', 'brazo', 'codo',
    'muneca', 'mano', 'panza', 'cintura', 'pene', 'vagina', 'piernas', 'rodilla',
    'tobillo', 'pie'
]

# Crear diccionario ID -> Label
id_to_label = {i: label for i, label in enumerate(LABELS)}

# Guardar en archivo (Ruta relativa desde CWD)
OUTPUT_PATH = Path("services/vision_service/models/best_model_labels.pkl")

print(f"Guardando {len(id_to_label)} etiquetas en: {OUTPUT_PATH}")

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(id_to_label, f)

print("✅ Archivo creado exitosamente.")
print(f"   ID 0: {id_to_label[0]}")
print(f"   ID 149: {id_to_label[149]}")
