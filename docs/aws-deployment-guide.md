# 🚀 Guía de Deployment en AWS - SignSpeak

## 📖 Introducción: ¿Qué vamos a hacer?

Vamos a desplegar tu aplicación SignSpeak en AWS siguiendo las mejores prácticas de DevOps.

---

## 🏗️ Arquitectura del Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DESARROLLO LOCAL                          │
│  ┌─────────────┐                                                    │
│  │   Docker    │──── docker build ────┐                             │
│  │   Image     │                      │                             │
│  └─────────────┘                      ▼                             │
└───────────────────────────────────────│─────────────────────────────┘
                                        │
                                   docker push
                                        │
┌───────────────────────────────────────▼─────────────────────────────┐
│                              AWS CLOUD                               │
│                                                                      │
│  ┌──────────────────┐                                               │
│  │   ECR            │  ◄── Repositorio de imágenes Docker           │
│  │   (Container     │      (como Docker Hub pero privado de AWS)    │
│  │    Registry)     │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           │ pull image                                               │
│           ▼                                                          │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │   EC2            │     │   S3             │                      │
│  │   (Servidor)     │     │   (Storage)      │                      │
│  │                  │     │                  │                      │
│  │  ┌────────────┐  │     │  - Modelos ML    │                      │
│  │  │  Docker    │  │     │  - Backups       │                      │
│  │  │  Container │  │     │                  │                      │
│  │  └────────────┘  │     └──────────────────┘                      │
│  └──────────────────┘                                               │
│           │                                                          │
│           │ :8002                                                    │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │   Internet       │  ◄── Usuarios acceden a tu API                │
│  │   (Puerto 8002)  │                                               │
│  └──────────────────┘                                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Servicios AWS que Usaremos

### 1. **ECR (Elastic Container Registry)**

- **Qué es**: Repositorio privado para imágenes Docker
- **Por qué**: Para guardar tu imagen `signspeak/vision-service` de forma segura
- **Costo**: ~$0.10/GB/mes (Free Tier: 500MB gratis)
- **Alternativa**: Docker Hub (público o privado)

### 2. **EC2 (Elastic Compute Cloud)**

- **Qué es**: Servidor virtual en la nube
- **Por qué**: Para ejecutar tu contenedor Docker
- **Costo**: t2.micro es GRATIS (Free Tier, 750 hrs/mes primer año)
- **Nota**: Para ML pesado, necesitarías t3.medium (~$30/mes)

### 3. **S3 (Simple Storage Service)** - Opcional

- **Qué es**: Almacenamiento de archivos
- **Por qué**: Para guardar modelos ML, backups, logs
- **Costo**: ~$0.023/GB/mes (Free Tier: 5GB gratis)

### 4. **RDS (Relational Database Service)** - Opcional/Futuro

- **Qué es**: Base de datos administrada (PostgreSQL)
- **Por qué**: Para persistir datos de usuarios, traducciones
- **Costo**: db.t3.micro GRATIS (Free Tier: 750 hrs/mes)

---

## 🔄 Flujo de Deployment (CI/CD)

```
    DESARROLLO                    CI/CD                         PRODUCCIÓN
    ──────────                    ─────                         ──────────

    1. Escribes código
           │
           ▼
    2. git push ─────────────► GitHub Actions detecta push
                                      │
                                      ▼
                               3. Ejecuta tests
                                      │
                                      ▼
                               4. docker build
                                      │
                                      ▼
                               5. docker push ──────────────► ECR
                                                                │
                                                                ▼
                                                         6. EC2 hace pull
                                                                │
                                                                ▼
                                                         7. Contenedor corriendo
                                                                │
                                                                ▼
                                                         8. API disponible 🎉
```

---

## 📋 Pasos Detallados que Vamos a Seguir

### FASE 3A: Configuración Inicial ✅ (COMPLETADA)

- [x] Instalar AWS CLI
- [x] Crear Access Keys
- [x] Configurar `aws configure`
- [x] Verificar conexión

### FASE 3B: ECR (Container Registry)

- [ ] Crear repositorio ECR
- [ ] Autenticar Docker con ECR
- [ ] Etiquetar imagen local
- [ ] Subir imagen a ECR

### FASE 3C: S3 (Storage para Modelos)

- [ ] Crear bucket S3
- [ ] Subir modelos ML como backup
- [ ] Configurar permisos

### FASE 3D: EC2 (Servidor)

- [ ] Crear Security Group (firewall)
- [ ] Crear instancia EC2
- [ ] Instalar Docker en EC2
- [ ] Configurar conexión SSH
- [ ] Desplegar contenedor

### FASE 4: CI/CD (Automatización)

- [ ] Crear GitHub Actions workflow
- [ ] Configurar secrets en GitHub
- [ ] Automatizar deploy

---

## 🔐 Mejores Prácticas DevOps que Seguiremos

### 1. **Infrastructure as Code (IaC)**

No haremos click en la consola manualmente (excepto la primera vez).
Guardaremos comandos en scripts reutilizables.

### 2. **Principio de Mínimo Privilegio**

Cada servicio solo tendrá los permisos que necesita.

### 3. **Secrets Management**

Nunca guardaremos credenciales en el código.
Usaremos variables de entorno y AWS Secrets Manager.

### 4. **Tagging**

Todos los recursos tendrán tags para identificarlos:

- `Project: signspeak`
- `Environment: production`
- `Owner: alan`

### 5. **Monitoreo**

Configuraremos logs y alertas básicas.

---

## 💰 Estimación de Costos

| Servicio      | Free Tier          | Después de Free Tier |
| ------------- | ------------------ | -------------------- |
| EC2 t2.micro  | 750 hrs/mes GRATIS | ~$8.50/mes           |
| ECR           | 500MB GRATIS       | ~$0.10/GB            |
| S3            | 5GB GRATIS         | ~$0.023/GB           |
| Data Transfer | 100GB GRATIS       | ~$0.09/GB            |
| **TOTAL**     | **$0/mes**         | **~$10-15/mes**      |

> ⚠️ El Free Tier dura 12 meses desde que creaste la cuenta.

---

## 🚀 ¿Listos para empezar?

Siguiente paso: **FASE 3B - Crear repositorio ECR**

Esto implica:

1. Crear el repositorio en AWS
2. Autenticar Docker con ECR
3. Subir tu imagen

¿Procedemos?
