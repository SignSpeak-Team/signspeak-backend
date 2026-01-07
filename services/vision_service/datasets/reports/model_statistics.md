# Estadisticas de Modelos - SignSpeak

## Resumen

| Modelo | Capas | Parametros | Clases | Input Shape |
|--------|-------|------------|--------|-------------|
| Abecedario Estatico (Dense) | 4 | 17,813 | 21 | (None, 63) |
| Letras Dinamicas (LSTM) | 10 | 187,430 | 6 | (None, 30, 63) |

---

## Abecedario Estatico (Dense)

- **Archivo**: `sign_model.keras`
- **Capas**: 4
- **Parametros totales**: 17,813
- **Input shape**: (None, 63)
- **Output shape**: (None, 21)
- **Numero de clases**: 21
- **Clases**: A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y

---

## Letras Dinamicas (LSTM)

- **Archivo**: `lstm_letters.keras`
- **Capas**: 10
- **Parametros totales**: 187,430
- **Input shape**: (None, 30, 63)
- **Output shape**: (None, 6)
- **Numero de clases**: 6
- **Clases**: J, K, Q, X, Z, Ñ

---

