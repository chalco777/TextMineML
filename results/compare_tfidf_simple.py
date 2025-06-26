#!/usr/bin/env python3
import sys, os, joblib
import numpy as np

if len(sys.argv) != 3:
    print(f"Uso: {sys.argv[0]} archivo1.pkl archivo2.pkl")
    sys.exit(1)

f1, f2 = sys.argv[1], sys.argv[2]

# Tamaños en bytes
s1, s2 = os.path.getsize(f1), os.path.getsize(f2)
print("Tamaños:", f1, "=", s1, "bytes;", f2, "=", s2, "bytes")

# Carga matrices
m1 = joblib.load(f1)
m2 = joblib.load(f2)

# Dimensiones y nnz
d1, d2 = getattr(m1, "shape", None), getattr(m2, "shape", None)
nnz1 = m1.nnz if hasattr(m1, "nnz") else np.count_nonzero(m1)
nnz2 = m2.nnz if hasattr(m2, "nnz") else np.count_nonzero(m2)
print("Shapes:", d1, "vs", d2)
print("Non-zero elements:", nnz1, "vs", nnz2)

# Comparación numérica si tienen igual shape
if d1 == d2:
    A1 = m1.toarray() if hasattr(m1, "toarray") else np.array(m1)
    A2 = m2.toarray() if hasattr(m2, "toarray") else np.array(m2)
    print("Contenido igual (tol=1e-6)?", np.allclose(A1, A2, atol=1e-6))
else:
    print("No se comparó contenido: shapes diferentes.")
