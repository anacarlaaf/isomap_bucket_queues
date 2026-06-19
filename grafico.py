import pandas as pd
import matplotlib.pyplot as plt

# =========================
# leitura do CSV
# =========================

df = pd.read_csv("res_roll.csv")

# garante ordenação correta por tamanho do grafo
df = df.sort_values(by="n")

print(df[["n", "m", "std_time", "bucket_time", "speedup"]])

# ========================= 
# gráfico 1: tempos         
# ========================= 

plt.figure()

plt.plot(df["n"], df["std_time"], marker="o", label="std priority_queue")
plt.plot(df["n"], df["bucket_time"], marker="o", label="bucket queue")

plt.xlabel("Número de nós (n)")
plt.ylabel("Tempo (s)")
plt.title("Comparação de tempo: Dijkstra all-pairs")
plt.legend()
plt.grid(True)

plt.show()