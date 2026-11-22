import re
import numpy as np
import matplotlib.pyplot as plt

def parse_secuencial(filename):
    sec = {}
    pattern = re.compile(r"n=(\d+).*?T_total=([\d\.]+)")
    with open(filename, 'r') as f:
        for line in f:
            if "SECUENCIAL" in line:
                m = pattern.search(line)
                if m:
                    n, t = int(m.group(1)), float(m.group(2))
                    sec.setdefault(n, []).append(t)
    return {n: np.mean(times) for n, times in sec.items()}

def parse_mpi(filename):
    # mpi[n][p] = {'total': [], 'comp': [], 'comm': [], 'gflops': []}
    mpi = {}
    pattern = re.compile(r"p=(\d+),\s*n=(\d+).*?T_total=([\d\.]+),\s*T_comp=([\d\.]+),\s*T_comm=([\d\.]+),\s*GFLOPs/s=([\d\.]+)")
    with open(filename, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                p, n = int(m.group(1)), int(m.group(2))
                tt, tc, tcm, gf = float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))
                mpi.setdefault(n, {}).setdefault(p, {'total': [], 'comp': [], 'comm': [], 'gflops': []})
                mpi[n][p]['total'].append(tt)
                mpi[n][p]['comp'].append(tc)
                mpi[n][p]['comm'].append(tcm)
                mpi[n][p]['gflops'].append(gf)
    
    # Calcular promedios
    avg = {}
    for n in mpi:
        avg[n] = {}
        for p in mpi[n]:
            avg[n][p] = {
                'total': np.mean(mpi[n][p]['total']),
                'comp': np.mean(mpi[n][p]['comp']),
                'comm': np.mean(mpi[n][p]['comm']),
                'gflops': np.mean(mpi[n][p]['gflops'])
            }
    return avg

t_seq = parse_secuencial("resultados_secuenciales.txt")
mpi_data = parse_mpi("resultados_mpi_61969.txt")

p_vals = [1, 2, 4, 8, 16]
colores = {512: '#1f77b4', 1000: '#ff7f0e', 1024: '#2ca02c', 2048: '#d62728', 4096: '#9467bd'}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Gráfica de Speedup Relativo Canónico (Strong Scaling)
plt.figure(figsize=(8, 6))
plt.plot(p_vals, p_vals, 'k--', label='Speedup Ideal ($S_p = p$)', alpha=0.7)
for n in sorted(mpi_data.keys()):
    if 1 in mpi_data[n]:
        t1 = mpi_data[n][1]['total']
        sp = [t1 / mpi_data[n][p]['total'] for p in p_vals if p in mpi_data[n]]
        plt.plot(p_vals[:len(sp)], sp, marker='o', label=f'N = {n}', color=colores.get(n, None))
plt.title('Speedup Fuerte Relativo ($S_p$) vs. Procesos ($p$)', fontsize=13, fontweight='bold')
plt.xlabel('Número de Procesos ($p$)', fontsize=11)
plt.ylabel('Speedup Relativo ($S_p = T_1 / T_p$)', fontsize=11)
plt.xticks(p_vals)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_speedup.png', dpi=300)
plt.close()

# 2. Gráfica de Eficiencia Paralela Relativa
plt.figure(figsize=(8, 6))
plt.axhline(1.0, color='k', linestyle='--', label='Eficiencia Ideal ($E_p = 1.0$)', alpha=0.7)
for n in sorted(mpi_data.keys()):
    if 1 in mpi_data[n]:
        t1 = mpi_data[n][1]['total']
        ep = [(t1 / mpi_data[n][p]['total']) / p for p in p_vals if p in mpi_data[n]]
        plt.plot(p_vals[:len(ep)], ep, marker='s', label=f'N = {n}', color=colores.get(n, None))
plt.title('Eficiencia Paralela ($E_p$) vs. Procesos ($p$)', fontsize=13, fontweight='bold')
plt.xlabel('Número de Procesos ($p$)', fontsize=11)
plt.ylabel('Eficiencia ($E_p = S_p / p$)', fontsize=11)
plt.xticks(p_vals)
plt.ylim(0, 1.2)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_eficiencia.png', dpi=300)
plt.close()

# 3. Gráfica de Rendimiento en GFLOPs/s
plt.figure(figsize=(8, 6))
for n in sorted(mpi_data.keys()):
    gf = [mpi_data[n][p]['gflops'] for p in p_vals if p in mpi_data[n]]
    plt.plot(p_vals[:len(gf)], gf, marker='^', label=f'N = {n}', color=colores.get(n, None))
plt.title('Rendimiento Computacional (GFLOPs/s) vs. Procesos ($p$)', fontsize=13, fontweight='bold')
plt.xlabel('Número de Procesos ($p$)', fontsize=11)
plt.ylabel('Throughput (GFLOPs/s)', fontsize=11)
plt.xticks(p_vals)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_gflops.png', dpi=300)
plt.close()

# 4. Desglose de Cómputo vs Comunicación (N=4096)
if 4096 in mpi_data:
    plt.figure(figsize=(8, 6))
    t_comp_vals = [mpi_data[4096][p]['comp'] for p in p_vals]
    t_comm_vals = [mpi_data[4096][p]['comm'] for p in p_vals]
    
    x = np.arange(len(p_vals))
    width = 0.35
    plt.bar(x - width/2, t_comp_vals, width, label='Tiempo Cómputo ($T_{comp}$)', color='#2ca02c')
    plt.bar(x + width/2, t_comm_vals, width, label='Tiempo Comunicación ($T_{comm}$)', color='#d62728')
    plt.xticks(x, [f'p={p}' for p in p_vals])
    plt.title('Desglose de Tiempo: Cómputo vs. Comunicación ($N=4096$)', fontsize=13, fontweight='bold')
    plt.ylabel('Tiempo promedio (segundos)', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafica_desglose_4096.png', dpi=300)
    plt.close()

# 5. Comparación: Tiempo Experimental vs. Complejidad Teórica Esperada
# Calibramos c1 (halo: n/sqrt(p)) y c2 (reducción: log2(p)) con el régimen sin saturación (p in [2, 4, 8])
X_fit, y_fit = [], []
for n in [1024, 2048, 4096]:
    t1 = mpi_data[n][1]['total']
    for p in [2, 4, 8]:
        overhead = max(0.0, mpi_data[n][p]['total'] - (t1 / p))
        X_fit.append([n / np.sqrt(p), np.log2(p)])
        y_fit.append(overhead)

c1, c2 = np.linalg.lstsq(np.array(X_fit), np.array(y_fit), rcond=None)[0]

plt.figure(figsize=(8, 6))
for n in [1024, 2048, 4096]:
    t1 = mpi_data[n][1]['total']
    t_exp = [mpi_data[n][p]['total'] for p in p_vals]
    t_teo = [(t1 / p) + (c1 * (n / np.sqrt(p)) + c2 * np.log2(p) if p > 1 else 0.0) for p in p_vals]
    
    plt.plot(p_vals, t_exp, marker='o', linestyle='-', label=f'Experimental (N={n})', color=colores[n])
    plt.plot(p_vals, t_teo, marker='x', linestyle='--', label=f'Teórico $\\alpha$-$\\beta$ (N={n})', color=colores[n], alpha=0.75)

plt.title('Tiempo Experimental vs. Complejidad Teórica Esperada', fontsize=13, fontweight='bold')
plt.xlabel('Número de Procesos ($p$)', fontsize=11)
plt.ylabel('Tiempo de Ejecución $T_p$ (segundos)', fontsize=11)
plt.xticks(p_vals)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig('grafica_teorico_vs_exp.png', dpi=300)
plt.close()

print("¡Gráficas generadas exitosamente!")