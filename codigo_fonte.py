"""
=============================================================
 SIGIC — Sistema Inteligente de Gerenciamento da
         Infraestrutura da Colônia
         Missão Aurora Siger — Fase 4
=============================================================
Implementa grafos, BFS, DFS, Dijkstra, estruturas de dados
e modelagem matemática com cálculo diferencial aplicado.

Bibliotecas: numpy, sympy, collections.deque
"""

import numpy as np
import sympy as sp
from collections import deque

# =============================================================
# 1. DEFINIÇÃO DA INFRAESTRUTURA — MÓDULOS E ATRIBUTOS
# =============================================================

# Tuplas: dados fixos de cada módulo (imutáveis)
# Formato: (nome, prioridade, consumo_kw, capacidade_kwh, status)
MODULOS_INFO = [
    ("Habitação",             1, 15.0, 80.0,  "ativo"),
    ("Centro de Controle",    2, 12.0, 50.0,  "ativo"),
    ("Suporte Médico",        3, 10.0, 40.0,  "ativo"),
    ("Armazenamento Energia", 4,  5.0, 200.0, "ativo"),
    ("Produção de Oxigênio",  5,  8.0, 30.0,  "ativo"),
    ("Laboratório Científico",6,  9.0, 45.0,  "ativo"),
    ("Agricultura",           7,  7.0, 60.0,  "manutenção"),
    ("Comunicação",           8,  6.0, 25.0,  "ativo"),
]

N = len(MODULOS_INFO)

# Dicionário: acesso rápido por nome
modulos_dict = {m[0]: {
    "id":             i,
    "nome":           m[0],
    "prioridade":     m[1],
    "consumo_kw":     m[2],
    "capacidade_kwh": m[3],
    "status":         m[4],
    "freq_comunicacao": round(np.random.uniform(1, 10), 1),
} for i, m in enumerate(MODULOS_INFO)}

# Lista de módulos ordenada por ID
modulos_lista = list(modulos_dict.values())

# =============================================================
# 2. REPRESENTAÇÃO DA REDE — GRAFO PONDERADO
# =============================================================
# Arestas: (módulo_a, módulo_b, distância_metros)
ARESTAS = [
    ("Habitação",             "Centro de Controle",     50),
    ("Habitação",             "Suporte Médico",         80),
    ("Habitação",             "Produção de Oxigênio",   60),
    ("Centro de Controle",    "Armazenamento Energia",  40),
    ("Centro de Controle",    "Laboratório Científico", 70),
    ("Centro de Controle",    "Comunicação",            55),
    ("Suporte Médico",        "Armazenamento Energia",  45),
    ("Armazenamento Energia", "Produção de Oxigênio",   35),
    ("Armazenamento Energia", "Agricultura",            90),
    ("Armazenamento Energia", "Laboratório Científico", 65),
    ("Produção de Oxigênio",  "Agricultura",            75),
    ("Laboratório Científico","Comunicação",            50),
    ("Agricultura",           "Comunicação",            85),
]

# ── 2.1 Matriz de adjacência ──────────────────────────────────
# Justificativa: verifica existência de aresta em O(1).
# Adequada para a rede da colônia (grafo relativamente denso).
def criar_matriz_adjacencia():
    mat = [[0] * N for _ in range(N)]
    for a, b, w in ARESTAS:
        i = modulos_dict[a]["id"]
        j = modulos_dict[b]["id"]
        mat[i][j] = w
        mat[j][i] = w
    return mat

# ── 2.2 Lista de adjacência ───────────────────────────────────
# Justificativa: armazena apenas conexões existentes — O(V + E)
# para BFS/DFS. Ideal para percursos em grafos esparsos.
def criar_lista_adjacencia():
    adj = {i: [] for i in range(N)}
    for a, b, w in ARESTAS:
        i = modulos_dict[a]["id"]
        j = modulos_dict[b]["id"]
        adj[i].append((j, w))
        adj[j].append((i, w))
    return adj

MATRIZ   = criar_matriz_adjacencia()
LISTA_ADJ = criar_lista_adjacencia()

# =============================================================
# 3. ALGORITMOS DE BUSCA
# =============================================================

# ── 3.1 BFS — Busca em Largura ────────────────────────────────
def bfs(origem_id):
    """
    Percorre por camadas usando fila (deque) e matriz de adjacência.
    Retorna ordem de visita e parent[] para reconstrução do caminho.
    """
    visited = [False] * N
    parent  = [-1]   * N
    ordem   = []

    fila = deque()
    visited[origem_id] = True
    fila.append(origem_id)

    while fila:
        u = fila.popleft()
        ordem.append(u)

        # Varre a linha da matriz para encontrar vizinhos
        for v in range(N):
            if MATRIZ[u][v] > 0 and not visited[v]:
                visited[v] = True
                parent[v]  = u
                fila.append(v)

    return ordem, parent

def reconstruir_caminho(parent, origem_id, destino_id):
    """
    Reconstrói caminho usando parent[] — reutilizável para BFS e Dijkstra.
    """
    caminho = []
    v = destino_id
    while v != -1:
        caminho.append(v)
        if v == origem_id:
            break
        v = parent[v]

    if not caminho or caminho[-1] != origem_id:
        return []

    caminho.reverse()
    return caminho

# ── 3.2 DFS — Busca em Profundidade ──────────────────────────
def dfs(u, visited, ordem, adj):
    """
    DFS recursiva com lista de adjacência.
    Avança o máximo possível antes de retornar (backtracking).
    """
    visited[u] = True
    ordem.append(u)
    for v, _ in adj[u]:
        if not visited[v]:
            dfs(v, visited, ordem, adj)

def executar_dfs(origem_id):
    visited = [False] * N
    ordem   = []
    dfs(origem_id, visited, ordem, LISTA_ADJ)
    return ordem

# ── 3.3 Dijkstra — Caminho Mínimo Ponderado ──────────────────
INF = float('inf')

def dijkstra(origem_id):
    """
    Dijkstra O(V²): seleciona menor dist[] por varredura + relaxamento.
    Retorna dist[] e parent[] para reconstrução do caminho.
    """
    dist   = [INF]   * N
    parent = [-1]    * N
    usado  = [False] * N

    dist[origem_id] = 0

    for _ in range(N):
        # Seleciona vértice não finalizado com menor dist[]
        u = -1
        for i in range(N):
            if not usado[i] and dist[i] < INF:
                if u == -1 or dist[i] < dist[u]:
                    u = i

        if u == -1:
            break

        usado[u] = True

        # Relaxamento das arestas saindo de u
        for v in range(N):
            w = MATRIZ[u][v]
            if w > 0 and not usado[v]:
                if dist[u] + w < dist[v]:
                    dist[v]   = dist[u] + w
                    parent[v] = u

    return dist, parent

# =============================================================
# 4. DETECÇÃO DE CONEXÕES CRÍTICAS
# =============================================================

def detectar_conexoes_criticas():
    """
    Remove cada módulo e verifica se a rede fica desconectada via DFS.
    """
    criticos = []
    for removido in range(N):
        visited = [False] * N
        visited[removido] = True
        inicio = next((i for i in range(N) if not visited[i]), -1)
        if inicio == -1:
            continue
        componente = []
        dfs(inicio, visited, componente, LISTA_ADJ)
        if len(componente) < N - 1:
            criticos.append(modulos_lista[removido]["nome"])
    return criticos

# =============================================================
# 5. ESTRUTURAS DE DADOS
# =============================================================

def exibir_estruturas():
    sep = "=" * 60
    print("\n" + sep)
    print("ESTRUTURAS DE DADOS UTILIZADAS")
    print(sep)

    print("\nTUPLAS — dados imutáveis de cada módulo:")
    print("  (nome, prioridade, consumo_kw, capacidade_kwh, status)")
    for t in MODULOS_INFO[:3]:
        print(f"  {t}")
    print("  ...")

    print("\nDICIONÁRIO — acesso por nome de módulo (ex: Habitação):")
    m = modulos_dict["Habitação"]
    for k, v in m.items():
        print(f"  '{k}': {v}")

    print("\nLISTA DE ADJACÊNCIA — vizinhos do módulo 0 (Habitação):")
    for v, w in LISTA_ADJ[0]:
        print(f"  -> {modulos_lista[v]['nome']} ({w} m)")

    print("\nMATRIZ DE ADJACÊNCIA (3x3 — primeiros módulos):")
    nomes = [modulos_lista[i]["nome"][:12] for i in range(3)]
    print(f"  {'':14}", end="")
    for n in nomes:
        print(f"{n:14}", end="")
    print()
    for i in range(3):
        print(f"  {nomes[i]:14}", end="")
        for j in range(3):
            print(f"{MATRIZ[i][j]:<14}", end="")
        print()

# =============================================================
# 6. MODELAGEM MATEMÁTICA — CÁLCULO DIFERENCIAL
# =============================================================

def modelagem_matematica():
    """
    Modela o consumo energético E(n) em função do número de módulos.

    Função quadrática:
      E(n) = a*n² + b*n + c
      n  = número de módulos ativos
      a  = fator de crescimento acelerado (interdependências)
      b  = consumo base por módulo (kW)
      c  = consumo fixo de infraestrutura (kW)
    """
    n = sp.Symbol('n', positive=True)

    consumo_medio = np.mean([m["consumo_kw"] for m in modulos_lista])
    a = sp.Rational(3, 10)
    b = sp.Rational(int(consumo_medio * 10), 10)
    c = sp.Integer(20)

    E      = a * n**2 + b * n + c
    dE_dn  = sp.diff(E, n)
    d2E_dn2 = sp.diff(E, n, 2)

    # Otimização por gradiente descendente numérico (numpy)
    f_num  = sp.lambdify(n, E, 'numpy')
    n_val  = 8.0
    alpha  = 0.01
    eps    = 1e-5

    historico = [n_val]
    for _ in range(200):
        grad  = (f_num(n_val + eps) - f_num(n_val - eps)) / (2 * eps)
        n_val = n_val - alpha * grad
        n_val = max(1.0, n_val)
        historico.append(n_val)

    return {
        "funcao":           E,
        "derivada":         dE_dn,
        "segunda_derivada": d2E_dn2,
        "n_otimo":          round(n_val, 2),
        "consumo_otimo":    round(float(f_num(n_val)), 2),
    }

# =============================================================
# 7. ANÁLISE ENERGÉTICA
# =============================================================

def analisar_eficiencia():
    ativos = [m for m in modulos_lista if m["status"] == "ativo"]
    total_consumo    = sum(m["consumo_kw"] for m in ativos)
    total_capacidade = sum(m["capacidade_kwh"] for m in modulos_lista)
    maior = max(ativos, key=lambda m: m["consumo_kw"])
    menor = min(ativos, key=lambda m: m["consumo_kw"])
    return {
        "modulos_ativos":  len(ativos),
        "total_consumo":   total_consumo,
        "total_capacidade":total_capacidade,
        "eficiencia_pct":  round((1 - total_consumo / total_capacidade) * 100, 1),
        "maior_consumo":   maior,
        "menor_consumo":   menor,
    }

# =============================================================
# 8. MENU INTERATIVO
# =============================================================

def imprimir_cabecalho():
    print("\n" + "=" * 60)
    print("  SIGIC — Sistema Inteligente de Gerenciamento da")
    print("          Infraestrutura da Colônia Aurora Siger")
    print("=" * 60)

def imprimir_menu():
    print("\n  [1]  Visualizar rede da colônia")
    print("  [2]  Consultar módulos")
    print("  [3]  BFS — Busca em Largura")
    print("  [4]  DFS — Busca em Profundidade")
    print("  [5]  Dijkstra — Caminho mínimo")
    print("  [6]  Detectar conexões críticas")
    print("  [7]  Estruturas de dados")
    print("  [8]  Modelagem matemática")
    print("  [9]  Análise energética")
    print("  [0]  Sair\n")

def selecionar_modulo(prompt="  Módulo (ID): "):
    print()
    for m in modulos_lista:
        print(f"    [{m['id']}] {m['nome']}")
    while True:
        try:
            idx = int(input(prompt))
            if 0 <= idx < N:
                return idx
            print("  ID inválido.")
        except ValueError:
            print("  Digite um número.")

def opcao_1():
    print("\n" + "-" * 60)
    print("REDE DA COLÔNIA")
    print(f"\n  Módulos : {N}  |  Conexões: {len(ARESTAS)}\n")
    for a, b, w in ARESTAS:
        print(f"  {a:<25} <-> {b:<25}  {w} m")

def opcao_2():
    print("\n" + "-" * 60)
    print("MÓDULOS DA COLÔNIA")
    for m in modulos_lista:
        print(f"\n  [{m['id']}] {m['nome']}")
        print(f"       Prioridade       : {m['prioridade']}")
        print(f"       Consumo          : {m['consumo_kw']} kW")
        print(f"       Capacidade       : {m['capacidade_kwh']} kWh")
        print(f"       Status           : {m['status']}")
        print(f"       Freq. comunicação: {m['freq_comunicacao']}/10")

def opcao_3():
    print("\n  BFS percorre a rede por camadas (menor número de saltos).")
    origem  = selecionar_modulo("  Origem (ID): ")
    destino = selecionar_modulo("  Destino (ID): ")
    ordem, parent = bfs(origem)
    caminho = reconstruir_caminho(parent, origem, destino)
    print(f"\n  Ordem BFS a partir de '{modulos_lista[origem]['nome']}':")
    for i, uid in enumerate(ordem):
        print(f"    {i+1}. {modulos_lista[uid]['nome']}")
    print(f"\n  Caminho BFS até '{modulos_lista[destino]['nome']}':")
    if caminho:
        print("    " + " -> ".join(modulos_lista[v]["nome"] for v in caminho))
    else:
        print("    Sem caminho disponível.")

def opcao_4():
    print("\n  DFS avança o máximo possível antes de retornar (backtracking).")
    origem = selecionar_modulo("  Origem (ID): ")
    ordem  = executar_dfs(origem)
    print(f"\n  Ordem DFS a partir de '{modulos_lista[origem]['nome']}':")
    for i, uid in enumerate(ordem):
        print(f"    {i+1}. {modulos_lista[uid]['nome']}")

def opcao_5():
    print("\n  Dijkstra calcula a menor distância total (em metros).")
    origem  = selecionar_modulo("  Origem (ID): ")
    destino = selecionar_modulo("  Destino (ID): ")
    dist, parent = dijkstra(origem)
    caminho = reconstruir_caminho(parent, origem, destino)
    print(f"\n  Distâncias a partir de '{modulos_lista[origem]['nome']}':")
    for i, d in enumerate(dist):
        d_str = f"{d} m" if d < INF else "inacessível"
        print(f"    {modulos_lista[i]['nome']:<30}: {d_str}")
    print(f"\n  Melhor rota até '{modulos_lista[destino]['nome']}':")
    if caminho:
        print("    " + " -> ".join(modulos_lista[v]["nome"] for v in caminho))
        print(f"    Distância total: {dist[destino]} m")
    else:
        print("    Sem rota disponível.")

def opcao_6():
    print("\n  Verificando módulos críticos da rede...")
    criticos = detectar_conexoes_criticas()
    print("\n  Módulos cuja remoção desconectaria a rede:")
    if criticos:
        for nome in criticos:
            print(f"    [CRITICO] {nome}")
    else:
        print("    Nenhum — a rede é totalmente redundante.")

def opcao_7():
    exibir_estruturas()

def opcao_8():
    print("\n  Calculando modelo de consumo energético...")
    r = modelagem_matematica()
    print("\n" + "-" * 60)
    print("MODELAGEM MATEMÁTICA — CONSUMO ENERGÉTICO")
    print("-" * 60)
    print(f"\n  Função:           E(n) = {r['funcao']}")
    print(f"  Derivada:         dE/dn = {r['derivada']}")
    print(f"  Segunda derivada: d²E/dn² = {r['segunda_derivada']}")
    print(f"\n  Análise qualitativa:")
    print(f"    d²E/dn² > 0: curva côncava para cima")
    print(f"    O consumo cresce de forma acelerada com novos módulos")
    print(f"    Cada módulo adicional aumenta o consumo marginal")
    print(f"\n  Gradiente descendente (otimização numérica):")
    print(f"    Número ótimo de módulos : {r['n_otimo']}")
    print(f"    Consumo no ponto ótimo  : {r['consumo_otimo']} kW")
    print(f"\n  Decisão de engenharia:")
    print(f"    Com {r['n_otimo']} módulos ativos, a taxa de crescimento")
    print(f"    do consumo é mínima — ponto de operação mais eficiente.")

def opcao_9():
    r = analisar_eficiencia()
    print("\n" + "-" * 60)
    print("ANÁLISE ENERGÉTICA DA REDE")
    print("-" * 60)
    print(f"\n  Módulos ativos  : {r['modulos_ativos']}")
    print(f"  Consumo total   : {r['total_consumo']} kW")
    print(f"  Capacidade total: {r['total_capacidade']} kWh")
    print(f"  Eficiência      : {r['eficiencia_pct']}%")
    print(f"\n  Maior consumo: {r['maior_consumo']['nome']}"
          f" ({r['maior_consumo']['consumo_kw']} kW)")
    print(f"  Menor consumo: {r['menor_consumo']['nome']}"
          f" ({r['menor_consumo']['consumo_kw']} kW)")

# =============================================================
# 9. EXECUÇÃO
# =============================================================

ACOES = {
    "1": opcao_1, "2": opcao_2, "3": opcao_3,
    "4": opcao_4, "5": opcao_5, "6": opcao_6,
    "7": opcao_7, "8": opcao_8, "9": opcao_9,
}

def executar():
    imprimir_cabecalho()
    while True:
        imprimir_menu()
        opcao = input("  Escolha uma opção: ").strip()
        if opcao == "0":
            print("\n  Encerrando o SIGIC. Até logo.\n")
            break
        elif opcao in ACOES:
            ACOES[opcao]()
        else:
            print("\n  Opção inválida.")

if __name__ == "__main__":
    executar()
