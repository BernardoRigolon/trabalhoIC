#!/usr/bin/env python3
import argparse, csv, itertools, math, random, re, statistics, sys, time
import numpy as np

EPS = 1e-9
TIPOS = ["M1", "M2", "M3e", "M3s", "M3h", "M3d"]


class Instancia:
    def __init__(self, caminho):
        cids, ctipos, eids, salas, mat = [], [], [], [], []
        secao = None
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                if linha.startswith("NUM_CANDIDATOS") or linha.startswith("NUM_ESCOLAS"):
                    for chave, valor in re.findall(r"(NUM_CANDIDATOS|NUM_ESCOLAS)\s*:\s*(\d+)", linha):
                        if chave == "NUM_CANDIDATOS": self.n = int(valor)
                        else: self.m = int(valor)
                    continue
                if linha.startswith("#"):
                    if linha.startswith("# CANDIDATOS"): secao = "C"
                    elif linha.startswith("# ESCOLAS"): secao = "E"
                    elif linha.startswith("# MATRIZ"): secao = "M"
                    continue
                p = [x.strip() for x in linha.split(",")]
                if secao == "C":
                    cids.append(p[0]); ctipos.append(p[2])
                elif secao == "E":
                    eids.append(p[0]); salas.append([int(x) for x in p[5].split(";")])
                elif secao == "M":
                    mat.append([float(x) for x in p])
        assert len(cids) == self.n and len(eids) == self.m and len(mat) == self.n
        self.cids, self.eids = cids, eids
        self.D = np.array(mat)
        self.escola_salas = []
        sala_escola, sala_cap = [], []
        for s, caps in enumerate(salas):
            self.escola_salas.append(list(range(len(sala_cap), len(sala_cap) + len(caps))))
            sala_escola += [s] * len(caps); sala_cap += caps
        self.sala_escola = np.array(sala_escola); self.sala_cap = np.array(sala_cap)
        self.R = len(sala_cap)
        self.cap = np.array([sum(c) for c in salas], dtype=int)  # capacidade total de cada escola
        self.nomes_tipos = sorted(set(ctipos), key=lambda t: TIPOS.index(t) if t in TIPOS else 99)
        self.T = len(self.nomes_tipos)
        self.ctype = np.array([self.nomes_tipos.index(t) for t in ctipos])
        self.idx = [np.where(self.ctype == t)[0] for t in range(self.T)]
        self.demanda = [len(x) for x in self.idx]
        self.Dt = [self.D[x] for x in self.idx]
        self.media = [d.mean(axis=0) if len(d) else np.zeros(self.m) for d in self.Dt]
        self.lb = float(self.D.min(axis=1).sum())                # limitante trivial: escola mais proxima


def caps_tipo(inst, rtype, t):
    sel = rtype == t
    return np.bincount(inst.sala_escola[sel], weights=inst.sala_cap[sel], minlength=inst.m).astype(int)


def guloso_local(d, caps):
    """Aloca candidatos (linhas de d) a escolas (colunas) por arrependimento; None se nao couber."""
    nt, ns = d.shape
    if caps.sum() < nt:
        return None
    dm = np.where(caps > 0, d, np.inf)
    ds = np.sort(dm, axis=1)
    if ns > 1:
        reg = np.where(np.isfinite(ds[:, 1]), ds[:, 1] - ds[:, 0], 1e9)   # so uma escola possivel: urgente
    else:
        reg = np.zeros(nt)
    ordem = np.argsort(-reg, kind="stable")
    rest = caps.copy()
    a = np.empty(nt, dtype=int)
    for i in ordem:
        for j in np.argsort(dm[i]):
            if rest[j] > 0:
                a[i] = j; rest[j] -= 1
                break
    return a


def descida_local(d, caps, a):
    """N1 (realocar) e N2 (trocar), melhor vizinho por candidato, ate otimo local. Modifica a."""
    nt, ns = d.shape
    carga = np.bincount(a, minlength=ns)
    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(nt):
            cur = a[i]
            delta = d[i] - d[i, cur]
            delta[carga >= caps] = np.inf
            delta[cur] = np.inf
            j = int(np.argmin(delta))
            if delta[j] < -EPS:
                carga[cur] -= 1; carga[j] += 1; a[i] = j; melhorou = True
        for i in range(nt - 1):
            ai = a[i]
            outros = np.arange(i + 1, nt)
            outros = outros[a[outros] != ai]
            if outros.size == 0:
                continue
            ao = a[outros]
            delta = d[i, ao] + d[outros, ai] - d[i, ai] - d[outros, ao]
            k = int(np.argmin(delta))
            if delta[k] < -EPS:
                o = outros[k]; a[i], a[o] = a[o], ai; melhorou = True
    return float(d[np.arange(nt), a].sum())


def n1_lote(d, caps, a):
    nt, m = d.shape
    carga = np.bincount(a, minlength=m)
    idxs = np.arange(nt)
    while True:
        delta = d - d[idxs, a][:, None]
        delta[:, carga >= caps] = np.inf
        delta[idxs, a] = np.inf
        i, j = divmod(int(np.argmin(delta)), m)
        if delta[i, j] >= -EPS:
            break
        carga[a[i]] -= 1; carga[j] += 1; a[i] = j
    return float(d[idxs, a].sum())


def monta_tipo(inst, rtype, t):
    caps = caps_tipo(inst, rtype, t)
    d = inst.Dt[t]
    a = guloso_local(d, caps)
    if a is None:
        return None
    return {"a": a, "caps": caps, "custo": float(d[np.arange(len(a)), a].sum())}


def reparo_salas(inst, rtype, rng, max_iter=3000):
    tipos = [t for t in range(inst.T) if inst.demanda[t] > 0]
    dem, q, R = inst.demanda, inst.sala_cap, inst.R
    rt = rtype.copy()
    capt = [0] * inst.T
    for r in range(R):
        if rt[r] >= 0: capt[rt[r]] += int(q[r])
    df = lambda t, c: max(0, dem[t] - c)
    deficit = lambda: sum(df(t, capt[t]) for t in tipos)
    tabu = {}
    for it in range(max_iter):
        d0 = deficit()
        if d0 == 0:
            return rt
        movs = []
        for r in range(R):
            for t2 in tipos + [-1]:
                if t2 != rt[r]: movs.append(((r, t2),))
        for _ in range(400):                                   # amostra de trocas de tipo entre duas salas
            r1, r2 = rng.randrange(R), rng.randrange(R)
            if rt[r1] != rt[r2]: movs.append(((r1, rt[r2]), (r2, rt[r1])))
        rng.shuffle(movs)
        melhor, melhor_d = None, 10 ** 9
        for mv in movs:
            delta = {}
            for r, t2 in mv:
                if rt[r] >= 0: delta[rt[r]] = delta.get(rt[r], 0) - int(q[r])
                if t2 >= 0: delta[t2] = delta.get(t2, 0) + int(q[r])
            dd = d0 + sum(df(t, capt[t] + c) - df(t, capt[t]) for t, c in delta.items())
            if dd >= d0 and any(tabu.get(r, -1) > it for r, _ in mv):
                continue
            if dd < melhor_d:
                melhor, melhor_d = mv, dd
        if melhor is None:
            continue
        for r, t2 in melhor:
            if rt[r] >= 0: capt[rt[r]] -= int(q[r])
            if t2 >= 0: capt[t2] += int(q[r])
            rt[r] = t2; tabu[r] = it + 5 + rng.randrange(5)
    return None


def construtivo_uma_vez(inst, rng, alpha, ruido):
    tipos = [t for t in range(inst.T) if inst.demanda[t] > 0]
    chave = {t: inst.demanda[t] * (1 + ruido * rng.uniform(-1, 1)) for t in tipos}
    ordem = sorted(tipos, key=lambda t: -chave[t])             # tipos de maior demanda primeiro
    q = inst.sala_cap
    rtype = -np.ones(inst.R, dtype=int)
    livre = np.ones(inst.R, dtype=bool)
    cap_livre = int(q.sum())
    travou = False
    for pos, t in enumerate(ordem):
        dem_resto = sum(inst.demanda[u] for u in ordem[pos + 1:])
        falta = inst.demanda[t]
        while falta > 0:
            il = np.where(livre)[0]
            nova_falta = np.maximum(0, falta - q[il])
            ok = (cap_livre - q[il]) >= (nova_falta + dem_resto)
            if not ok.any():
                travou = True
                break
            cand = il[ok]
            util = np.minimum(q[cand], falta)
            score = inst.media[t][inst.sala_escola[cand]] / util   # custo por vaga util
            ord_sc = np.argsort(score, kind="stable")
            k = max(1, math.ceil(alpha * len(cand)))               # lista restrita de candidatas (GRASP)
            j = int(cand[ord_sc[rng.randrange(k)]])
            rtype[j] = t; livre[j] = False; cap_livre -= int(q[j]); falta -= int(q[j])
        if travou:
            break
    if travou:
        rtype = reparo_salas(inst, rtype, rng)
        if rtype is None:
            return None
    sol = {}
    for t in tipos:
        r = monta_tipo(inst, rtype, t)
        if r is None:
            return None
        sol[t] = r
    return rtype, sol, travou


def construtivo(inst, rng, alpha, ruido=0.0):
    for _ in range(6):
        r = construtivo_uma_vez(inst, rng, alpha, ruido)
        if r is not None:
            return (r[0], r[1]), int(r[2])
    raise RuntimeError("sem solucao viavel: nao foi possivel cobrir a demanda de todos os tipos com as salas")


def tabelas(inst, t, st):
    """Somas acumuladas p/ limites otimistas: ganho maximo ao abrir vagas em uma escola; perda minima ao tirar vagas."""
    d, a, caps = inst.Dt[t], st["a"], st["caps"]
    nt, m = d.shape
    cur = d[np.arange(nt), a]
    ativos = caps > 0
    ganho, perda = [], []
    for s in range(m):
        g = cur - d[:, s]
        g = np.sort(g[(a != s) & (g > 0)])[::-1]
        ganho.append(np.concatenate(([0.0], np.cumsum(g))))
        mem = a == s
        if mem.any():
            dm = np.where(ativos, d[mem], np.inf); dm[:, s] = np.inf
            p = np.sort(dm.min(axis=1) - d[mem, s])
            perda.append(np.concatenate(([0.0], np.cumsum(p))))
        else:
            perda.append(np.zeros(1))
    return ganho, perda, np.bincount(a, minlength=m)


def limite_tipo(tab, caps, dcap):
    ganho, perda, carga = tab
    b = 0.0
    for s, dc in dcap.items():
        if dc < 0:
            exc = max(0, -dc - int(caps[s] - carga[s]))
            if exc > 0: b += perda[s][min(exc, len(perda[s]) - 1)]
        elif dc > 0:
            b -= ganho[s][min(dc, len(ganho[s]) - 1)]
    return b


def reavalia(inst, t, st, novas):
    d = inst.Dt[t]
    a = st["a"].copy()
    m = d.shape[1]
    carga = np.bincount(a, minlength=m)
    ativos = novas > 0
    soltos = []
    for s in np.where(carga > novas)[0]:
        exc = int(carga[s] - novas[s])
        mem = np.where(a == s)[0]
        dm = np.where(ativos, d[mem], np.inf); dm[:, s] = np.inf
        pen = dm.min(axis=1) - d[mem, s]
        sai = mem[np.argsort(pen)[:exc]]
        a[sai] = -1; carga[s] -= exc; soltos += sai.tolist()
    for i in soltos:
        dd = np.where(carga < novas, d[i], np.inf)
        j = int(np.argmin(dd))
        if not np.isfinite(dd[j]):
            return None
        a[i] = j; carga[j] += 1
    custo = n1_lote(d, novas, a)
    return {"a": a, "caps": novas, "custo": custo}


def representantes(inst, rtype):
    grupos = {}
    for r in range(inst.R):
        grupos.setdefault((int(inst.sala_escola[r]), int(rtype[r])), []).append(r)
    reps = []
    for g in grupos.values():
        g.sort(key=lambda r: inst.sala_cap[r])
        reps += [g[0]] if len(g) == 1 or inst.sala_cap[g[0]] == inst.sala_cap[g[-1]] else [g[0], g[-1]]
    return reps


def efeito(inst, rtype, mov):
    ef = {}
    for r, t2 in mov:
        s, qr, a = int(inst.sala_escola[r]), int(inst.sala_cap[r]), int(rtype[r])
        if a >= 0: ef.setdefault(a, {}); ef[a][s] = ef[a].get(s, 0) - qr
        if t2 >= 0: ef.setdefault(t2, {}); ef[t2][s] = ef[t2].get(s, 0) + qr
    return ef


def busca_local(inst, rtype, sol, rng, max_aval=150):
    tipos = [t for t in range(inst.T) if inst.demanda[t] > 0]
    tabs = {}
    for t in tipos:
        sol[t]["custo"] = descida_local(inst.Dt[t], sol[t]["caps"], sol[t]["a"])
        tabs[t] = tabelas(inst, t, sol[t])
    n_mov = 0
    while True:
        reps = representantes(inst, rtype)
        ops = tipos + [-1]
        singles = [[(r, b)] for r in reps for b in ops if b != rtype[r]]
        swaps = None
        aceito = False
        for etapa in (0, 1):
            if etapa == 1:                                       # trocas entre duas salas so se nenhuma troca simples melhora
                swaps = [[(r1, int(rtype[r2])), (r2, int(rtype[r1]))]
                         for r1, r2 in itertools.combinations(reps, 2) if rtype[r1] != rtype[r2]]
            movs = singles if etapa == 0 else swaps
            ranq = []
            for mv in movs:
                ef = efeito(inst, rtype, mv)
                lim, viavel = 0.0, True
                for t, dc in ef.items():
                    if sol[t]["caps"].sum() + sum(dc.values()) < inst.demanda[t]:
                        viavel = False; break
                    lim += limite_tipo(tabs[t], sol[t]["caps"], dc)
                if viavel and lim < -EPS:
                    ranq.append((lim, mv, ef))
            ranq.sort(key=lambda x: x[0])
            for lim, mv, ef in ranq[:max_aval]:                  # avaliacao exata dos mais promissores
                novos, delta = {}, 0.0
                for t, dc in ef.items():
                    novas = sol[t]["caps"].copy()
                    for s, c in dc.items(): novas[s] += c
                    r = reavalia(inst, t, sol[t], novas)
                    if r is None:
                        novos = None; break
                    novos[t] = r; delta += r["custo"] - sol[t]["custo"]
                if novos is not None and delta < -EPS:
                    for r, t2 in mv: rtype[r] = t2
                    sol.update(novos)
                    for t in novos:
                        sol[t]["custo"] = descida_local(inst.Dt[t], sol[t]["caps"], sol[t]["a"])
                        tabs[t] = tabelas(inst, t, sol[t])
                    n_mov += 1; aceito = True
                    break
            if aceito:
                break
        if not aceito:
            break
    return rtype, sol, n_mov


def custo_total(sol):
    return sum(v["custo"] for v in sol.values())


def assign_global(inst, sol):
    assign = -np.ones(inst.n, dtype=int)
    for t, v in sol.items():
        assign[inst.idx[t]] = v["a"]
    return assign


def verifica(inst, rtype, sol):
    assign = assign_global(inst, sol)
    assert (assign >= 0).all(), "candidato sem escola"
    for t in range(inst.T):
        if inst.demanda[t] == 0: continue
        caps = caps_tipo(inst, rtype, t)
        carga = np.bincount(assign[inst.idx[t]], minlength=inst.m)
        assert (carga <= caps).all(), f"capacidade das salas do tipo {t} excedida"
    custo = float(inst.D[np.arange(inst.n), assign].sum())
    assert abs(custo - custo_total(sol)) < 1e-6, "custo inconsistente"
    return assign, custo


def distribui_salas(inst, rtype, assign):
    sala = -np.ones(inst.n, dtype=int)
    for s in range(inst.m):
        for t in range(inst.T):
            salas = [r for r in inst.escola_salas[s] if rtype[r] == t]
            cands = [i for i in inst.idx[t] if assign[i] == s]
            k = 0
            for r in salas:
                for _ in range(int(inst.sala_cap[r])):
                    if k < len(cands): sala[cands[k]] = r; k += 1
    return sala


def resolve(inst, seed, alpha, ruido):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    (rtype, sol), reparo = construtivo(inst, rng, alpha, ruido)
    verifica(inst, rtype, sol)
    ini = custo_total(sol)
    t1 = time.perf_counter()
    rtype, sol, n_mov = busca_local(inst, rtype, sol, rng)
    t2 = time.perf_counter()
    assign, fim = verifica(inst, rtype, sol)
    carga = np.bincount(assign, minlength=inst.m)
    return dict(seed=seed, inicial=ini, final=fim, melhoria_pct=100 * (ini - fim) / ini,
                t_constr=t1 - t0, t_busca=t2 - t1, escolas_fim=int((carga > 0).sum()),
                salas_abertas=int((rtype >= 0).sum()), mov_N3=n_mov, reinicios=reparo,
                gap_lb_pct=100 * (fim - inst.lb) / inst.lb, rtype=rtype.copy(), assign=assign, carga=carga)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instancia")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1, help="0 = construtivo guloso puro; >0 = GRASP")
    ap.add_argument("--ruido", type=float, default=0.0, help="perturbacao na ordem dos tipos")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv")
    ap.add_argument("--out", help="grava a melhor solucao (candidato, escola, sala)")
    a = ap.parse_args()

    inst = Instancia(a.instancia)
    print(f"{a.instancia}: {inst.n} candidatos, {inst.m} escolas, {inst.R} salas, cap. total {inst.cap.sum()} "
          f"({100*inst.n/inst.cap.sum():.1f}%), demanda por tipo {dict(zip(inst.nomes_tipos, inst.demanda))}")
    print(f"limitante inferior trivial (escola mais proxima, sem restricoes): {inst.lb:.2f} km")
    res = [resolve(inst, a.seed + r, a.alpha, a.ruido) for r in range(a.runs)]

    def est(k):
        v = [r[k] for r in res]
        return f"{statistics.mean(v):10.2f} +- {statistics.pstdev(v):7.2f}  [min {min(v):.2f}, max {max(v):.2f}]"
    print(f"\n{a.runs} execucoes (alpha={a.alpha}, ruido={a.ruido})")
    for k, nome in [("inicial", "solucao inicial (km)"), ("final", "apos busca local (km)"),
                    ("melhoria_pct", "melhoria (%)"), ("gap_lb_pct", "gap ao limitante trivial (%)"),
                    ("t_constr", "tempo construtivo (s)"), ("t_busca", "tempo busca local (s)"),
                    ("escolas_fim", "escolas usadas"), ("salas_abertas", "salas abertas"),
                    ("mov_N3", "movimentos N3 aceitos"), ("reinicios", "construtivo usou reparo (0/1)")]:
        print(f"  {nome:30s} {est(k)}")

    melhor = min(res, key=lambda r: r["final"])
    print(f"\nmelhor solucao: seed {melhor['seed']}, custo {melhor['final']:.2f} km")
    print("  escola  candidatos/cap   salas por tipo")
    for j in range(inst.m):
        por = {}
        for r in inst.escola_salas[j]:
            if melhor["rtype"][r] >= 0:
                por[inst.nomes_tipos[melhor["rtype"][r]]] = por.get(inst.nomes_tipos[melhor["rtype"][r]], 0) + 1
        print(f"  {inst.eids[j]:6s}  {melhor['carga'][j]:4d}/{inst.cap[j]:<4d}      {por}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            cols = ["seed", "inicial", "final", "melhoria_pct", "gap_lb_pct", "t_constr", "t_busca",
                    "escolas_fim", "salas_abertas", "mov_N3", "reinicios"]
            w.writerow(cols)
            for r in res:
                w.writerow([r[c] for c in cols])
    if a.out:
        sala = distribui_salas(inst, melhor["rtype"], melhor["assign"])
        with open(a.out, "w") as f:
            f.write(f"# custo_total_km: {melhor['final']:.4f}\n# id_candidato, id_escola, indice_da_sala_na_escola\n")
            for i in range(inst.n):
                s = melhor["assign"][i]
                f.write(f"{inst.cids[i]}, {inst.eids[s]}, {inst.escola_salas[s].index(sala[i]) + 1}\n")


if __name__ == "__main__":
    main()