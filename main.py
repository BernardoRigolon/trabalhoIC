#!/usr/bin/env python3
import argparse, csv, itertools, math, random, statistics, sys, time
import numpy as np

EPS = 1e-9
TIPOS = ["M1", "M2", "M3e", "M3s", "M3h", "M3d"]


class Instancia:
    def __init__(self, caminho):
        cids, ctipos, eids, caps, mat = [], [], [], [], []
        secao = None
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                if linha.startswith("NUM_CANDIDATOS"):
                    self.n = int(linha.split(":")[1]); continue
                if linha.startswith("NUM_ESCOLAS"):
                    self.m = int(linha.split(":")[1]); continue
                if linha.startswith("#"):
                    if linha.startswith("# CANDIDATOS"): secao = "C"
                    elif linha.startswith("# ESCOLAS"): secao = "E"
                    elif linha.startswith("# MATRIZ"): secao = "M"
                    continue
                p = [x.strip() for x in linha.split(",")]
                if secao == "C":
                    cids.append(p[0]); ctipos.append(p[2])
                elif secao == "E":
                    eids.append(p[0]); caps.append(sum(int(x) for x in p[5].split(";")))
                elif secao == "M":
                    mat.append([float(x) for x in p])
        assert len(cids) == self.n and len(eids) == self.m and len(mat) == self.n
        self.cids, self.eids = cids, eids
        self.D = np.array(mat)
        self.cap = np.array(caps, dtype=int)
        self.nomes_tipos = sorted(set(ctipos), key=lambda t: TIPOS.index(t) if t in TIPOS else 99)
        self.T = len(self.nomes_tipos)
        self.ctype = np.array([self.nomes_tipos.index(t) for t in ctipos])
        self.idx = [np.where(self.ctype == t)[0] for t in range(self.T)]
        self.demanda = [len(x) for x in self.idx]
        self.media = [self.D[x].mean(axis=0) for x in self.idx]   # dist. media da demanda do tipo a cada escola
        self.lb = float(self.D.min(axis=1).sum())                  # limitante trivial: cada um na escola mais proxima


def guloso_local(d, caps):
    """Aloca candidatos (linhas de d) a escolas (colunas) por arrependimento; None se nao couber."""
    nt, ns = d.shape
    if caps.sum() < nt:
        return None
    if ns == 1:
        return np.zeros(nt, dtype=int)
    ds = np.sort(d, axis=1)
    ordem = np.argsort(-(ds[:, 1] - ds[:, 0]), kind="stable")   # maior arrependimento primeiro
    rest = caps.copy()
    a = np.empty(nt, dtype=int)
    for i in ordem:
        for j in np.argsort(d[i]):
            if rest[j] > 0:
                a[i] = j; rest[j] -= 1
                break
    return a


def descida_local(d, caps, a):
    """N1 (realocar) e N2 (trocar) com melhor vizinho por candidato, ate otimo local. Modifica a."""
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


def monta_tipo(inst, stype, t, refinar):
    S = np.where(stype == t)[0]
    if S.size == 0 or inst.cap[S].sum() < inst.demanda[t]:
        return None
    d = inst.D[np.ix_(inst.idx[t], S)]
    a = guloso_local(d, inst.cap[S])
    if a is None:
        return None
    custo = descida_local(d, inst.cap[S], a) if refinar else float(d[np.arange(len(a)), a].sum())
    return {"S": S, "a": a, "custo": custo}


def construtivo_uma_vez(inst, rng, alpha, ruido):
    tipos = [t for t in range(inst.T) if inst.demanda[t] > 0]
    chave = {t: inst.demanda[t] * (1 + ruido * rng.uniform(-1, 1)) for t in tipos}
    ordem = sorted(tipos, key=lambda t: -chave[t])             # tipos de maior demanda primeiro
    livres = set(range(inst.m))
    stype = -np.ones(inst.m, dtype=int)
    for pos, t in enumerate(ordem):
        dem_resto = sum(inst.demanda[u] for u in ordem[pos + 1:])
        tipos_resto = len(ordem) - pos - 1
        falta = inst.demanda[t]
        while falta > 0:
            cap_livre = sum(inst.cap[k] for k in livres)
            cands = []
            for j in livres:
                nova_falta = max(0, falta - inst.cap[j])
                precisa_escolas = tipos_resto + (1 if nova_falta > 0 else 0)
                if len(livres) - 1 < precisa_escolas:           # sobram escolas para os tipos restantes?
                    continue
                if cap_livre - inst.cap[j] < nova_falta + dem_resto:   # sobra capacidade?
                    continue
                util = min(inst.cap[j], falta)
                cands.append((inst.media[t][j] / util, j))      # custo por vaga util
            if not cands:
                return None
            cands.sort()
            k = max(1, math.ceil(alpha * len(cands)))           # lista restrita de candidatas (GRASP)
            j = cands[rng.randrange(k)][1]
            stype[j] = t; livres.discard(j); falta -= inst.cap[j]
    sol = {}
    for t in tipos:
        r = monta_tipo(inst, stype, t, refinar=False)
        if r is None:
            return None
        sol[t] = r
    return stype, sol


def construtivo(inst, rng, alpha, ruido=0.0):
    for tentativa in range(1000):
        if tentativa > 0:                                       # reinicio com mais aleatoriedade
            alpha_, ruido_ = max(alpha, 0.3), max(ruido, 0.5)
        else:
            alpha_, ruido_ = alpha, ruido
        r = construtivo_uma_vez(inst, rng, alpha_, ruido_)
        if r is not None:
            return r, tentativa
    raise RuntimeError("construtivo nao achou solucao viavel")


def gera_movimentos(inst, stype, tipos):
    mov = []
    for j in range(inst.m):
        for t2 in tipos + [-1]:
            if t2 != stype[j]:
                mov.append({j: t2})
    abertas = [j for j in range(inst.m) if stype[j] >= 0]
    fechadas = [j for j in range(inst.m) if stype[j] < 0]
    for a, b in itertools.combinations(abertas, 2):
        if stype[a] != stype[b]:
            mov.append({a: stype[b], b: stype[a]})                # trocar tipos entre duas escolas
    for a in abertas:
        for c in fechadas:
            mov.append({a: -1, c: stype[a]})
    return mov


def avalia(inst, stype, sol, mov):
    novo = stype.copy()
    afet = set()
    for j, t in mov.items():
        if stype[j] >= 0: afet.add(int(stype[j]))
        if t >= 0: afet.add(int(t))
        novo[j] = t
    novos, delta = {}, 0.0
    for t in afet:
        r = monta_tipo(inst, novo, t, refinar=True)
        if r is None:
            return None
        novos[t] = r
        delta += r["custo"] - sol[t]["custo"]
    return delta, novo, novos


def busca_local(inst, stype, sol, rng):
    for t in sol:
        S, a = sol[t]["S"], sol[t]["a"]
        d = inst.D[np.ix_(inst.idx[t], S)]
        sol[t]["custo"] = descida_local(d, inst.cap[S], a)
    tipos = [t for t in range(inst.T) if inst.demanda[t] > 0]
    n_mov = 0
    while True:
        movs = gera_movimentos(inst, stype, tipos)
        rng.shuffle(movs)
        aceito = False
        for mov in movs:
            r = avalia(inst, stype, sol, mov)
            if r is not None and r[0] < -EPS:
                _, stype, novos = r
                sol.update(novos)
                n_mov += 1; aceito = True
                break
        if not aceito:
            break
    return stype, sol, n_mov


def custo_total(sol):
    return sum(v["custo"] for v in sol.values())


def assign_global(inst, sol):
    assign = -np.ones(inst.n, dtype=int)
    for t, v in sol.items():
        assign[inst.idx[t]] = v["S"][v["a"]]
    return assign


def verifica(inst, stype, sol):
    assign = assign_global(inst, sol)
    assert (assign >= 0).all(), "candidato sem escola"
    assert all(stype[assign[i]] == inst.ctype[i] for i in range(inst.n)), "tipo incompativel"
    carga = np.bincount(assign, minlength=inst.m)
    assert (carga <= inst.cap).all(), "capacidade excedida"
    custo = float(inst.D[np.arange(inst.n), assign].sum())
    assert abs(custo - custo_total(sol)) < 1e-6, "custo inconsistente"
    return assign, carga, custo


def resolve(inst, seed, alpha, ruido):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    (stype, sol), reinicios = construtivo(inst, rng, alpha, ruido)
    for t, v in sol.items():
        d = inst.D[np.ix_(inst.idx[t], v["S"])]
        v["custo"] = float(d[np.arange(len(v["a"])), v["a"]].sum())
    verifica(inst, stype, sol)
    ini = custo_total(sol)
    n_ini = int((stype >= 0).sum())
    t1 = time.perf_counter()
    stype, sol, n_mov = busca_local(inst, stype, sol, rng)
    t2 = time.perf_counter()
    assign, carga, fim = verifica(inst, stype, sol)
    return dict(seed=seed, inicial=ini, final=fim, melhoria_pct=100 * (ini - fim) / ini,
                t_constr=t1 - t0, t_busca=t2 - t1, escolas_ini=n_ini, escolas_fim=int((stype >= 0).sum()),
                mov_N3=n_mov, reinicios=reinicios, gap_lb_pct=100 * (fim - inst.lb) / inst.lb,
                stype=stype.copy(), assign=assign, carga=carga)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instancia")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.3, help="0 = construtivo guloso puro; >0 = GRASP")
    ap.add_argument("--ruido", type=float, default=0.0, help="perturbacao na ordem dos tipos")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv")
    ap.add_argument("--out", help="grava a melhor solucao")
    a = ap.parse_args()

    inst = Instancia(a.instancia)
    print(f"{a.instancia}: {inst.n} candidatos, {inst.m} escolas, cap. total {inst.cap.sum()} "
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
                    ("escolas_ini", "escolas abertas (inicial)"), ("escolas_fim", "escolas abertas (final)"),
                    ("mov_N3", "movimentos N3 aceitos"), ("reinicios", "reinicios do construtivo")]:
        print(f"  {nome:30s} {est(k)}")

    melhor = min(res, key=lambda r: r["final"])
    print(f"\nmelhor solucao: seed {melhor['seed']}, custo {melhor['final']:.2f} km")
    print("  escola  tipo  carga/cap")
    for j in range(inst.m):
        tp = inst.nomes_tipos[melhor["stype"][j]] if melhor["stype"][j] >= 0 else "-"
        print(f"  {inst.eids[j]:6s}  {tp:4s}  {melhor['carga'][j]:3d}/{inst.cap[j]}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            cols = ["seed", "inicial", "final", "melhoria_pct", "gap_lb_pct", "t_constr", "t_busca",
                    "escolas_ini", "escolas_fim", "mov_N3", "reinicios"]
            w.writerow(cols)
            for r in res:
                w.writerow([r[c] for c in cols])
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"# custo_total_km: {melhor['final']:.4f}\n# id_candidato, id_escola\n")
            for i in range(inst.n):
                f.write(f"{inst.cids[i]}, {inst.eids[melhor['assign'][i]]}\n")


if __name__ == "__main__":
    main()
