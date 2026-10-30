#!/usr/bin/env python3
"""
Roda o main.py em todas as instancias de uma pasta (execucoes em paralelo, com retomada) e gera
o resumo para a planilha: melhor resultado, media de N execucoes e tempo medio.

Uso:
    python rodar_todas.py PASTA --runs 10 --alpha 0.3 --processos 4

Saidas (na pasta atual):
    resultados_execucoes.csv   uma linha por execucao (e o que permite retomar se o programa for interrompido)
    resultados_resumo.csv      uma linha por instancia: melhor, media, desvio, tempo medio, status
"""
import argparse, csv, glob, hashlib, multiprocessing as mp, os, statistics, sys, time
from functools import lru_cache

import main as T

CAMPOS_EXEC = ["instancia", "seed", "inicial", "final", "tempo_total_s", "escolas_fim", "mov_N3", "usou_reparo"]


@lru_cache(maxsize=2)
def carrega(caminho):
    return T.Instancia(caminho)


def tarefa(args):
    caminho, seed, alpha = args
    try:
        inst = carrega(caminho)
        t0 = time.perf_counter()
        r = T.resolve(inst, seed, alpha, 0.0)
        tempo = time.perf_counter() - t0          # construtivo + busca local (sem leitura do arquivo)
        return dict(instancia=caminho, seed=seed, inicial=r["inicial"], final=r["final"], tempo_total_s=tempo,
                    escolas_fim=r["escolas_fim"], mov_N3=r["mov_N3"], usou_reparo=r["reinicios"], erro="")
    except Exception as e:                       # noqa
        return dict(instancia=caminho, seed=seed, erro=f"{type(e).__name__}: {e}")


def descobre_instancias(pasta, alpha):
    """Le todas as instancias validas; devolve (validas, problemas)."""
    validas, problemas, vistos = [], {}, {}
    for f in sorted(glob.glob(os.path.join(pasta, "**", "*.txt"), recursive=True)):
        nome = os.path.relpath(f, pasta)
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h in vistos:
            problemas[nome] = f"duplicada de {vistos[h]} (nao executada)"
            continue
        vistos[h] = nome
        try:
            inst = T.Instancia(f)
        except BaseException as e:               # noqa
            problemas[nome] = f"arquivo fora do formato ({type(e).__name__})"
            continue
        try:                                      # teste rapido: o construtivo acha alguma solucao viavel?
            T.construtivo(inst, T.random.Random(1), alpha)
        except RuntimeError:
            problemas[nome] = "inviavel: as salas nao conseguem cobrir a demanda de todos os tipos"
            continue
        validas.append((f, nome, inst))
    return validas, problemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--execucoes_csv", default="resultados_execucoes.csv")
    ap.add_argument("--resumo_csv", default="resultados_resumo.csv")
    a = ap.parse_args()

    validas, problemas = descobre_instancias(a.pasta, a.alpha)
    for nome, motivo in problemas.items():
        print(f"[pulada] {nome}: {motivo}")

    feitas = set()                                # retomada
    if os.path.exists(a.execucoes_csv):
        for r in csv.DictReader(open(a.execucoes_csv, newline="")):
            feitas.add((r["instancia"], int(r["seed"])))
    tarefas = [(f, s, a.alpha) for f, nome, inst in validas for s in range(1, a.runs + 1) if (f, s) not in feitas]
    tamanho = {f: inst.n * inst.m for f, nome, inst in validas}
    tarefas.sort(key=lambda t: -tamanho[t[0]])    # maiores primeiro: melhor balanceamento entre processos
    print(f"{len(validas)} instancias validas, {len(tarefas)} execucoes a fazer "
          f"({len(feitas)} ja feitas), {a.processos} processo(s)")

    novo = not os.path.exists(a.execucoes_csv)
    f_out = open(a.execucoes_csv, "a", newline="")
    w = csv.DictWriter(f_out, fieldnames=CAMPOS_EXEC, extrasaction="ignore")
    if novo:
        w.writeheader()
    erros = {}
    t_ini = time.perf_counter()
    with mp.Pool(a.processos) as pool:
        for k, r in enumerate(pool.imap_unordered(tarefa, tarefas, chunksize=1), 1):
            if r["erro"]:
                erros[r["instancia"]] = r["erro"]
                print(f"  [{k}/{len(tarefas)}] ERRO {os.path.relpath(r['instancia'], a.pasta)} seed {r['seed']}: {r['erro'][:70]}")
            else:
                w.writerow(r); f_out.flush()
                print(f"  [{k}/{len(tarefas)}] {os.path.relpath(r['instancia'], a.pasta):32s} seed {r['seed']:2d} "
                      f"custo {r['final']:10.2f}  {r['tempo_total_s']:6.1f}s", flush=True)
    f_out.close()
    print(f"\ntempo de parede total: {time.perf_counter() - t_ini:.0f}s")

    # ---- resumo
    por = {}
    for r in csv.DictReader(open(a.execucoes_csv, newline="")):
        por.setdefault(r["instancia"], []).append(r)
    linhas = []
    for f, nome, inst in validas:
        rs = por.get(f, [])
        if len(rs) == 0:
            linhas.append(dict(instancia=nome, candidatos=inst.n, escolas=inst.m, execucoes=0,
                               status=erros.get(f, "sem resultados"))); continue
        fin = [float(r["final"]) for r in rs]
        tem = [float(r["tempo_total_s"]) for r in rs]
        linhas.append(dict(instancia=nome, candidatos=inst.n, escolas=inst.m, execucoes=len(rs),
                           melhor=round(min(fin), 2), media=round(statistics.mean(fin), 2),
                           desvio=round(statistics.pstdev(fin), 2), tempo_medio_s=round(statistics.mean(tem), 2),
                           status="ok" if len(rs) >= a.runs else "incompleta"))
    for nome, motivo in problemas.items():
        linhas.append(dict(instancia=nome, status=motivo))
    cols = ["instancia", "candidatos", "escolas", "execucoes", "melhor", "media", "desvio", "tempo_medio_s", "status"]
    with open(a.resumo_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(linhas)
    print(f"\n{'instancia':32s} {'melhor':>10s} {'media':>10s} {'tempo(s)':>9s}  status")
    for l in linhas:
        print(f"{l['instancia']:32s} {str(l.get('melhor','-')):>10s} {str(l.get('media','-')):>10s} "
              f"{str(l.get('tempo_medio_s','-')):>9s}  {l['status'][:60]}")
    print(f"\nresumo gravado em {a.resumo_csv}")


if __name__ == "__main__":
    mp.freeze_support()
    main()