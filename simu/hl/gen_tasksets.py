#!/usr/bin/env python3

import pathlib
import sys


def put_file(path: str, content: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            cur = f.read()
    except FileNotFoundError:
        cur = None

    if content != cur:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def put_ts(outdir: str, name: str, simu_steps: int, ts: str) -> None:
    put_file(f"{outdir}/{name}.txt", ts)
    put_file(f"{outdir}/{name}.steps", f"{simu_steps}")


def gen_ex1(outdir: str) -> None:
    instances = (
        ((0, 0, 0), "synchronous"),
        ((2, 1, 0), "ripple"),
        ((1, 4, 0), "worst"),
    )
    simu_steps = 12

    for (o1, o2, o3), name in instances:
        ts = (
            "3 => number of tasks\n"
            f"{o1} 1  6  6 1 1 cyan  => Task1 (Oi Ci Di Ti SDi RDi)\n"
            f"{o2} 1 12 12 1 1 coral => Task2\n"
            f"{o3} 1 12 12 1 1 blue  => Task3\n"
        )

        put_ts(outdir, f"ex1_{name}", simu_steps, ts)


def gen_ex2(outdir: str) -> None:
    instances = (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
        (5, 1),
        (9, 9),
    )
    simu_steps = 12

    for i, j in instances:
        name = f"{i}{j}"
        t2 = 6
        t3 = 12

        if (i, j) == (9, 9):
            i = 2
            j = 1
            t2 = 7
            t3 = 13
            name = f"{i}{j}_spo"
        ts = (
            "3 => number of tasks\n"
            f"{i} 2 12 12 1 1 cyan  => Task1 (Oi Ci Di Ti SDi RDi)\n"
            f"{j} 1  6  {t2} 1 1 coral => Task2\n"
            f"0 1 12 {t3} 1 1 blue  => Task3\n"
        )

        put_ts(outdir, f"ex2_{name}", simu_steps, ts)


def gen_ex3(outdir: str) -> None:
    X = None
    instances = (
        (0, 0, X, 6),
        (0, 1, X, 6),
        (0, 2, X, 6),
        (0, 3, 6, 10),
        (0, 3, 7, 7),
        (0, 4, X, 4),
        (1, 0, X, 7),
        (1, 1, X, 7),
        (1, 2, X, 7),
        (1, 3, X, 7),
        (1, 4, 7, 11),
        (1, 4, 8, 8),
        (1, 5, X, 5),
        (2, 0, X, 6),
        (2, 1, X, 8),
        (2, 2, X, 2),
        (3, 0, X, 7),
        (3, 1, X, 7),
        (4, 0, X, 4),
        (4, 1, X, 8),
        (5, 1, X, 5),
    )
    for o1, o2, o1b, simu_steps in instances:
        if o1b is None:
            o1b_name = "x"
            t1 = 6
        else:
            o1b_name = f"{o1b}"
            t1 = o1b - o1
        t2 = 7
        if o1b is None and o1 + t1 == simu_steps:
            t1 += 1
        if o2 + t2 == simu_steps:
            t2 += 1

        ts = (
            "3 => number of tasks\n"
            f"{o1} 1  6  {t1} 1 1 cyan  => Task1 (Oi Ci Di Ti SDi RDi)\n"
            f"{o2} 1  7  {t2} 1 1 coral => Task2\n"
            "0 1 11 12 1 1 blue  => Task3\n"
        )

        name = f"{o1}{o2}_{o1b_name}"
        put_ts(outdir, f"ex3_{name}", simu_steps, ts)


def gen_ex4(outdir: str) -> None:
    instances = (
        ((2, 1, 0), 9, "worst_tau2"),
        ((5, 1, 0), 9, "worst_tau3"),
    )

    for (o1, o2, o3), simu_steps, name in instances:
        ts = (
            "3 => number of tasks\n"
            f"{o1} 1 99 99 0 0 cyan  => Task1 (Oi Ci Di Ti SDi RDi)\n"
            f"{o2} 1 99 99 1 1 coral => Task2\n"
            f"{o3} 1 99 99 1 2 blue  => Task3\n"
        )

        put_ts(outdir, f"ex4_{name}", simu_steps, ts)

def gen_ex5(outdir: str) -> None:
    instances = (
        ((10, 99), 5, "non_resumable_1"),
        ((1, 4), 9, "non_resumable_2"),
    )

    for (o1, t1), simu_steps, name in instances:
        ts = (
            "2 => number of tasks\n"
            f"{o1} 1 {t1} {t1} 0 0 cyan  => Task1 (Oi Ci Di Ti SDi RDi)\n"
            f"0 2 9 99 2 2 coral => Task2\n"
        )

        put_ts(outdir, f"ex5_{name}", simu_steps, ts)

def gen_ex6(outdir: str) -> None:
    simu_steps = 20
    ts = (
        "1 => number of tasks\n"
        f"0 2 6 6 2 2 coral  => Task1 (Oi Ci Di Ti SDi RDi)\n"
    )

    put_ts(outdir, f"ex6_base", simu_steps, ts)

def main() -> None:
    outdir = "ts"
    pathlib.Path(outdir).mkdir(exist_ok=True)

    gen_ex1(outdir)
    gen_ex2(outdir)
    gen_ex3(outdir)
    gen_ex4(outdir)
    gen_ex5(outdir)
    gen_ex6(outdir)


if __name__ == "__main__":
    main()
    sys.exit(0)
