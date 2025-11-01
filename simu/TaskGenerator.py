#!/usr/bin/env python3

import enum
import itertools
import math
import typing

import task_generator


@enum.verify(enum.UNIQUE)
class UtilAlgorithm(enum.Enum):
    RandFixedSum = 1
    UUniFast_Discard = 2
    Kato = 3
    Bruteforce = 4


TaskSet = list[typing.Tuple[int, int]]


class Bruteforce:
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self.nb_tasks = 0
        self.min_delay = 0
        self.the_generator = self.cost_generator()

    # Generates all possibilites by increasing sum of Ti (cost)
    # 'min_delay' is a lower bound on loading delays. Avoid generating tasksets with too low 'compute'.
    def cost_generator(self) -> typing.Generator[TaskSet, None, None]:
        # Since a period of 0 or 1 is useless, start at 2 */
        MIN_PERIOD = 2

        for cost in itertools.count(0):
            if self.verbose > 0:
                print(f"  CG: {cost=}")
            p = [MIN_PERIOD] * self.nb_tasks  # List of periods
            p[0] += cost
            index = 0 if cost > 0 else -1
            # Generate all period lists (in 'p') such that sum(p) == cost
            while True:
                # We have periods, delegate compute generation.
                yield from self.subgen_compute(p)

                if index < 0:
                    break

                if index + 1 < self.nb_tasks:  # ripple to next index
                    p[index] -= 1
                    index += 1
                    p[index] += 1
                    if self.verbose > 0:
                        print(f"    CG: ripple {index=} {p=}")
                else:  # move back p[-1] and ripple
                    new_index = index - 1
                    while new_index >= 0 and p[new_index] == MIN_PERIOD:
                        new_index -= 1
                    if new_index < 0:
                        break
                    rightmost = p[-1] - MIN_PERIOD
                    p[-1] = MIN_PERIOD
                    p[new_index] -= 1
                    new_index += 1
                    p[new_index] += 1 + rightmost
                    index = new_index
                    if self.verbose > 0:
                        print(f"    CG: backt {index=} {p=}")

            assert p[-1] == MIN_PERIOD + cost

    # Generate all computes for given periods
    def subgen_compute(self, p: list[int]) -> typing.Generator[TaskSet, None, None]:
        S2US = 1_000_000  # TODO: API mandates we return periods in µs. To be removed someday.

        min_compute = self.min_delay + 1
        p_gcd = math.gcd(*p)
        utilization_one = math.lcm(*p)  # Avoid floats for utilization
        c = [min_compute] * len(p)  # initialize compute to the minimum
        utilization = sum((ci * utilization_one) // pi for (ci, pi) in zip(c, p))  # sum c_i/p_i
        if utilization > utilization_one:
            return  # Bail out if utilization > 1
        if self.verbose > 0:
            print(f"      SG: {p=}")
        while True:
            gcd = math.gcd(p_gcd, math.gcd(*c))
            if self.verbose > 0:
                print(f"        SG: {c=} {p=} {gcd=}")
            if gcd > 1:
                return

            yield [(ci * S2US, pi * S2US) for (ci, pi) in zip(c, p)]

            index = 0
            dirty = True
            while dirty and index < len(c):
                dirty = False
                utilization += utilization_one // p[index]
                c[index] += 1
                if c[index] >= p[index] or utilization > utilization_one:
                    utilization -= (c[index] - min_compute) * utilization_one // p[index]
                    c[index] = min_compute
                    index += 1
                    dirty = True
            if index >= len(c):
                if self.verbose > 0:
                    print("      SG: --> OUT")
                break

    def generate(self, nb_tasks: int, min_delay: int) -> TaskSet:
        if self.nb_tasks != nb_tasks or self.min_delay != self.min_delay:
            self.nb_tasks = nb_tasks
            self.min_delay = min_delay
            self.the_generator = self.cost_generator()
        return next(self.the_generator)


class TaskGenerator:
    def __init__(self) -> None:
        # RandFixedSum, UUniFast-Discard, Kato's method
        self.comboGenerator = UtilAlgorithm.RandFixedSum

        # Utilization
        self.utilization = 0.8  # [0, 1] on monocore

        # Number of periodic tasks:
        self.nb_tasks = 4  # [1, 999]

        # Min / Max utilizations
        self.interval_utilization_min = 0.01  # [.0, 1.], for Kato
        self.interval_utilization_max = 0.99  # [.0, 1.], for Kato

        # Periods, Uniform or Log uniform
        self.period_lunif = True  # true/false
        self.period_interval_min = 1.0
        self.period_interval_max = 1000.0
        self.period_interval_round_to_integer = True

        # For bruteforce generation algorithm
        self.bruteforce = Bruteforce(verbose=0)
        self.c_tlb_s = 0

        # Number of precision digits for tasks values
        self.trunc_digits = 6  # Range: [0;6]

    def generate(self) -> TaskSet:
        n = self.nb_tasks
        if n <= 0:
            raise ValueError("Generation failed, please check the utilization and the number of tasks.")

        if self.comboGenerator == UtilAlgorithm.Bruteforce:
            return self.bruteforce.generate(self.nb_tasks, self.c_tlb_s)

        match self.comboGenerator:
            case UtilAlgorithm.RandFixedSum:
                u = task_generator.gen_randfixedsum(1, self.utilization, n)
            case UtilAlgorithm.UUniFast_Discard:
                u = task_generator.gen_uunifastdiscard(1, self.utilization, n)
            case UtilAlgorithm.Kato:
                while True:
                    u = task_generator.gen_kato_utilizations(
                        1,
                        self.interval_utilization_min,
                        self.interval_utilization_max,
                        self.utilization,
                    )
                    if n == len(u[0]):
                        break
            case _:
                raise ValueError("Invalid comboGenerator")

        period_min = self.period_interval_min
        period_max = self.period_interval_max
        period_rti = self.period_interval_round_to_integer
        if self.period_lunif:
            p = task_generator.gen_periods_loguniform(n, 1, period_min, period_max, period_rti)
        else:
            p = task_generator.gen_periods_uniform(n, 1, period_min, period_max, period_rti)

        if not u or not p:
            raise ValueError("Generation failed", "Please check the periods.")
        taskset_float = task_generator.gen_tasksets(u, p, self.trunc_digits)[0]
        # Must convert to µs instead of floats
        S2US = 1000_000
        return [(int(task_c * S2US), int(task_p * S2US)) for task_c, task_p in taskset_float]


if __name__ == "__main__":
    tg = TaskGenerator()
    print(tg.generate())
