#!/usr/bin/env python3

"""
Tools for generating task sets.
https://github.com/kevinzagalo/simso/blob/main/generator/task_generator.py
"""

import random
import typing

import numpy as np
import numpy.typing as npt

Utilizations = list[list[float]]
Periods = list[list[float]]


def gen_ripoll(
    nsets: int, compute: int, deadline: int, period: int, target_util: float
) -> list[list[tuple[int, int, int]]]:
    """
    Ripoll et al. tasksets generator.

    Args:
        - `nsets`: Number of tasksets to generate.
        - `compute`: Maximum computation time of a task.
        - `deadline`: Maximum slack time.
        - `period`: Maximum delay after the deadline.
        - `target_util`: Total utilization to reach.
    """
    sets = []
    for _ in range(nsets):
        task_set = []
        total_util = 0.0
        while total_util < target_util:
            c = random.randint(1, compute)
            d = c + random.randint(0, deadline)
            p = d + random.randint(0, period)
            task_set.append((c, d, p))
            total_util += float(c) / p
        sets.append(task_set)
    return sets


def gen_uunifastdiscard(nsets: int, u: float, n: int) -> Utilizations:
    """
    The UUniFast algorithm was proposed by Bini for generating task
    utilizations on uniprocessor architectures.

    The UUniFast-Discard algorithm extends it to multiprocessor by
    discarding task sets containing any utilization that exceeds 1.

    This algorithm is easy and widely used. However, it suffers from very
    long computation times when n is close to u. Stafford's algorithm is
    faster.

    Args:
        - `n`: The number of tasks in a task set.
        - `u`: Total utilization of the task set.
        - `nsets`: Number of sets to generate.

    Returns `nsets` of `n` task utilizations.
    """

    def UUniFastDiscard(n: int, u: float, nsets: int) -> Utilizations:
        sets: Utilizations = []
        while len(sets) < nsets:
            # Classic UUniFast algorithm:
            utilizations = []
            sumU = u
            for i in range(1, n):
                nextSumU = sumU * random.random() ** (1.0 / (n - i))
                utilizations.append(sumU - nextSumU)
                sumU = nextSumU
            utilizations.append(sumU)

            # If no task utilization exceeds 1:
            if all(ut <= 1 for ut in utilizations):
                sets.append(utilizations)

        return sets

    return UUniFastDiscard(n, u, nsets)


def gen_randfixedsum(nsets: int, u: float, n: int) -> Utilizations:
    """
    Stafford's RandFixedSum algorithm implementated in Python.

    Based on the Python implementation given by Paul Emberson, Roger Stafford,
    and Robert Davis. Available under the Simplified BSD License.

    Args:
        - `n`: The number of tasks in a task set.
        - `u`: Total utilization of the task set.
        - `nsets`: Number of sets to generate.
    """

    def StaffordRandFixedSum(n: int, u: float, nsets: int) -> Utilizations:
        """
        Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice,
            this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice,
            this list of conditions and the following disclaimer in the documentation
            and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
        OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
        OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
        EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
        INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
        OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
        LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
        OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
        ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        The views and conclusions contained in the software and documentation are
        those of the authors and should not be interpreted as representing official
        policies, either expressed or implied, of Paul Emberson, Roger Stafford or
        Robert Davis.

        Includes Python implementation of Roger Stafford's randfixedsum implementation
        http://www.mathworks.com/matlabcentral/fileexchange/9700
        Adapted specifically for the purpose of taskset generation with fixed
        total utilisation value

        Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
        any questions regarding this software.
        """
        if n < u:
            raise ValueError(f"Too many tasks for such an utilization (n:{n} ,u:{u})")

        # deal with n=1 case
        if n == 1:
            return [[u] for _ in range(nsets)]

        k = min(int(u), n - 1)
        su = u
        s1 = su - np.arange(k, k - n, -1.0)
        s2 = np.arange(k + n, k, -1.0) - su

        tiny = np.finfo(float).tiny
        huge = np.finfo(float).max

        w = np.zeros((n, n + 1))
        w[0, 1] = huge
        t = np.zeros((n - 1, n))

        for i in np.arange(2, n + 1):
            tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
            tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
            w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
            tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
            tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
            t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

        x = np.zeros((n, nsets))
        rt = np.random.uniform(size=(n - 1, nsets))  # rand simplex type
        rs = np.random.uniform(size=(n - 1, nsets))  # rand position in simplex
        s = np.repeat(su, nsets)
        j = np.repeat(k + 1, nsets)
        sm = np.repeat(0, nsets)
        pr = np.repeat(1, nsets)

        for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
            # decide which direction to move in this dimension (1 or 0):
            e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
            sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
            sm = sm + (1.0 - sx) * pr * s / (i + 1)
            pr = sx * pr
            x[(n - i) - 1, ...] = sm + pr * e
            s = s - e
            j = j - e  # change transition table column if required

        x[n - 1, ...] = sm + pr * s

        # iterated in fixed dimension order but needs to be randomised
        # permute x row order within each column
        for ii in range(nsets):
            x[..., ii] = x[np.random.permutation(n), ii]

        return x.T.tolist()

    return StaffordRandFixedSum(n, u, nsets)


def gen_kato_utilizations(nsets: int, umin: float, umax: float, target_util: float) -> Utilizations:
    """
    Kato et al. tasksets generator.

    A task set Γ is generated as follows. A new periodic task is appended
    to Γ as long as U(Γ) ≤ Utot is satisfied. For each task τi, its
    utilization Ui is computed based on a uniform distribution within the
    range of [Umin, Umax]. Only the utilization of the task generated at the
    very end is adjusted so that U(Γ) becomes equal to Utot (thus the Umin
    constraint might not be satisfied for this task).

    Args:
        - `nsets`: Number of tasksets to generate.
        - `umin`: Minimum task utilization.
        - `umax`: Maximum task utilization.
        - `target_util`:
    """
    sets = []
    for _ in range(nsets):
        task_set = []
        total_util = 0.0
        while total_util < target_util:
            u = random.uniform(umin, umax)
            if u + total_util > target_util:
                u = target_util - total_util
            total_util += u
            task_set.append(u)
        sets.append(task_set)
    return sets


def gen_periods_loguniform(n: int, nsets: int, min_: float, max_: float, round_to_int: float = False) -> Periods:
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    loguniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `min_`: Period min.
        - `max_`: Period max.
    """
    periods: npt.NDArray[np.floating] = np.exp(np.random.uniform(low=np.log(min_), high=np.log(max_), size=(nsets, n)))
    # Casting until I figure out how to properly propagate numpy shape information
    if round_to_int:
        return typing.cast(Periods, np.rint(periods).tolist())
    return typing.cast(Periods, periods.tolist())


def gen_periods_uniform(n: int, nsets: int, min_: float, max_: float, round_to_int: float = False) -> Periods:
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    uniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `min_`: Period min.
        - `max_`: Period max.
    """
    periods: npt.NDArray[np.floating] = np.random.uniform(low=min_, high=max_, size=(nsets, n))
    # Casting until I figure out how to properly propagate numpy shape information
    if round_to_int:
        return typing.cast(Periods, np.rint(periods).tolist())
    return typing.cast(Periods, periods.tolist())


def gen_periods_discrete(n: int, nsets: int, periods: Periods) -> Periods:
    """
    Generate a matrix of (nsets x n) random periods chosen randomly in the
    list of periods.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `periods`: A list of available periods.
    """
    # Casting until I figure out how to properly propagate numpy shape information
    gen_periods: npt.NDArray[np.floating] = np.random.choice(periods, size=(nsets, n))
    return typing.cast(Periods, gen_periods.tolist())


def gen_tasksets(utilizations: Utilizations, periods: Periods, trunc_digits: int) -> list[list[tuple[float, float]]]:
    """
    Take a list of task utilization sets and a list of task period sets and
    return a list of couples (c, p) sets. The computation times are truncated
    at a precision of 10^-10 to avoid floating point precision errors.

    Args:
        - `utilization`: The list of task utilization sets. For example::

            [[0.3, 0.4, 0.8], [0.1, 0.9, 0.5]]
        - `periods`: The list of task period sets. For examples::

            [[100, 50, 1000], [200, 500, 10]]

    Returns:
        For the above example, it returns::

            [[(30.0, 100), (20.0, 50), (800.0, 1000)],
             [(20.0, 200), (450.0, 500), (5.0, 10)]]
    """

    def trunc(x: float, p: int) -> float:
        r = int(x * 10**p) / float(10**p)
        return 1.0 if r == 0 else r

    return [
        [(trunc(ui * pi, trunc_digits), trunc(pi, trunc_digits)) for ui, pi in zip(us, ps)]
        for us, ps in zip(utilizations, periods)
    ]
