#!/usr/bin/env python3

import dataclasses
import enum
import functools
import itertools
import math
import os
import queue
import subprocess
import sys
import tempfile
import typing

import colorama
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

import TaskGenerator


class td:
    S2US = 1000_000

    def __init__(self, s: int = 0, us: int = 0):
        self.val = s * self.S2US + us

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, td):
            return NotImplemented
        return self.val == other.val

    def __lt__(self, other: "td") -> bool:
        return self.val < other.val

    def __le__(self, other: "td") -> bool:
        return self.val <= other.val

    def __add__(self, other: "td") -> "td":
        return td(us=self.val + other.val)

    def __sub__(self, other: "td") -> "td":
        return td(us=self.val - other.val)

    @staticmethod
    def multconst(first: int, other: "td") -> "td":
        return td(us=first * other.val)

    @staticmethod
    def ceildiv(first: "td", other: "td") -> int:
        val = -(first.val // -other.val)
        return val

    def __str__(self) -> str:
        return f"{self.val / self.S2US:9.6f}"

    def __repr__(self) -> str:
        return self.__str__()

    def get_s(self) -> int:
        return self.val // self.S2US

    def get_us(self) -> int:
        return self.val % self.S2US

    def to_s(self) -> int:
        assert self.val % self.S2US == 0
        return self.val // self.S2US

    def to_us(self) -> int:
        return self.val


C_TLB = td(s=1)
C_SD = C_TLB
C_RD = C_TLB


@dataclasses.dataclass
class ITask:
    compute: td
    period: td
    tlb_response_time: td

    def __str__(self) -> str:
        return (
            f"ITask(C:{colorama.Fore.CYAN}{self.compute}{colorama.Style.RESET_ALL}"
            + f",P:{colorama.Style.BRIGHT}{colorama.Fore.CYAN}{self.period}{colorama.Style.RESET_ALL})"
        )

    def __repr__(self) -> str:
        return self.__str__()


TaskSet = list[ITask]


class BoundingTheNumberOfPreemptions:
    # Regular response-time without delays (but we reintegrate the starting delay)
    def compute_rn(self, tasks: TaskSet, Dn: td) -> typing.Tuple[td, int]:
        rn = td(0)
        dirty = True
        loops = 0
        while dirty and rn <= Dn:
            dirty = False
            loops += 1
            new_rn = C_SD + tasks[-1].compute
            for j in range(len(tasks) - 1):
                new_rn += td.multconst(td.ceildiv(rn, tasks[j].period), C_SD + tasks[j].compute)
            if rn != new_rn:
                dirty = True
            rn = new_rn
        return rn, loops

    def compute_rntlb(self, tasks: TaskSet, Dn: td) -> typing.Tuple[list[td], int]:
        base_time = C_SD + tasks[-1].compute
        rntlb = [base_time]
        loops = 0
        while rntlb[-1] <= Dn:
            loops += 1
            s = td(0)
            for j in range(len(tasks) - 1):
                nb_prempts = td.ceildiv((rntlb[-1] - C_SD), tasks[j].period)
                s += td.multconst(nb_prempts, (C_SD + tasks[j].compute + C_RD))
            new_val = base_time + s
            if new_val == rntlb[-1]:
                break
            rntlb.append(new_val)
        return rntlb, loops


BTOP = BoundingTheNumberOfPreemptions()


class Instance:
    def __init__(self, tg: TaskGenerator.TaskGenerator, verbose: int = 0):
        self.tg = tg
        self.generate(verbose)

    def generate(self, verbose: int = 0) -> None:
        while True:
            # TODO: handle this directly in the task generator
            tasks_int = self.tg.generate()  # First task in list: highest priority
            good = all(c > C_SD.to_us() for (c, _) in tasks_int)
            if good:
                break

        tasks = [ITask(td(us=c) - C_SD, td(us=t), td(0)) for (c, t) in tasks_int]
        if verbose > 0:
            print(f"Tasks: {tasks}")

        self.tasks = tasks
        self.regular_schedulable = True
        self.tlb_schedulable = True

        for i in reversed(range(len(tasks))):
            tasks_slice = tasks[: i + 1]
            Di = tasks[i].period  # Use deadline == period

            Ri, ri_loops = BTOP.compute_rn(tasks_slice, Di)

            if Ri > Di:
                if verbose > 0:
                    print("System not even schedulable")
                self.regular_schedulable = False

            Ritlb, Ritlb_loops = BTOP.compute_rntlb(tasks_slice, Di)
            tasks[i].tlb_response_time += Ritlb[-1]
            if verbose > 0:
                print(f"[R] task {i}: Ri:{Ri} Ritlb:{Ritlb} Di:{Di} (ri_loops:{ri_loops} rntlb_loops:{Ritlb_loops})")
            if tasks[i].tlb_response_time > Di and self.tlb_schedulable:
                self.tlb_schedulable = False

            if i == len(tasks) - 1:
                self.Rn = Ri
                self.Rntlb = Ritlb

            if not self.regular_schedulable:
                assert not self.tlb_schedulable
                break

    def __str__(self) -> str:
        return (
            ""
            + (f"{colorama.Fore.GREEN}reg_sched" if self.regular_schedulable else f"{colorama.Fore.RED}reg_nosch")
            + " "
            + (f"{colorama.Fore.GREEN}tlb_sched" if self.tlb_schedulable else f"{colorama.Fore.RED}tlb_nosch")
            + f"{colorama.Style.RESET_ALL}"
            + f", Rn: {self.Rn}, Rntlb: {self.Rntlb[-1]}, tasks: {self.tasks}"
        )

    def draw(self, filename: str, trunc_digits: int, steps: int = 100, verbose: int = 0) -> None:
        task_colors = ["#56B4E9", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#F0E442", "#D55E00"]
        dm_taskset = f"{len(self.tasks)}\n"
        assert 0 <= trunc_digits <= 6
        div_factor = 10 ** (6 - trunc_digits)
        for i, task in enumerate(self.tasks):
            # (Oi Ci Di Ti Li [color1] [color2] [color3])
            Oi = 0
            Ci = int(task.compute.val // div_factor)
            Di = int(task.period.val // div_factor)
            Ti = Di
            Li = C_TLB.val // div_factor
            color1 = task_colors[i % len(task_colors)]
            dm_taskset += f"{Oi} {Ci} {Di} {Ti} {Li} {color1}\n"

        with tempfile.NamedTemporaryFile(delete_on_close=False) as temp:
            temp.write(dm_taskset.encode("utf-8"))
            temp.close()
            DRAW_DIR = "img"
            os.makedirs(DRAW_DIR, exist_ok=True)
            call = [
                "../draw-schedule/bin/ds_l",
                "-s",
                "FP",  # fixed priority scheduling policy
                "-m",
                "NRA",  # non-resumable delay
                "-v",
                "LT",  # starting + resuming delays
                "-d",
                f"{steps}",  # scheduling steps
                "-f",  # highlight missed deadlines
                "-c",  # highlight first cycle
                "-q",  # disable GUI
                "-o",
                f"{DRAW_DIR}/{filename}.svg",
                temp.name,
            ]
            if verbose > 0:
                print(" ".join(call))
            subprocess.check_call(
                call,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


class DeadlineMissException(Exception):
    pass


class SimulationPeriodic:
    @dataclasses.dataclass
    class STask:
        id: int
        compute: td
        period: td
        offset: td
        priority: int
        tlb_response_time: td
        max_seen_response_time: td = dataclasses.field(default_factory=lambda: td(0))
        compute_left: td = dataclasses.field(default_factory=lambda: td(0))
        job_start_time: td = dataclasses.field(default_factory=lambda: td(0))

    @enum.verify(enum.UNIQUE)
    class TlbCostMode(enum.Enum):
        No = 1  # No TLB cost
        Add = 2  # Add TLB cost to the compute cost
        Deduce = 3  # Add TLB cost, but deduce it from the compute cost

    def __init__(self, verbose: int = 0) -> None:
        self.task_running: typing.Optional[int] = None
        self.s_tasks: list[SimulationPeriodic.STask] = []
        self.next_activations_queue: queue.PriorityQueue[tuple[td, int]] = queue.PriorityQueue()
        self.needs_running_queue: queue.PriorityQueue[tuple[int, int]] = queue.PriorityQueue()
        self.verbose = verbose

    def set_needs_running(self, tid: int, rerun: bool = False, activation_time: typing.Optional[td] = None) -> None:
        if not all(item[1] is not tid for item in self.needs_running_queue.queue):
            raise DeadlineMissException("Previous task job (in running queue) did not finish on time...")
        if not rerun:
            if tid == self.task_running:
                raise DeadlineMissException("Previous task job (which is currently running) did not finish on time...")
        self.needs_running_queue.put((self.s_tasks[tid].priority, tid))
        if not rerun:
            assert activation_time is not None
            self.s_tasks[tid].job_start_time = activation_time
            self.s_tasks[tid].compute_left = self.s_tasks[tid].compute

    def running_task_run_left(self, with_tlbcost: bool) -> td:
        assert self.task_running is not None
        return self.s_tasks[self.task_running].compute_left + (C_TLB if with_tlbcost else td(0))

    def task_run_slot(self, old_time: td, new_time: typing.Optional[td] = None, with_tlbcost: bool = False) -> td:
        # TODO: decide how to handle end of job i which is immediatly followed by start of job i+1 of same task
        # Run then deschedule old task
        if self.task_running is None:
            assert new_time is not None
            return new_time
        if new_time is None:
            ran_for = self.running_task_run_left(with_tlbcost)
        else:
            ran_for = new_time - old_time
        if with_tlbcost:
            computed_for = max(td(0), ran_for - C_TLB)
        else:
            computed_for = ran_for
        self.s_tasks[self.task_running].compute_left -= computed_for
        if self.s_tasks[self.task_running].compute_left > td(0):
            if self.verbose >= 2:
                print(
                    f"Task {self.task_running} ran for {ran_for}, "
                    + f"compute_left:{self.s_tasks[self.task_running].compute_left}"
                )
            self.set_needs_running(self.task_running, rerun=True)
        else:
            response_time = old_time + ran_for - self.s_tasks[self.task_running].job_start_time
            self.s_tasks[self.task_running].max_seen_response_time = max(
                self.s_tasks[self.task_running].max_seen_response_time, response_time
            )
            assert response_time <= self.s_tasks[self.task_running].tlb_response_time, (
                "Have response time worse than computed worse: "
                + f"computed:{self.s_tasks[self.task_running].tlb_response_time } "
                + f"now:{response_time} start:{self.s_tasks[self.task_running].job_start_time}"
            )
            if self.verbose >= 2:
                print(f"Task {self.task_running} job is done after running for {ran_for}")
        self.task_running = None

        if new_time is None:
            return old_time + ran_for
        return new_time

    def schedule(self) -> None:
        assert self.task_running is None
        # Pick and start new task
        if self.needs_running_queue.empty():
            return
        _, self.task_running = self.needs_running_queue.get()
        if self.verbose >= 2:
            print(f"Now running task {self.task_running}, compute_left:{self.s_tasks[self.task_running].compute_left}")
        assert self.s_tasks[self.task_running].compute_left > td(0)

    def simulate(self, instance: Instance, tlbcost_mode: TlbCostMode) -> None:
        self.s_tasks = []
        iter_tid = 0
        for itask in instance.tasks:
            compute = itask.compute
            if tlbcost_mode == self.TlbCostMode.Deduce:
                compute -= C_SD
                assert compute > td(0)
            self.s_tasks.append(
                SimulationPeriodic.STask(
                    id=iter_tid,
                    compute=compute,
                    offset=td(0),
                    period=itask.period,
                    priority=iter_tid,
                    tlb_response_time=itask.tlb_response_time,
                )
            )
            iter_tid += 1
        if self.verbose >= 3:
            print(self.s_tasks)

        with_tlbcost = tlbcost_mode != self.TlbCostMode.No

        self.next_activations_queue = queue.PriorityQueue()
        for task in self.s_tasks:
            self.next_activations_queue.put((task.offset, task.id))

        self.needs_running_queue = queue.PriorityQueue()

        max_offset = max(*[task.offset for task in self.s_tasks])
        offset_sn = functools.reduce(  # goossens1997 Theorem 7
            lambda s, task: task.offset
            + td.multconst(td.ceildiv(max(td(0), s - task.offset), task.period), task.period),
            self.s_tasks,
            td(0),
        )
        hyperperiod = td(us=math.lcm(*[task.period.to_us() for task in self.s_tasks]))
        end_simu_time = offset_sn + hyperperiod
        if self.verbose >= 1:
            print(f"Running simulation up to {end_simu_time}  ({hyperperiod=} {offset_sn= } {max_offset=})")

        current_time = td(0)
        self.task_running = None
        while current_time <= end_simu_time:  # Need go to end_simu_time+epsilon to trigger the last activations
            next_activation_time, next_activation_tid = self.next_activations_queue.get()
            if self.verbose >= 2:
                print(
                    f"[{current_time.get_s():4}.{current_time.get_us():>6}]"
                    + f" Next activation: TID:{next_activation_tid} activation_time:{next_activation_time}",
                    end="",
                )
            delta = next_activation_time - current_time
            if self.task_running is not None and self.running_task_run_left(with_tlbcost) <= delta:
                # Aka running task ends before next activation
                if self.verbose >= 2:
                    print(" --> RTEBNA")
                self.next_activations_queue.put((next_activation_time, next_activation_tid))
                current_time = self.task_run_slot(current_time, with_tlbcost=with_tlbcost)
                self.schedule()
            else:
                self.next_activations_queue.put(
                    (next_activation_time + self.s_tasks[next_activation_tid].period, next_activation_tid)
                )
                if (
                    self.task_running is None
                    or self.s_tasks[next_activation_tid].priority < self.s_tasks[self.task_running].priority
                ):
                    if self.verbose >= 2:
                        print(" --> SCH")
                    current_time = self.task_run_slot(current_time, next_activation_time, with_tlbcost=with_tlbcost)
                    self.set_needs_running(next_activation_tid, activation_time=next_activation_time)
                    self.schedule()
                else:
                    if self.verbose >= 2:
                        print(" --> lessprio")
                    self.set_needs_running(next_activation_tid, activation_time=next_activation_time)


class SimulationSporadic:
    @dataclasses.dataclass(frozen=True)
    class State:
        """
        State of system:
        - forall task i:
          - how long ago the job started (range [0; Ti]),
          - if starting delay has been performed,
          - compute left for job (range [0; Ci]).
        - how much LD is left for the running task
        """

        jobs_started: tuple[int, ...]
        start_delays_done: tuple[bool, ...]
        computes_left: tuple[int, ...]
        ld_left: int

        def __str__(self) -> str:
            sdd = [
                f"{colorama.Style.BRIGHT}{'T' if sdd else 'F'}{colorama.Style.RESET_ALL}"
                for sdd in self.start_delays_done
            ]
            js = [f"{j:2}" for j in self.jobs_started]
            return (
                f"State(JS:{colorama.Fore.CYAN}({','.join(js)}){colorama.Style.RESET_ALL}"
                + f",SDD({','.join(sdd)}),"
                + f"CL:{colorama.Style.BRIGHT}{colorama.Fore.CYAN}{self.computes_left}{colorama.Style.RESET_ALL}),"
                + f"ld_left:{self.ld_left}"
            )

    def _push_state(self, state: State) -> None:
        seen = True
        if state not in self.seen_states:
            seen = False
            self.seen_states.add(state)
            self.new_states.add(state)
        if self.verbose > 0:
            print(
                f"    push_state: {'SEEN' if seen else 'NEW '} {state}"
                + f"  || seen:{len(self.seen_states)} new:{len(self.new_states)}"
            )

    def _get_running_task_index(self, computes_left: tuple[int, ...]) -> int:
        try:
            running_task_index = next(i for i, compute_left in enumerate(computes_left) if compute_left > 0)
        except StopIteration:
            running_task_index = len(self.tasks)
        return running_task_index

    def run(self) -> None:
        if self.verbose > 0:
            print("  SS: run simulation")
        while len(self.new_states) > 0:
            state = self.new_states.pop()
            if self.verbose > 0:
                print(f"  SS: POP {state}")

            running_task_index = self._get_running_task_index(state.computes_left)

            # Try to start any job which could start
            can_start_job = {
                i for i, job_started in enumerate(state.jobs_started) if job_started >= self.tasks[i].period.get_s()
            }
            for task_index in can_start_job:
                # Cannot start a new job if the previous one did not finish
                assert state.computes_left[task_index] == 0

                self._push_state(
                    self.State(
                        # Update job start
                        jobs_started=tuple(0 if i == task_index else js for i, js in enumerate(state.jobs_started)),
                        # Reset to needing start delay
                        start_delays_done=tuple(
                            False if i == task_index else sdd for i, sdd in enumerate(state.start_delays_done)
                        ),
                        # Reset compute for the new job
                        computes_left=tuple(
                            self.tasks[i].compute.get_s() if i == task_index else cl
                            for i, cl in enumerate(state.computes_left)
                        ),
                        # If the new task is the new running task, reset 'ld_left'
                        ld_left=C_TLB.get_s() if task_index < running_task_index else state.ld_left,
                    )
                )

            #
            # Go forward by one time-unit
            #
            if not self.only_periodic or len(can_start_job) == 0:
                new_start_delays_done = state.start_delays_done
                new_computes_left = state.computes_left
                new_ld_left = state.ld_left

                if running_task_index < len(self.tasks):
                    # Update compute & delay status of the running task
                    if state.ld_left == 0:
                        if not new_start_delays_done[running_task_index]:
                            # Unlikely case where SD=0 and RD>0.
                            # Cannot be done before, since we must wait for the first 'real'
                            # compute slot to register the zero-length starting delay.
                            new_start_delays_done = tuple(
                                True if i == running_task_index else sdd
                                for i, sdd in enumerate(state.start_delays_done)
                            )

                        new_computes_left = tuple(
                            cl - 1 if i == running_task_index else cl for i, cl in enumerate(state.computes_left)
                        )

                        if new_computes_left[running_task_index] == 0:
                            # Job ended, update max response time && set ld_left for the next task
                            self.max_seen_response_time[running_task_index] = max(
                                self.max_seen_response_time[running_task_index],
                                state.jobs_started[running_task_index] + 1,
                            )

                            assert new_ld_left == 0
                            new_running_task_index = self._get_running_task_index(new_computes_left)
                            if new_running_task_index < len(self.tasks):
                                new_ld_left = C_TLB.get_s()  # TODO: differentiate starting/resume
                    else:
                        new_ld_left -= 1
                        if new_ld_left == 0 and not new_start_delays_done[running_task_index]:
                            new_start_delays_done = tuple(
                                True if i == running_task_index else sdd
                                for i, sdd in enumerate(state.start_delays_done)
                            )

                new_state = self.State(
                    # Jobs started are now one unit older
                    jobs_started=tuple(
                        min(job_started + 1, self.tasks[i].period.get_s())
                        for i, job_started in enumerate(state.jobs_started)
                    ),
                    start_delays_done=new_start_delays_done,
                    computes_left=new_computes_left,
                    ld_left=new_ld_left,
                )

                # Check for deadline miss
                if any(
                    job_started >= self.tasks[i].period.get_s()
                    for i, (job_started, compute_left) in enumerate(
                        zip(new_state.jobs_started, new_state.computes_left)
                    )
                    if compute_left > 0
                ):
                    if self.verbose > 0:
                        print(f"  SS: deadline miss for: {state}")
                    raise DeadlineMissException("Some job did not finish on time.")

                self._push_state(new_state)

    def __init__(self, instance: Instance, only_periodic: bool = False, verbose: int = 0) -> None:
        self.only_periodic = only_periodic  # TODO: add "has started" to state to handle periodic + offset
        self.verbose = verbose

        assert C_TLB.get_us() == 0
        assert all(task.period.get_us() == 0 for task in instance.tasks)
        assert all(task.compute.get_us() == 0 for task in instance.tasks)
        self.tasks = instance.tasks

        self.seen_states: set[SimulationSporadic.State] = set()
        self.new_states: set[SimulationSporadic.State] = set()
        self.max_seen_response_time = [0] * len(instance.tasks)

        # Initialization: any task can start a job, none has any compute left
        initial_state = self.State(
            jobs_started=tuple(task.period.get_s() for task in instance.tasks),
            start_delays_done=tuple([False] * len(instance.tasks)),
            computes_left=tuple([0] * len(instance.tasks)),
            ld_left=0,
        )
        if self.verbose > 0:
            print("  SS: pushing initial state")
        self._push_state(initial_state)

        self.run()


class Experience:
    DO_DRAW = False  # slow
    ExpData = dict[int, dict[str, int]]

    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose

        self.tg = TaskGenerator.TaskGenerator()

    def run_single_instance(self, g: Instance) -> typing.Optional[bool]:
        tlb_sim_worked = None
        if g.regular_schedulable:
            SimulationPeriodic(verbose=0).simulate(g, tlbcost_mode=SimulationPeriodic.TlbCostMode.No)
            tlb_sim_worked = True
            if g.tlb_schedulable:
                SimulationPeriodic(verbose=0).simulate(g, tlbcost_mode=SimulationPeriodic.TlbCostMode.Add)
            else:
                try:
                    SimulationPeriodic(verbose=0).simulate(g, tlbcost_mode=SimulationPeriodic.TlbCostMode.Add)
                except DeadlineMissException:
                    tlb_sim_worked = False
        return tlb_sim_worked

    def plot_util_range(self, data: ExpData, filename: str, title: str) -> None:
        tp = [data[k]["tp"] for k in data]
        fp = [data[k]["fp"] for k in data]
        _, ax = plt.subplots()
        ax.stackplot(data.keys(), tp, fp, labels=["tlb_sched", "unexpected_works"])
        ax.legend(loc="upper right")
        ax.margins(0, 0)
        ax.set_ylim(0, 100)
        plt.title(title)

        save_dir = "rfig"
        os.makedirs(save_dir, exist_ok=True)
        for save_format in ["svg", "pdf"]:
            plt.savefig(f"{save_dir}/{filename}.{save_format}", format=save_format, bbox_inches="tight")

    def run_util_range(self) -> ExpData:
        NB_TRIES = 1_000
        MAX_SAMPLES = 100
        pad_a = math.ceil(math.log10(NB_TRIES + 1))
        self.verbose = 0

        results = {}
        for util in range(10, 99, 1):
            self.tg.utilization = util / 100.0
            stats_schedulable_reg = 0
            stats_schedulable_tlb = 0
            stats_unexpected_unschedulable_tlb = 0
            for i in range(NB_TRIES):
                if stats_schedulable_reg >= MAX_SAMPLES:
                    break
                do_simu = util >= 0 and i >= 0
                g = Instance(self.tg, verbose=0)
                if do_simu:
                    if self.verbose > 0:
                        print(f"{colorama.Style.BRIGHT}Instance {i:3}:{colorama.Style.NORMAL} {g}]")
                    if self.DO_DRAW:
                        g.draw(f"util{util:02}_{i}", self.tg.trunc_digits)

                    if not g.regular_schedulable:
                        continue
                    tlb_sim_worked = self.run_single_instance(g)
                    stats_schedulable_reg += 1
                    if g.tlb_schedulable:
                        stats_schedulable_tlb += 1
                    elif tlb_sim_worked:
                        stats_unexpected_unschedulable_tlb += 1
                        # raise RuntimeError("Simulation unexpectedly worked")

            true_positive_percent = int(stats_schedulable_tlb / stats_schedulable_reg * 100)
            false_positive_percent = int(stats_unexpected_unschedulable_tlb / stats_schedulable_reg * 100)
            print(
                f"Util: {util}  -> {stats_schedulable_tlb:>{pad_a}}/{stats_schedulable_reg:>{pad_a}}",
                f"({true_positive_percent:>3}%) (false pos.:{false_positive_percent:>3}%)",
            )
            results[util] = {"tp": true_positive_percent, "fp": false_positive_percent}
        return results

    def run_bruteforce(self) -> None:
        self.tg.comboGenerator = TaskGenerator.UtilAlgorithm.Bruteforce
        assert C_TLB.get_us() == 0
        self.tg.c_tlb_s = C_TLB.get_s()
        SKIP_NOT_TLB_SCHEDULABLE = False
        RUN_PERIODIC = False
        RUN_SPORADIC = True

        for i in itertools.count():
            g = Instance(self.tg, verbose=0)

            if SKIP_NOT_TLB_SCHEDULABLE and not g.tlb_schedulable:
                continue

            if self.verbose >= 0:
                print(f"{colorama.Style.BRIGHT}Instance {i:5}:{colorama.Style.NORMAL}", g)

            for do_run, only_periodic, run_name in [
                (RUN_PERIODIC, True, "Periodic"),
                (RUN_SPORADIC, False, "Sporadic"),
            ]:
                if not do_run:
                    continue
                try:
                    ss = SimulationSporadic(g, only_periodic=only_periodic, verbose=0)
                except DeadlineMissException:
                    ss = None
                if g.tlb_schedulable:
                    assert ss is not None
                elif ss is not None:
                    print(f"[{run_name}] Simulation unexpectedly worked ({len(ss.seen_states):5} states)")

                if ss is not None:
                    for ss_seen_ri, task in zip(ss.max_seen_response_time, g.tasks):
                        if ss_seen_ri < task.tlb_response_time.get_s():
                            print(
                                f"[{run_name}] Did not see computed response time for task {task}:"
                                + f" {ss_seen_ri} < {task.tlb_response_time.get_s()}"
                            )


def make_util_graphs() -> None:
    exp = Experience()
    exp.tg.period_interval_max = 100

    for nb_tasks in range(2, 10):
        for combogen, combotxt in [
            (TaskGenerator.UtilAlgorithm.RandFixedSum, "RFS"),
            (TaskGenerator.UtilAlgorithm.UUniFast_Discard, "UUD"),
            (TaskGenerator.UtilAlgorithm.Kato, "Kato"),
        ]:
            exp.tg.nb_tasks = nb_tasks
            exp.tg.comboGenerator = combogen
            name = f"{combotxt}_{exp.tg.nb_tasks}t"
            print(f"## Run EXP {name}")
            res = exp.run_util_range()
            exp.plot_util_range(res, name, name)


def main() -> None:
    np.random.seed(0)

    exp = Experience(verbose=0)
    exp.tg.nb_tasks = 3
    exp.tg.comboGenerator = TaskGenerator.UtilAlgorithm.RandFixedSum

    exp.tg.period_interval_max = 100  # For smaller hyperperiod

    mcase = 1
    match mcase:
        case 0:
            exp.run_util_range()
        case 1:
            exp.run_bruteforce()
        case 2:
            make_util_graphs()


if __name__ == "__main__":
    main()
    sys.exit(0)
