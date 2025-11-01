#!/usr/bin/env python3

import csv
import dataclasses
import enum
import itertools
import math
import os
import subprocess
import sys
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatOrNPFloat = typing.TypeVar("FloatOrNPFloat", float, npt.NDArray[np.floating[typing.Any]])
T = typing.TypeVar("T")


@dataclasses.dataclass
class DataPoint:
    all: int = 0
    bad: int = 0
    pessimism: float = 0.0


RawDataN = dict[int, dict[int, DataPoint]]
RawData = dict[int, RawDataN]
NPData = np.ndarray[tuple[typing.Any, typing.Any], np.dtype[np.float64]]


class Plot:
    UTIL_BINS = 101
    Y_BINS = 101
    CONFIDENCE_FULL = 100  # TODO: think about this

    @enum.verify(enum.UNIQUE)
    class PlotType(enum.Enum):
        FalseNegativeRatio = 1
        PessimismRatio = 2

        def __str__(self, fn: bool = False) -> str:
            if fn:
                match self.value:
                    case self.FalseNegativeRatio.value:  # type: ignore
                        return "fnr"
                    case self.PessimismRatio.value:  # type: ignore
                        return "pr"
                    case _:
                        raise ValueError(f"Bad PlotType: '{self.value}'")
            else:
                match self.value:
                    case self.FalseNegativeRatio.value:  # type: ignore
                        return "False negative ratio"
                    case self.PessimismRatio.value:  # type: ignore
                        return "Pessimism ratio"
                    case _:
                        raise ValueError(f"Bad PlotType: '{self.value}'")

    @enum.verify(enum.UNIQUE)
    class YBinType(enum.Enum):
        rnld_rn = 1
        max_rild_ri = 2
        rtn_rn = 3
        max_rti_ri = 4
        badldutil_util = 5
        badldutil = 6
        badldutil_delta = 7
        ldutil_util = 8
        ldutil = 9
        ldutil_delta = 10

    @staticmethod
    def tbool(s: str) -> bool:
        if s == "T":
            return True
        if s == "F":
            return False
        assert False

    def init_raw_data_for_i_tasks(self, raw_data: RawData, n: int) -> None:
        raw_data[n] = {}
        for i in range(self.UTIL_BINS):
            raw_data[n][i] = {}
            for j in range(self.Y_BINS):
                raw_data[n][i][j] = DataPoint()

    def yify(self, y_values: FloatOrNPFloat, reverse: bool = False) -> FloatOrNPFloat:
        match self.ybin_type:
            case (
                self.YBinType.rnld_rn
                | self.YBinType.max_rild_ri
                | self.YBinType.rtn_rn
                | self.YBinType.max_rti_ri
                | self.YBinType.badldutil_util
                | self.YBinType.ldutil_util
            ):
                Y_MAX = 10.0
                mult_factor = (self.Y_BINS - 1) / math.log(Y_MAX)
                if reverse:
                    return np.exp(y_values / mult_factor)
                return np.log(y_values) * mult_factor
            case (
                self.YBinType.badldutil
                | self.YBinType.badldutil_delta
                | self.YBinType.ldutil
                | self.YBinType.ldutil_delta
            ):
                Y_MAX = 2.0
                if self.ybin_type == self.YBinType.ldutil_delta:
                    Y_MAX = 1.0
                mult_factor = (self.Y_BINS - 1) / Y_MAX
                if reverse:
                    return y_values / mult_factor
                return y_values * mult_factor
            case _:
                raise ValueError(f"Unknown YBin type: {self.ybin_type}")

    @staticmethod
    def simutype_str(simutype: str, filename: bool = False) -> str:
        match simutype:
            case "S":
                out = "sporadic"
            case "O":
                out = "periodic any offset"
            case "P":
                out = "periodic given offsets"
            case _:
                raise ValueError(f"Unknown simulation type '{simutype}'")
        if filename:
            out = out.replace(" ", "-")
        return out

    @staticmethod
    def parse_csv_array_elem(elem: str) -> list[float]:
        if elem == "":
            return []
        return [float(x) for x in elem.split(",")]

    @staticmethod
    def optional_csv_val(old_val: typing.Optional[T], new_val: T) -> T:
        if old_val is None:
            return new_val
        assert old_val == new_val
        return new_val

    def parse(self, infile: str) -> tuple[RawData, str, int, bool]:
        raw_data: RawData = {}
        simu_type = None
        max_period = None
        sd_geq_rd = None

        with open(infile, encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=";", quotechar="|")
            had_error = False
            for row in csv_reader:
                try:
                    (
                        _nb_tasks,
                        _simu_type,
                        _max_period,
                        _sd_geq_rd,
                        _util,
                        _badldutil,
                        _ldutil_delta,
                        _ld_schedulable,
                        _simu_worked,
                        _ri_arr,
                        _rild_arr,
                        _rt_arr,
                    ) = row
                except ValueError as exc:
                    if had_error:
                        raise exc
                    had_error = True
                nb_tasks = int(_nb_tasks)
                simu_type = self.optional_csv_val(simu_type, _simu_type)
                max_period = self.optional_csv_val(max_period, int(_max_period))
                sd_geq_rd = self.optional_csv_val(sd_geq_rd, self.tbool(_sd_geq_rd))
                util = float(_util)
                badldutil = float(_badldutil)
                ldutil = float(_ldutil_delta) + util
                ld_schedulable = self.tbool(_ld_schedulable)
                simu_worked = self.tbool(_simu_worked)
                ri_arr = self.parse_csv_array_elem(_ri_arr)
                rild_arr = self.parse_csv_array_elem(_rild_arr)
                rt_arr = self.parse_csv_array_elem(_rt_arr)

                if badldutil < util:
                    if util - badldutil > 0.001:  # Not a floating-point issue...
                        raise ValueError(f"Have badldutil < util | {row=}")
                    badldutil = util

                if nb_tasks not in raw_data:
                    self.init_raw_data_for_i_tasks(raw_data, nb_tasks)
                util_pct = int(util * (self.UTIL_BINS - 1))
                match self.ybin_type:
                    case self.YBinType.rnld_rn:
                        y_bin = int(self.yify(rild_arr[-1] / ri_arr[-1]))
                    case self.YBinType.max_rild_ri:
                        max_rild_ri_ratio = max(rild / ri for ri, rild in zip(ri_arr, rild_arr))
                        y_bin = int(self.yify(max_rild_ri_ratio))
                    case self.YBinType.rtn_rn:
                        raw_ratio = rt_arr[-1] / ri_arr[-1]
                        y_bin = int(self.yify(max(1.0, raw_ratio)))
                    case self.YBinType.max_rti_ri:
                        max_rti_ri_ratio = max(rti / ri for ri, rti in zip(ri_arr, rt_arr))
                        y_bin = int(self.yify(max_rti_ri_ratio))
                    case self.YBinType.badldutil_util:
                        y_bin = int(self.yify(badldutil / util))
                    case self.YBinType.badldutil:
                        y_bin = int(self.yify(badldutil))
                    case self.YBinType.badldutil_delta:
                        y_bin = int(self.yify(badldutil - util))
                    case self.YBinType.ldutil_util:
                        y_bin = int(self.yify(ldutil / util))
                    case self.YBinType.ldutil:
                        y_bin = int(self.yify(ldutil))
                    case self.YBinType.ldutil_delta:
                        y_bin = int(self.yify(ldutil - util))
                    case _:
                        raise ValueError(f"Unknown YBin type: {self.ybin_type}")
                if y_bin < self.Y_BINS:
                    if simu_worked or True:
                        raw_data[nb_tasks][util_pct][y_bin].all += 1
                    if simu_worked and not ld_schedulable:
                        raw_data[nb_tasks][util_pct][y_bin].bad += 1
                    if simu_worked and self.plot_type == self.PlotType.PessimismRatio:
                        max_rild_rti_ratio = max(rild / rti for rti, rild in zip(rt_arr, rild_arr))
                        raw_data[nb_tasks][util_pct][y_bin].pessimism += max_rild_rti_ratio
        assert simu_type is not None
        assert max_period is not None
        assert sd_geq_rd is not None

        return raw_data, simu_type, max_period, sd_geq_rd

    @staticmethod
    def make_cmap(bad: float, good: float) -> tuple[matplotlib.colors.LinearSegmentedColormap, plt.Normalize]:
        col_vals = [0, 0.2, 0.4, 0.6, 0.8, 1]
        colors = ["magenta", "red", "orange", "yellow", "green", "lime"]

        if bad > good:
            colors.reverse()
            good, bad = bad, good

        col_vals = [bad + v * (good - bad) for v in col_vals]

        norm = plt.Normalize(vmin=min(col_vals), vmax=max(col_vals), clip=True)
        cc_tuples = list(zip(map(norm, col_vals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cc_tuples)

        return cmap, norm

    def plot_one(
        self,
        raw_data: RawData,
        simu_type: str,
        max_period: int,
        sd_geq_rd: bool,
        enable_confidence: bool,
        grid_x: int,
    ) -> None:
        grid_y = len(raw_data) // grid_x

        data_list = []
        for raw_data_n in raw_data.values():
            data_n = np.empty((self.UTIL_BINS, self.Y_BINS)).astype(complex)
            for i, rrow in raw_data_n.items():
                for j, val in rrow.items():
                    ratio = 1.0
                    confidence = 0.0
                    if val.all > 0:
                        match self.plot_type:
                            case self.PlotType.FalseNegativeRatio:
                                ratio = val.bad / val.all
                            case self.PlotType.PessimismRatio:
                                ratio = val.pessimism / val.all
                            case _:
                                raise ValueError(f"Unknown plot_type: {self.plot_type}")

                        confidence = 1.0
                        if enable_confidence:
                            val_all = min(val.all, self.CONFIDENCE_FULL)
                            confidence = math.log(val_all) / math.log(self.CONFIDENCE_FULL)
                    data_n[j][i] = ratio + confidence * 1j
            data_list.append(data_n)

        delay_ratio_str: dict[Plot.YBinType, tuple[str, str]] = {
            self.YBinType.rnld_rn: ("Delay ratio (rnld/rn)", "darkorange"),
            self.YBinType.max_rild_ri: ("Delay ratio (max_i(rild/ri))", "darkgreen"),
            self.YBinType.rtn_rn: ("Delay ratio (rtn/rn)", "darkblue"),
            self.YBinType.max_rti_ri: ("Delay ratio (max_i(rti/ri))", "darkviolet"),
            self.YBinType.badldutil_util: ("badldutil/util", "grey"),
            self.YBinType.badldutil: ("badldutil", "grey"),
            self.YBinType.badldutil_delta: ("badldutil - util", "goldenrod"),
            self.YBinType.ldutil_util: ("LDutil/util)", "grey"),
            self.YBinType.ldutil: ("LDutil", "grey"),
            self.YBinType.ldutil_delta: ("UDE", "black"),
        }

        match self.plot_type:
            case self.PlotType.FalseNegativeRatio:
                cmap, norm = self.make_cmap(good=0.0, bad=self.max_false_negative_ratio)
            case self.PlotType.PessimismRatio:
                cmap, norm = self.make_cmap(good=1.0, bad=self.max_pessimism_ratio)
            case _:
                raise ValueError(f"Unknown plot_type: {self.plot_type}")
        rgba_data_list = []
        for data_n in data_list:
            rgba_data = cmap(norm(np.real(data_n)))
            rgba_data[..., -1] = np.imag(data_n)
            rgba_data_list.append(rgba_data)

        match self.ybin_type:
            case (
                self.YBinType.rnld_rn
                | self.YBinType.max_rild_ri
                | self.YBinType.rtn_rn
                | self.YBinType.max_rti_ri
                | self.YBinType.badldutil_util
                | self.YBinType.ldutil_util
            ):
                # Draw the "utilrd = 1" limit
                utilmax_x = np.arange(1, self.UTIL_BINS)
                utilmax_y = self.yify(self.UTIL_BINS / utilmax_x)
            case self.YBinType.badldutil | self.YBinType.ldutil:
                utilmax_x = np.arange(0, self.UTIL_BINS)
                utilmax_y = self.yify(np.ones_like(utilmax_x))
            case self.YBinType.badldutil_delta | self.YBinType.ldutil_delta:
                utilmax_x = np.arange(0, self.UTIL_BINS)
                utilmax_y = self.yify(np.ones_like(utilmax_x) - utilmax_x / self.UTIL_BINS)
            case _:
                raise ValueError(f"Unknown YBin type: {self.ybin_type}")

        greys = np.full((*data_list[0].shape, 3), 0, dtype=np.uint8)  # Grey background

        fontsize = None
        labelsize = None
        if self.paper:
            fontsize = 8
            labelsize = 6

        fig, axs = plt.subplots(grid_y, grid_x, sharex=True, sharey=True)
        for idx, (n, rgba_data) in enumerate(zip(raw_data.keys(), rgba_data_list)):
            ax_x = idx % grid_x
            ax_y = idx // grid_x
            if grid_y == 1:
                ax_i = axs[ax_x]
            else:
                ax_i = axs[ax_y, ax_x]

            ax_i.imshow(greys)
            ax_i.imshow(rgba_data)
            min_utilization = n * (1.0 / max_period) * self.UTIL_BINS
            if self.enable_lines:
                ax_i.vlines(min_utilization, 0, self.Y_BINS, color="cyan", alpha=0.5)  # min-util vertical line
                ax_i.plot(utilmax_x, utilmax_y, color="cyan", alpha=0.5)

            title = (
                f"{self.plot_type}\n"
                f"{n} tasks, {self.simutype_str(simu_type)}\n"
                f"{"SD≥RD, " if sd_geq_rd else ""}"
                f"with{"" if enable_confidence else "out"} confidence"
            )
            title_loc = None
            title_y = None
            if self.short_title:
                title = f" {n} tasks"
                title_loc = "left"
                title_y = 0.85
            ax_i.set_title(title, loc=title_loc, y=title_y, color="white", fontsize=fontsize)

            if ax_y == grid_y - 1:
                ax_i.set_xlabel("Utilization", fontsize=fontsize)
            if ax_x == 0:
                ax_i.set_ylabel(
                    f"{delay_ratio_str[self.ybin_type][0]}", color=delay_ratio_str[self.ybin_type][1], fontsize=fontsize
                )

            xticks = np.linspace(0, self.UTIL_BINS - 1, num=11)
            xticks_labels = [xtick / 100 for xtick in xticks]
            yticks = np.linspace(0, self.Y_BINS - 1, num=11)
            yticks_labels = self.yify(yticks, reverse=True)
            yticks_labels = np.round(yticks_labels, 2)
            plt.ylim((yticks_labels[-1], 0))
            ax_i.set_xticks(xticks, labels=xticks_labels, rotation=45, ha="right", rotation_mode="anchor")
            ax_i.invert_yaxis()
            ax_i.set_yticks(yticks, labels=yticks_labels)
            ax_i.tick_params(axis="both", which="major", labelsize=labelsize)

        # fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=-0.4)

        # Colors legend, i.e. the colorbar on the right
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        clb_axs = axs[: grid_x - 1]
        clb_shrink = 0.5
        clb_labelpad = -32
        if grid_y == 1:
            clb_axs = axs
            clb_shrink = 0.33
            clb_labelpad = -30
        clb = fig.colorbar(sm, ax=clb_axs, pad=0.03, shrink=clb_shrink)
        clb.set_label(label=f"{self.plot_type}", labelpad=clb_labelpad, fontsize=fontsize)
        clb.ax.tick_params(labelsize=labelsize)

        os.makedirs(self.SAVE_DIR, exist_ok=True)
        filename = (
            f"{self.simutype_str(simu_type, filename = True)}_{"conf" if enable_confidence else "noconf"}"
            + f"_{self.distrib}_{self.plot_type.__str__(fn=True)}"
        )
        for save_format in ["svg", "pdf"]:
            plt.savefig(
                f"{self.SAVE_DIR}/{filename}.{save_format}", format=save_format, bbox_inches="tight", pad_inches=0.0
            )
        plt.close(fig)

    def plot_all(self, raw_data: RawData, simu_type: str, max_period: int, sd_geq_rd: bool) -> None:
        match self.confidence:
            case "both":
                conf_array = [True, False]
            case "yes":
                conf_array = [True]
            case "no":
                conf_array = [False]

        grid_x = 3

        for enable_confidence in conf_array:
            self.plot_one(raw_data, simu_type, max_period, sd_geq_rd, enable_confidence, grid_x)

    def montage(self) -> None:
        files = ""
        if self.confidence in ["yes", "both"]:
            files += f' "{self.SAVE_DIR}/*_conf_*.svg"'
        if self.confidence in ["no", "both"]:
            files += f' "{self.SAVE_DIR}/*_noconf_*.svg"'

        subprocess.check_call(
            [f"magick montage {files}" + f' -geometry +2+2 -tile x2 "{self.SAVE_DIR}/out.png"'],
            shell=True,
        )

    def __init__(
        self,
        infile: str,
        out_name: str,
        plot_type: PlotType,
        ybin_type: YBinType,
        distrib: str,
        confidence: str = "both",
        enable_lines: bool = True,
        short_title: bool = False,
        paper: bool = False,
        max_false_negative_ratio: float = 1.0,
        max_pessimism_ratio: float = 2.0,
    ) -> None:
        self.SAVE_DIR = f"fig/{out_name}"
        self.plot_type = plot_type
        self.ybin_type = ybin_type
        self.distrib = distrib
        self.confidence = confidence
        self.enable_lines = enable_lines
        self.short_title = short_title
        self.paper = paper
        self.max_false_negative_ratio = max_false_negative_ratio
        self.max_pessimism_ratio = max_pessimism_ratio

        raw_data, simu_type, max_period, sd_geq_rd = self.parse(infile)
        self.plot_all(raw_data, simu_type, max_period, sd_geq_rd)
        # self.montage()


def plot_B(inpath: str) -> None:
    simu_types = ["pergiven", "spo"]  #  ["pergiven", "perany", "spo"]
    per_gens = ["shp"]  # ["dumb", "shp"]
    plot_types = [Plot.PlotType.FalseNegativeRatio, Plot.PlotType.PessimismRatio]
    ybin_ratios = [
        # Plot.YBinType.rnld_rn,
        # Plot.YBinType.max_rild_ri,
        # Plot.YBinType.rtn_rn,
        Plot.YBinType.max_rti_ri,
        # Plot.YBinType.badldutil_util,
        # Plot.YBinType.badldutil,
        Plot.YBinType.badldutil_delta,
        Plot.YBinType.ldutil_delta,
    ]
    for elem in itertools.product(simu_types, per_gens, plot_types, ybin_ratios):
        print(f"Plotting {elem=}")
        simu_type, per_gen, plot_type, ybin_ratio = elem
        plot_type_fn = str(plot_type).replace(" ", "-")
        ybin_ratio_fn = f"YBinRatio{ybin_ratio.value}"
        out_name = f"{simu_type}_{per_gen}_{plot_type_fn}_{ybin_ratio_fn}"
        Plot(f"{inpath}/out_{simu_type}_{per_gen}.txt", out_name, plot_type, ybin_ratio, "unknown")


def plot_paper(inpath: str) -> None:
    do_plot = [1, 1, 1]
    out_name="paper"
    if do_plot[0]:
        print("Plotting vsPeriodic")
        simu_type = "pergiven"
        per_gen = "shp"
        distrib = "mix"
        Plot(
            f"{inpath}/out_{simu_type}_{per_gen}_{distrib}.txt",
            out_name=out_name,
            plot_type=Plot.PlotType.FalseNegativeRatio,
            ybin_type=Plot.YBinType.ldutil_delta,
            distrib=distrib,
            confidence="yes",
            enable_lines=False,
            short_title=True,
            paper=True,
        )

    if do_plot[1]:
        print("Plotting Sporadic FNR")
        simu_type = "spo"
        per_gen = "shp"
        distrib = "mix"
        Plot(
            f"{inpath}/out_{simu_type}_{per_gen}_{distrib}.txt",
            out_name=out_name,
            plot_type=Plot.PlotType.FalseNegativeRatio,
            ybin_type=Plot.YBinType.ldutil_delta,
            distrib=distrib,
            confidence="yes",
            enable_lines=False,
            short_title=True,
            paper=True,
            max_false_negative_ratio = 0.4
        )

    if do_plot[2]:
        print("Plotting Sporadic PR")
        simu_type = "spo"
        per_gen = "shp"
        distrib = "mix"
        Plot(
            f"{inpath}/out_{simu_type}_{per_gen}_{distrib}.txt",
            out_name=out_name,
            plot_type=Plot.PlotType.PessimismRatio,
            ybin_type=Plot.YBinType.ldutil_delta,
            distrib=distrib,
            confidence="yes",
            enable_lines=False,
            short_title=True,
            paper=True,
            max_pessimism_ratio = 1.4
        )


def main() -> None:
    inpath = "../fsimu/out/"
    # Plot(f"{inpath}/out_3.txt", "todo_3", Plot.PlotType.FalseNegativeRatio)
    # Plot(f"{inpath}/out_4.txt", "todo_4", Plot.PlotType.FalseNegativeRatio)
    # Plot(f"{inpath}/out_spo_dumb.txt", "todo_spo", Plot.PlotType.PessimismRatio, Plot.YBinType.badldutil_delta)
    # plot_B(inpath)
    plot_paper(inpath)


if __name__ == "__main__":
    main()
    sys.exit(0)
