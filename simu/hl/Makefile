MAKEFLAGS += --no-builtin-rules --no-builtin-variables -j
.SUFFIXES:

TASKSETS := ex1_synchronous ex1_ripple ex1_worst \
            ex2_00 ex2_01 ex2_02 ex2_03 ex2_04 ex2_05 \
            ex2_10 ex2_11 ex2_12 ex2_13 ex2_14 ex2_15  ex2_16 \
            ex2_20 ex2_21 ex2_22 \
            ex2_30 ex2_31 \
            ex2_40 ex2_41 \
            ex2_51 \
            ex2_21_spo \
            ex3_00_x ex3_01_x ex3_02_x ex3_03_6 ex3_03_7 ex3_04_x \
            ex3_10_x ex3_11_x ex3_12_x ex3_13_x ex3_14_7 ex3_14_8 ex3_15_x \
            ex3_20_x ex3_21_x ex3_22_x \
            ex3_30_x ex3_31_x \
            ex3_40_x ex3_41_x \
            ex3_51_x \
            ex4_worst_tau2 ex4_worst_tau3 \
            ex5_non_resumable_1 ex5_non_resumable_2 \
            ex6_base

all: $(addprefix pdf/,$(TASKSETS:=.pdf))

$(addprefix ts/,$(TASKSETS:=.txt)) $(addprefix ts/,$(TASKSETS:=.steps)) &: gen_tasksets.py Makefile
	./$<

pdf:
	@mkdir -p $@

svg:
	@mkdir -p $@

define BUILD_TS
pdf/$(1).pdf: svg/$(1).svg | pdf
	@# SELF_CALL due to https://gitlab.com/inkscape/inkscape/-/issues/4716
	SELF_CALL=xxx inkscape $$< -o $$@

svg/$(1).svg: ts/$(1).txt ts/$(1).steps ds_l Makefile | svg
	./$$(word 3,$$^) -qf -s 'FP' -m 'NRA' -v 'LT' -d "$$$$(cat $$(word 2,$$^))" $$< -o $$@
endef

$(foreach _TS, $(TASKSETS), $(eval $(call BUILD_TS,$(_TS))))

clean:
	rm -rf pdf svg ts


.PHONY: all clean
