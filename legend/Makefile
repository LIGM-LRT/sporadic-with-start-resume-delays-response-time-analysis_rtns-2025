MAKEFLAGS += --no-builtin-rules --no-builtin-variables -j
.SUFFIXES:

ITEMS := release deadline sd rd compute

all: $(addprefix pdf/,$(ITEMS:=.pdf))

pdf:
	@mkdir -p $@

define BUILD_PDF
pdf/$(1).pdf: $(1).svg | pdf
	inkscape $$< -o $$@
endef

$(foreach _I, $(ITEMS), $(eval $(call BUILD_PDF,$(_I))))

clean:
	rm -rf pdf svg ts


.PHONY: all clean
