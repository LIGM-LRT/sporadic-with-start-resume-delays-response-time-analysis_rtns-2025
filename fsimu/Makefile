MAKEFLAGS += -j "$(shell grep -c ^processor /proc/cpuinfo)" --no-builtin-rules --no-builtin-variables
.SUFFIXES:

GXX = g++
CLANG = clang++
CXX = $(GXX)
CXX = $(CLANG)
BUILDDIR = build
BIN = $(BUILDDIR)/out
BIN_PAPER_PER = $(BUILDDIR)/out_paper_per
BIN_PAPER_OLDFOR = $(BUILDDIR)/out_paper_oldfor
BASE_CXXFLAGS = -std=c++23 -O3 -Wall -Wextra -Wshadow -march=native -g -MMD -flto=auto
CXXFLAGS = $(BASE_CXXFLAGS)

MAPPER_DIR = $(BUILDDIR)/gcm.cache
GXX_EXTRA_CXXFLAGS = -fmodules '-fmodule-mapper=|@g++-mapper-server -r'$(MAPPER_DIR)
GXX_CXXFLAGS = $(BASE_CXXFLAGS) $(GXX_EXTRA_CXXFLAGS)
CLANG_BUILDDIR = $(BUILDDIR)/clang
CLANG_EXTRA_CXXFLAGS = -stdlib=libc++ -fprebuilt-module-path=$(CLANG_BUILDDIR)
CLANG_CXXFLAGS = $(BASE_CXXFLAGS) $(CLANG_EXTRA_CXXFLAGS)

GXX_STD_MODULE = $(BUILDDIR)/std.pcm
CLANG_STD_MODULE = $(CLANG_BUILDDIR)/std.pcm

ifeq ($(CXX), clang++)
	STD_MODULE = $(CLANG_STD_MODULE)
	CXXFLAGS += $(CLANG_EXTRA_CXXFLAGS)
else
	STD_MODULE = $(GXX_STD_MODULE)
	CXXFLAGS += $(GXX_EXTRA_CXXFLAGS)
endif

all: run $(CLANG_STD_MODULE)

run: $(BIN)
	./$<

paper_per: $(BIN_PAPER_PER)
	./$< 3

paper_oldfor: $(BIN_PAPER_OLDFOR)
	./$< 4


$(BUILDDIR) $(CLANG_BUILDDIR):
	@mkdir -p "$@"

$(BIN): main.cpp $(STD_MODULE) Makefile | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_PAPER_PER): main.cpp $(STD_MODULE) Makefile | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -DOVERRIDE_MAX_TASKS=16

$(BIN_PAPER_OLDFOR): main.cpp $(STD_MODULE) Makefile | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -DOVERRIDE_OLD_FORMULA

$(CLANG_STD_MODULE): /usr/lib/llvm-20/share/libc++/v1/std.cppm | $(CLANG_BUILDDIR) .clangd
	$(CLANG) -o $@ --precompile $(CLANG_CXXFLAGS) -Wno-reserved-module-identifier $^

.clangd:
	@echo "CompileFlags:\n  Add: [-std=c++23, -fprebuilt-module-path=$$(realpath $(BUILDDIR)),"\
	      " -xc++-module]" > $@

$(GXX_STD_MODULE): | $(BUILDDIR)
	$(GXX) -o $@ -c $(GXX_CXXFLAGS) -fsearch-include-path bits/std.cc

clean:
	rm -rf "$(BUILDDIR)"
