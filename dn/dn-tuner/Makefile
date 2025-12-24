#
# Drivenets 23/12/2025
#

.DEFAULT_GOAL: build
PLUGIN_SO:=librccl-tunerv4-dn.so
include makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.c)

default: ${BUILDDIR}/$(PLUGIN_SO)

build: ${BUILDDIR}/$(PLUGIN_SO)

${BUILDDIR}/$(PLUGIN_SO): plugin.c
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -Inccl $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

# Optimize configurations from performance data
optimize-config:
	@if [ -z "$(CSV_FILE)" ]; then \
		echo "Usage: make optimize-config CSV_FILE=path/to/data.csv [OUTPUT=config.conf] [METRIC=latency_us]"; \
		echo "Example: make optimize-config CSV_FILE=scripts/sample_performance_data.csv"; \
		exit 1; \
	fi
	python3 scripts/optimize_config.py $(CSV_FILE) \
		$(if $(OUTPUT),-o $(OUTPUT)) \
		$(if $(METRIC),-m $(METRIC)) \
		$(if $(SIZE_RANGES),--size-ranges $(SIZE_RANGES)) \
		$(if $(DRY_RUN),--dry-run) \
		$(if $(NO_HEADER),--no-header)

clean:
	rm -f ${BUILDDIR}/$(PLUGIN_SO)

.PHONY: optimize-config clean
