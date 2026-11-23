#!/bin/sh
# Build the Senja v6 native addon (x.node) with gcc.
#
# Default: plain -O3 + LTO build (shipped). PGO is an option
# (`./build.sh pgo`) — measure A/B before trusting it; v5 proved PGO
# slower on this architecture, so the default stays no-PGO.
#   ./build.sh        — plain -O3 + LTO build (default, shipped)
#   ./build.sh pgo    — two-stage PGO build (profile workload:
#                       verify/profile-workload.mjs over practice +
#                       two fresh draws; .gcda regenerated each run)
#
# -march=x86-64-v2: SSE2/SSSE3 baseline (the fallback path). AVX2 is
# dispatched at runtime via __builtin_cpu_supports, so the binary stays
# correct on hosts without AVX2.
# -flto: link-time optimisation across engine.c / unicode_tables.c.
# -Wl,--version-script=exports.map: export only the two N-API entry
#   symbols (see exports.map — -fvisibility=hidden would hide
#   napi_register_module_v1 and Node could not load the addon).
# strip: drop the symbol table (dynamic symbols stay, per the map).
set -e
cd "$(dirname "$0")"
CC="${CC:-cc}"
CFLAGS="-O3 -march=x86-64-v2 -std=c11 -shared -fPIC -I/usr/include/node -DNAPI_VERSION=8 -flto -Wl,--version-script=exports.map -Wall -Wextra -Wno-unused-parameter"
if [ "$1" = pgo ]; then
  rm -f ./*.gcda
  # shellcheck disable=SC2086
  "$CC" $CFLAGS -fprofile-generate -o x.node engine.c unicode_tables.c
  node ../verify/profile-workload.mjs
  # shellcheck disable=SC2086
  "$CC" $CFLAGS -fprofile-use -o x.node engine.c unicode_tables.c
  rm -f ./*.gcda
else
  # shellcheck disable=SC2086
  "$CC" $CFLAGS -o x.node engine.c unicode_tables.c
fi
strip x.node
ls -la x.node
