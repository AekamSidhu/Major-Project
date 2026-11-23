#!/usr/bin/env bash
# Re-renders every diagram in this folder to PNG and SVG next to its source.
# Fails (non-zero exit) on the first source that does not render.
#
# Needs: Java 8 or newer, PlantUML (set PLANTUML_JAR to the plantuml .jar file), and
# Graphviz `dot` on PATH. Tested with PlantUML 1.2024.8 on Java 8 and Graphviz 14.
# Run from anywhere:  bash diagrams/render.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

jar="${PLANTUML_JAR:-}"
if [ -z "$jar" ] || [ ! -f "$jar" ]; then
  echo "PlantUML jar not found (set PLANTUML_JAR to the path of plantuml.jar)" >&2
  exit 2
fi
command -v dot >/dev/null 2>&1 || { echo "Graphviz dot not found on PATH" >&2; exit 2; }
command -v java >/dev/null 2>&1 || { echo "java not found on PATH" >&2; exit 2; }

status=0
count=0

# Every source must leave both a PNG and an SVG next to it (PlantUML can exit 0 with "no image").
check_outputs() {
  local base="$1"
  if [ ! -s "$base.png" ] || [ ! -s "$base.svg" ]; then
    echo "NO RENDER (png and svg must both exist): $base" >&2
    status=1
  fi
}

# PlantUML: -failfast2 stops at the first syntax error and returns non-zero.
while IFS= read -r -d '' src; do
  echo "plantuml: $src"
  base="${src%.puml}"
  rm -f "$base.png" "$base.svg"  # a stale render must not pass for a new one
  # PlantUML honours only the last output format given, so each format is a separate run.
  if ! java -Djava.awt.headless=true -jar "$jar" -charset UTF-8 -failfast2 -tpng "$src" ||
     ! java -Djava.awt.headless=true -jar "$jar" -charset UTF-8 -failfast2 -tsvg "$src"; then
    echo "RENDER FAILED: $src" >&2
    status=1
  fi
  check_outputs "$base"
  count=$((count + 1))
done < <(find . -name '*.puml' -print0 | sort -z)

# Graphviz sources: PNG and SVG next to each .dot.
while IFS= read -r -d '' src; do
  base="${src%.dot}"
  echo "dot: $src"
  rm -f "$base.png" "$base.svg"
  if ! dot -Tpng "$src" -o "$base.png" || ! dot -Tsvg "$src" -o "$base.svg"; then
    echo "RENDER FAILED: $src" >&2
    status=1
  fi
  check_outputs "$base"
  count=$((count + 1))
done < <(find . -name '*.dot' -print0 | sort -z)

# Every render must have a source (no orphans); a .dot or .puml renders to its own base name.
while IFS= read -r -d '' out; do
  base="${out%.*}"
  if [ ! -f "$base.puml" ] && [ ! -f "$base.dot" ]; then
    echo "ORPHAN RENDER (no source): $out" >&2
    status=1
  fi
done < <(find . \( -name '*.png' -o -name '*.svg' \) -print0)

echo "rendered $count sources, status $status"
exit $status
