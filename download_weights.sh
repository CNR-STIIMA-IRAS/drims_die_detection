#!/usr/bin/env bash
# Download the model weights into weights/ (they are not tracked by git).
#
#   ./download_weights.sh            # fetch missing / corrupted files
#   ./download_weights.sh --force    # re-download everything
#   ./download_weights.sh --no-yoloe # only the die CNN (skips the 600 MB MobileCLIP)
#
# Sources:
#   yoloe-11s-seg.pt, mobileclip_blt.ts  official Ultralytics release assets
#   die_mobilenet_v3.pt                  our trained CNN: a GitHub release asset of this
#                                        repo, or Google Drive (DIE_CNN_URL=gdrive:<file-id>)
#
# Override the CNN source / checksum without editing the script:
#   DIE_CNN_URL=gdrive:1AbC...xyz DIE_CNN_SHA256=<sha256> ./download_weights.sh
# After retraining: upload the new file, then update DIE_CNN_URL / DIE_CNN_SHA256 below
# (sha256sum weights/die_mobilenet_v3.pt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"

ULTRALYTICS="https://github.com/ultralytics/assets/releases/download/v8.3.0"
DIE_CNN_URL="${DIE_CNN_URL:-https://github.com/CNR-STIIMA-IRAS/drims_die_detection/releases/download/weights-v1/die_mobilenet_v3.pt}"
DIE_CNN_SHA256="${DIE_CNN_SHA256:-c0f81b4344c9a4858d7fc72ddd04514b2bf77d95b2093922df5e7a674855b186}"

# name | url | sha256
FILES=(
    "die_mobilenet_v3.pt|${DIE_CNN_URL}|${DIE_CNN_SHA256}"
    "yoloe-11s-seg.pt|${ULTRALYTICS}/yoloe-11s-seg.pt|8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3"
    "mobileclip_blt.ts|${ULTRALYTICS}/mobileclip_blt.ts|a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54"
)

FORCE=0
SKIP_YOLOE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --no-yoloe) SKIP_YOLOE=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
mkdir -p "$WEIGHTS_DIR"

sha_ok() {   # sha_ok <file> <sha256>
    [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

download() {   # download <url> <dest>
    local url="$1" dest="$2"
    if [[ "$url" == gdrive:* ]]; then
        # Google Drive file shared as "Anyone with the link"; confirm=t skips the
        # "can't scan this file for viruses" page of large files
        url="https://drive.usercontent.google.com/download?id=${url#gdrive:}&export=download&confirm=t"
    fi
    curl -fL --retry 3 --progress-bar -o "$dest" "$url"
}

failed=0
for entry in "${FILES[@]}"; do
    IFS='|' read -r name url sha <<< "$entry"
    dest="${WEIGHTS_DIR}/${name}"
    if [ "$SKIP_YOLOE" = 1 ] && [ "$name" != "die_mobilenet_v3.pt" ]; then
        continue
    fi
    if [ "$FORCE" = 0 ] && sha_ok "$dest" "$sha"; then
        echo "✓ ${name} (already present)"
        continue
    fi
    echo "↓ ${name}  <-  ${url}"
    if ! download "$url" "${dest}.part"; then
        echo "✗ ${name}: download failed (${url})" >&2
        rm -f "${dest}.part"; failed=1; continue
    fi
    if ! sha_ok "${dest}.part" "$sha"; then
        echo "✗ ${name}: checksum mismatch — got $(sha256sum "${dest}.part" | cut -d' ' -f1)" >&2
        echo "  (wrong file, an HTML error page, or new weights: update the sha256)" >&2
        rm -f "${dest}.part"; failed=1; continue
    fi
    mv "${dest}.part" "$dest"
    echo "✓ ${name}"
done

if [ "$failed" = 1 ]; then
    echo "Some weights could not be downloaded — see above." >&2
    exit 1
fi
echo "All weights in ${WEIGHTS_DIR}"
