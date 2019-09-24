#!/bin/bash

set -e

PRODUCT_DIR="${VALIDATE_DRP_DIR}"
if [[ ${PRODUCT_DIR} == '' ]]; then
    PRODUCT_DIR='.'
fi
VALIDATION_DATA_DIR="${CI_HSC_DIR}/raw"
#  "ps1_pv3_3pi_20170110" is stored at the base level in ci_hsc.
PHOTOMETRIC_REF_CAT_DIR="${CI_HSC_DIR}"
CALIB_DATA="${CI_HSC_DIR}/CALIB"

CAMERA=HscQuick
CONFIG_FILE="${PRODUCT_DIR}/config/hscConfig.py"
MAPPER=lsst.obs.hsc.HscMapper

print_error() {
    >&2 echo "$@"
}

DOPROCESS=true
DOVERIFY=true

usage() {
    print_error
    print_error "Usage: $0 [-PV] [-h] [-- <extra options to validateDrp.py>]"
    print_error
    print_error "Specifc options:"
    print_error "   -P          Skip processing?"
    print_error "   -V          Skip verification?"
    print_error "   -f          Config file path"
    print_error "   -h          show this message"
    exit 1
}

while getopts "VPf:h" option; do
    case "$option" in
        P)  DOPROCESS=false;;
        V)  DOVERIFY=false;;
        f)  CONFIG_FILE="$OPTARG";;
        h)  usage;;
        *)  usage;;
    esac
done
shift $((OPTIND-1))

if [[ $DOPROCESS == true ]]; then
    "${PRODUCT_DIR}/examples/processData.sh" \
        -c "$CAMERA" \
        -m "$MAPPER" \
        -v "$VALIDATION_DATA_DIR" \
        -p "$PHOTOMETRIC_REF_CAT_DIR" \
        -f "$CONFIG_FILE" \
        -e "fits" \
        -d "$CALIB_DATA" \
        -r
fi

if [[ $DOVERIFY == true ]]; then
    "${PRODUCT_DIR}/examples/validateRepo.sh" \
        -c "$CAMERA" \
        -- "$@"
fi
