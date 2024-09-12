#!/usr/bin/env bash
set -e

root=$1
device="${2:-cuda:0}"

if [ -z "$root" ]
then
    echo "Please enter the root folder of dataset or -h, --help for help."
    exit 1
fi

if [ "$root" = "-h" ] || [ "$root" = "--help" ]
then
    echo "Usage: $0 ROOT_FOLDER [DEVICE, e.g. cuda:4]"
    exit 0
fi

echo "Root folder: $root"
echo "Device: $device"

python inference.py "$root/backgrounds" "$root/foregrounds" --out-dir "$root/harmonized" --device "$device" --deterministic
