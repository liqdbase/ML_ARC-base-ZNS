#!/bin/bash

FEMU=/work/FEMU/build-femu/qemu-system-x86_64
VM=/work/images/u20s.qcow2
ZNS=/work/images/zns.raw

if [ ! -f "$ZNS" ]; then
    dd if=/dev/zero of=$ZNS bs=1M count=32768
fi

echo "Starting FEMU with standard QEMU ZNS configuration..."

$FEMU \
    -name "femu-zns" \
    -enable-kvm \
    -cpu host \
    -smp 4 \
    -m 4G \
    -nographic \
    -drive file=$VM,if=virtio,format=qcow2 \
    -blockdev node-name=zns0,driver=file,filename=$ZNS \
    -device nvme,id=nvme0,serial=deadbeef,mdts=7 \
    -device nvme-ns,bus=nvme0,drive=zns0,nsid=1,zoned=true,zoned.zone_size=268435456,zoned.zone_capacity=268435456,zoned.max_active=14,zoned.max_open=14,uuid=5e40ec5f-eeb6-4317-bc5e-c919796a5f79 \
    -net user,hostfwd=tcp::8080-:22 \
    -net nic