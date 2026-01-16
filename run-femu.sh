#!/bin/bash
# FEMU ZNS Mode with ARC Cache Integration

FEMU_DIR=~/FEMU
IMGDIR=~/images
OSIMGF=$IMGDIR/u20s.qcow2
ZNSIMGF=$IMGDIR/zns.raw
FEMU_BIN=$FEMU_DIR/build-femu/qemu-system-x86_64

# 백엔드 이미지 확인
if [[ ! -e "$ZNSIMGF" ]]; then
    echo "Creating ZNS backend image: $ZNSIMGF"
    dd if=/dev/zero of=$ZNSIMGF bs=1M count=16384
else
    echo "Using existing ZNS image: $ZNSIMGF"
fi

echo "=== FEMU ARC-ZNS Starting ==="
echo "OS Image: $OSIMGF"
echo "ZNS Image: $ZNSIMGF"
echo "=============================="

# FEMU 실행 (수정됨)
sudo $FEMU_BIN \
    -name "femu-arc-zns-vm" \
    -enable-kvm \
    -cpu host \
    -smp 4 \
    -m 8G \
    -device virtio-scsi-pci,id=scsi0 \
    -device scsi-hd,drive=hd0 \
    -drive file=$OSIMGF,if=none,aio=threads,cache=none,format=qcow2,id=hd0 \
    -device femu,devsz_mb=4096,femu_mode=3,id=nvme0 \
    -drive file=$ZNSIMGF,id=zns0,format=raw,if=none \
    -net user,hostfwd=tcp::2222-:22 \
    -net nic,model=virtio \
    -nographic \
    -qmp unix:./qmp-sock,server,nowait 2>&1 | tee femu-arc-zns.log
