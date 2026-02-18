#!/bin/bash

# System configuration for low-latency trading
# Run as root

echo "Configuring system for low-latency operation..."

# 1. Isolate CPU cores
echo "Isolating cores 0-3..."
# Add to /etc/default/grub:
# GRUB_CMDLINE_LINUX="isolcpus=0-3 nohz_full=0-3 rcu_nocbs=0-3"
# Then run: update-grub && reboot

# 2. Disable CPU frequency scaling
echo "Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null || true
done

# 3. Disable turbo boost (for deterministic latency)
echo "Disabling turbo boost..."
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

# 4. Enable huge pages
echo "Enabling huge pages..."
echo 1024 > /proc/sys/vm/nr_hugepages

# 5. Increase socket buffer sizes
echo "Increasing socket buffer sizes..."
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.core.rmem_default=16777216
sysctl -w net.core.wmem_default=16777216

# 6. Increase network queue length
echo "Increasing network queue length..."
sysctl -w net.core.netdev_max_backlog=5000

# 7. Disable IRQ balance (pin IRQs manually)
echo "Disabling IRQ balance..."
systemctl stop irqbalance
systemctl disable irqbalance

# 8. Set real-time limits
echo "Setting real-time limits..."
cat >> /etc/security/limits.conf << EOF
*    soft    rtprio    99
*    hard    rtprio    99
*    soft    memlock   unlimited
*    hard    memlock   unlimited
EOF

echo ""
echo "System configuration complete!"
echo ""
echo "IMPORTANT: For CPU isolation, edit /etc/default/grub and reboot"
echo "IMPORTANT: Pin NIC IRQs to non-isolated cores manually"
echo ""
echo "To pin NIC IRQs:"
echo "  1. Find NIC IRQ: cat /proc/interrupts | grep eth0"
echo "  2. Pin to core 4: echo 10 > /proc/irq/<IRQ>/smp_affinity"
