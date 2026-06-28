# POWER SAFETY — RDNA4 (gfx1201) WMMA prefill power overshoot (MANDATORY before serving the default image)

`vllm22-w4a8:combined` now defaults the native HIP serve path **on** (`VLLM_GDN_HIP=1`,
`VLLM_ATTN_DECODE_HIP=1`). Two of those paths — the **GDN WMMA chunked prefill** and the **attention
paged/chunked prefill** — are rocWMMA matrix-core kernels that *saturate* the matrix engine. The same
saturation that makes them fast also makes them draw peak power hard.

## The problem (observed)

On gfx1201 the WMMA prefill ramps power **faster than the RDNA4 power controller can damp the di/dt**.
A brief transient overshoot of **~500 W against a 374 W cap** was observed during a prefill A/B —
**and that run already had a −200 MHz GPU clock offset applied**, so −200 MHz was *not* enough; the
offset has to be aggressive (the box was subsequently moved to **−500 MHz**). This is a hardware
characteristic of RDNA4: unlike **RDNA3.5**, which exposes a **power-ramp-speed** control, RDNA4 has
no ramp-rate limiter, so the controller can only react *after* the spike.

> **A power cap ALONE does not fix this.** The cap bounds the *steady-state* board power, not the fast
> transient — the overshoot happens before the cap engages. You **must also apply a negative GPU clock
> offset**, which lowers the peak clock the kernel can reach and therefore the peak di/dt. *Without the
> clock offset the power limit is effectively meaningless for this workload.*

## Required mitigation (apply on the host, per card, before any serve)

Both levers together. Values below are **starting points — tune to your card** while watching the live
draw (you have the telemetry); the goal is "no transient above the rated cap during a long-prompt
prefill." Run for each compute card (gfx1201 = cards 0 and 1; the iGPU/card 2 is never a target).

```bash
# 1) Power cap (steady-state bound). ROCm 7.2.1:
rocm-smi -d 0 --setpoweroverdrive 304     # watts; set at/below the card's rated TBP (NOT 374)
rocm-smi -d 1 --setpoweroverdrive 304

# 2) Negative GPU (SCLK) clock offset — THE CRITICAL LEVER. Enable manual OverDrive, then apply a
#    negative offset so the WMMA kernel can't spike to peak clock. Exact interface varies by ROCm /
#    kernel; use whichever your build exposes and VERIFY it took:
#    Observed on this box: -200 MHz was INSUFFICIENT (still spiked to ~500W); -500 MHz is in use.
#    Start aggressive (-300 to -500 MHz) and relax only if headroom allows — do NOT start near -100.
#  a) rocm-smi (if your build has the offset verb):
rocm-smi -d 0 --setclkoffset -500         # MHz, NEGATIVE; -200 was not enough on this card
#  b) sysfs OverDrive fallback (pp_od_clk_voltage), if rocm-smi lacks the verb:
echo manual > /sys/class/drm/card0/device/power_dpm_force_performance_level
# inspect the OD table + its units/range first:
cat /sys/class/drm/card0/device/pp_od_clk_voltage
# then write a negative GFXCLK offset per that table's syntax, e.g. (verify the token on your kernel):
#   echo 'sclk_offset -500' > /sys/class/drm/card0/device/pp_od_clk_voltage   # MHz, negative
echo c > /sys/class/drm/card0/device/pp_od_clk_voltage    # commit
```

## Verify it worked (do this — don't assume)

Watch the draw during a real long-prompt prefill (the workload that triggered it):

```bash
# in one shell, sample fast while a prefill runs in another:
while :; do rocm-smi -d 0 --showpower --showgpuclocks | grep -iE 'power|sclk'; sleep 0.2; done
```

Pass = no transient above the rated cap during sustained prefill, and the WMMA speedup is still net
positive. If you still see overshoot, increase the negative clock offset (more negative) before
touching the power cap.

## Escape hatch (no host tuning available)

If you can't apply the host power/clock settings on a given box, fall back to the low-power paths at
runtime — no rebuild needed:

```bash
VLLM_GDN_HIP_RECURRENT_ONLY=1   # GDN: scalar recurrent prefill (low matrix-core occupancy, low power)
VLLM_ATTN_DECODE_HIP=0          # attention: stock triton_attn (no rocWMMA paged-prefill)
# VLLM_GDN_HIP stays 1 either way — the recurrent GDN path still kills the 15-30min Triton JIT cliff.
```

## Scope

Applies to any image with the WMMA prefill paths engaged — i.e. the default `:combined`, and any
quantized serve (the **W4A8 fp8 MoE** kernel is *literal* fp8 WMMA and pegs power even harder on the
35B/27B). The decode and scalar-recurrent paths are low-occupancy and not affected.
