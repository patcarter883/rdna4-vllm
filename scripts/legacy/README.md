# Legacy scripts (pre-combined-image)

These are **not used by the current build.** They date from the TheRock wheel-distribution
era, before the project consolidated onto the combined image (DIARY Act IX). Today vLLM,
RDNA4 attention, aiter, and flash-attention all come from the base image
(`tcclaviger/vllm22:dev`, pinned by digest), and the W4A8 kernel is compiled in-image from
`w4a8_fp8_wmma/` on every `docker compose build` — there are **no wheels to fetch or build**.

Kept for historical reference / the fully-from-source path only.

- `build-wheels.sh` — built the three ABI-locked wheels (vLLM/aiter/flash-attention) for
  `+rocm7.14` against the retired TheRock venv.
- `fetch-wheels.sh` — fetched those wheels from a GitHub Release.
