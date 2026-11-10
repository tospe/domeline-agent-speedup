#!/bin/sh
# Download the source PDFs that have a public URL, and check them against SOURCES.yaml.
set -e
cd "$(dirname "$0")"
get() {  # get <file> <url> <sha256>
  [ -f "$1" ] || { curl -sL -A "Mozilla/5.0" -o "$1" "$2"; sleep 3; }
  echo "$3  $1" | shasum -a 256 -c -
}
get star_softmax_arxiv2401.17582.pdf https://arxiv.org/pdf/2401.17582v1 080a78e1b1526ee2aec0bba552f640711b7f2a28111b133d62e52e42d2ed3e0c
get puma_arxiv1901.10351.pdf https://arxiv.org/pdf/1901.10351v2 f7e14319e551b8c68f4a746fa1573550beae902c064faa193c08f1b0614b1e9e
get viper_arxiv2608.23404.pdf https://arxiv.org/pdf/2608.23404v1 be5e9043349bae9b7281c7a897fa96c9ac996ae28d790f0667bdd070ebc5b9f4
get softermax_arxiv2103.09301.pdf https://arxiv.org/pdf/2103.09301v1 c11c9bbba96e3a461b49b5682539b2138483326b48ffff7748a5cac30b483b9e
get ita_arxiv2307.03493.pdf https://arxiv.org/pdf/2307.03493v2 7db8354dbf30c6961a8e7450c06e2e5890c61cbf3ecd036fd19b0237d050caad
get sole_arxiv2510.17189.pdf https://arxiv.org/pdf/2510.17189v1 acded6f06e95502844410b73f8fd98cede7afbe666d088d7f77ea79c86c2c156
get hyft_arxiv2311.13290.pdf https://arxiv.org/pdf/2311.13290v2 ee1d632ba73c03f5d0f170beff09ca1790b749794776ad51852ddd888042bd0f
get consmax_arxiv2402.10930.pdf https://arxiv.org/pdf/2402.10930v3 2f0e67414358313e940e7b7820c1e38d5e8215efd8f9b75c6c319c76256e98ac
get softmax_layernorm_edge_arxiv2604.23647.pdf https://arxiv.org/pdf/2604.23647v1 1468c7f0362e47e9214b69b7a75bc1586d124540bafbbbbcd580262aa4472520
get cimple_arxiv2604.15944.pdf https://arxiv.org/pdf/2604.15944v2 6656d37a27c7a155c85afe42294da762cd18a6be75cb8d4bf2838f387dfe5261
get softmap_arxiv2411.17847.pdf https://arxiv.org/pdf/2411.17847v1 3e24e0544ac0b47d61a832067fc22d7e1dd628481d977ab198234d2d2abb083d
get hastily_arxiv2502.12344.pdf https://arxiv.org/pdf/2502.12344v1 f7f3363acbf60e8ee87fc62579b63d3066cc2f3bc88af99026e8e5c5894680c0
get topkima_former_arxiv2411.13050.pdf https://arxiv.org/pdf/2411.13050v1 7021b152a60d6b728ddc33df50c435352cf11fae47ad543f84d7cd5327c39d20
get softmax_gelu_reuse_arxiv2402.10118.pdf https://arxiv.org/pdf/2402.10118v2 23e876c13594d43aa4b0b59f1706aad67666cb77e4f9b5a6b7888de57dcfcece
