# Brief for the agent working here

This repository holds the code and experiments for PIMDA, a DAC 2027 paper by Tomas (GitHub `tospe`). The paper itself lives in a separate repository.

**Read `build-plan.md` first.** It decides what to do and in what order. Where anything else in this repository disagrees with it, the plan wins.

## Layout

| Folder | What it holds | Rule |
|---|---|---|
| `data/` | Inputs: traces, measurements, the workload description, the oracle | Never edit |
| `old/` | The previous version of this work | Read-only reference; see `build-plan.md` §2 |
| `library/` | The numbers and options every result rests on | Built in step B1 |
| everything else | The new build | Yours |

## How Tomas works

- **Build only on what exists.** Do not invent models, formulas or numbers. When you must assume something, label it and name what will replace it.
- **Keep working, and respect the stop points.** `build-plan.md` §4 says what each stop blocks. Do everything else without waiting.
- **Keep text short and precise.** Every symbol has one definition; do not restate it.
- **Keep one owner per variable.** Software moves the point, hardware moves the roof, capacity moves the wall.
- **Do not commit or push** unless Tomas asks.
- **Leave raw data alone.** Derive every table and figure from it with a script.
- **Figures:** matplotlib on a white background. Palette: compute `#2a78d6`, transfer `#eb6834`, programming `#1baf7a`, wall `#4a3aa7`.
- **Keep the paper anonymous.** It is double-blind. VIPER (arXiv 2608.23404) is the authors' own work and is cited in the third person. Keep names out of anything that goes into the paper, including figure text and file metadata.
