# dscnn-16center-v1 sim-core Results

## yes sample 1 - (2026-05-08)

Command context:

```bash
make sim-core \
  KWS_MANIFEST_JSON=src/rtl/dscnn/kws_top/spectrograms/test_vectors.json \
  KWS_KEYWORD=yes \
  KWS_SAMPLE_INDEX=1
```

Selected sample:

- Sample index: `1`
- WAV: `0717b9f6_nohash_2.wav`
- Hex: `spectrograms/spectrogram_1.hex`
- Dataset label: `yes (6)`
- Manifest arithmetic: `yes (6)`
- Manifest PyTorch: `yes (6)`
- RTL class: `yes (6)`
- Result: `PASS`

RTL GAP scores:

| Class | Index | Score |
| --- | ---: | ---: |
| no | 0 | 466 |
| off | 1 | -319 |
| on | 2 | -1441 |
| silence | 3 | -2168 |
| unknown | 4 | 1016 |
| wow | 5 | -989 |
| yes | 6 | 1971 |

Ranking:

```text
yes(6)=1971 > unknown(4)=1016 > no(0)=466 > off(1)=-319 > wow(5)=-989 > on(2)=-1441 > silence(3)=-2168
```

- Margin: `955`
- Final progress sample: `since_start=2,600,000`, `approx_cyc=3,311,390`, `layer=9`
- Sim time: `145385400.00 ns`
- Real time: `3636.49 s`
- Ratio: `39979.61 ns/s`
- Regression: `TESTS=1 PASS=1 FAIL=0 SKIP=0`

## no sample 7 - (2026-05-09)

Command context:

```bash
make sim-core \
  KWS_MANIFEST_JSON=src/rtl/dscnn/kws_top/spectrograms/test_vectors.json \
  KWS_KEYWORD=no \
  KWS_SAMPLE_INDEX=7
```

Selected sample:

- Sample index: `7`
- WAV: `2039b9c1_nohash_1.wav`
- Hex: `spectrograms/spectrogram_7.hex`
- Dataset label: `no (0)`
- Manifest arithmetic: `no (0)`
- Manifest PyTorch: `no (0)`
- RTL class: `no (0)`
- Result: `PASS`

RTL GAP scores:

| Class | Index | Score |
| --- | ---: | ---: |
| no | 0 | 4995 |
| off | 1 | -2455 |
| on | 2 | -2790 |
| silence | 3 | -3880 |
| unknown | 4 | 1422 |
| wow | 5 | 984 |
| yes | 6 | -356 |

Ranking:

```text
no(0)=4995 > unknown(4)=1422 > wow(5)=984 > yes(6)=-356 > off(1)=-2455 > on(2)=-2790 > silence(3)=-3880
```

- Margin: `3573`
- Final progress sample: `since_start=7,300,000`, `approx_cyc=8,011,390`, `layer=9`
- Sim time: `346985400.00 ns`
- Real time: `not captured`
- Ratio: `not captured`
- Regression: `test_chip_core_e2e PASSED`

## wow sample 0 - (2026-05-09)

Command context:

```bash
make sim-core \
  KWS_MANIFEST_JSON=src/rtl/dscnn/kws_top/spectrograms/test_vectors.json \
  KWS_KEYWORD=wow \
  KWS_SAMPLE_INDEX=0 \
  KWS_SAMPLE_MATCH=wow
```

Selected sample:

- Sample index: `0`
- WAV: `0a7c2a8d_nohash_3.wav`
- Hex: `spectrograms/spectrogram_0.hex`
- Dataset label: `wow (5)`
- Manifest arithmetic: `wow (5)`
- Manifest PyTorch: `wow (5)`
- RTL class: `wow (5)`
- Result: `PASS`

RTL GAP scores:

| Class | Index | Score |
| --- | ---: | ---: |
| no | 0 | 786 |
| off | 1 | -1979 |
| on | 2 | -1474 |
| silence | 3 | -1503 |
| unknown | 4 | 452 |
| wow | 5 | 2419 |
| yes | 6 | -609 |

Ranking:

```text
wow(5)=2419 > no(0)=786 > unknown(4)=452 > yes(6)=-609 > on(2)=-1474 > silence(3)=-1503 > off(1)=-1979
```

- Margin: `1633`
- Final progress sample: `since_start=2,600,000`, `approx_cyc=3,311,390`, `layer=9`
- Sim time: `145385400.00 ns`
- Real time: `4619.37 s`
- Ratio: `31473.01 ns/s`
- Regression: `TESTS=1 PASS=1 FAIL=0 SKIP=0`
