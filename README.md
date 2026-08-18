---
title: Kreuzkopplung
emoji: 〰️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
short_description: Adaptive two-channel DualEntangledSystem.run()
---

# Kreuzkopplung

Adaptiver, klassischer Zwei-Kanal-Regler. Keine Quantenmechanik — ein Mix-/Balance-Kreis mit Kreuzkopplungsmatrix.

`DualEntangledSystem.run(inputs)` war im Originalfragment nur die Signatur. Hier ist sie vollständig: jeder Skalar wird ein `step`, Gewichte und Kopplung bleiben über den Batch erhalten.

## Install

```bash
pip install -r requirements.txt
python dual_entangled.py --steps 64
python app.py
```

## `run`

```python
from dual_entangled import DualEntangledSystem
import numpy as np

sys = DualEntangledSystem()
t = np.linspace(0, 4 * np.pi, 240)
outputs, telemetry = sys.run(np.sin(t))
# outputs.shape == (240, 2)
```

`reset=False` setzt eine vorherige Serie fort.

## Hugging Face Space

Dieses Repo ist eine Gradio-Space-Quelle.

1. [New Space](https://huggingface.co/new-space)
2. **Import from GitHub** → `SpaceBum9/kreuzkopplung`
3. SDK: Gradio, `app_file`: `app.py`

Danach läuft dieselbe `run()`-Schleife unter `huggingface.co/spaces/SpaceBum9/kreuzkopplung`.

## GitHub

[SpaceBum9/kreuzkopplung](https://github.com/SpaceBum9/kreuzkopplung)

## Lizenz

MIT
