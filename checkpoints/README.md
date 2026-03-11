# Model Checkpoints

Trained TRiX model weights from 500K-step runs across 3 seeds.

Format: `trix_s{seed}_step{step}.pt` (PyTorch state dict)

Sample checkpoints included here. Full set (153 files, ~210KB) available
on request.

To load:
```python
import torch
state = torch.load("checkpoints/trix_s0_step500000.pt")
```
