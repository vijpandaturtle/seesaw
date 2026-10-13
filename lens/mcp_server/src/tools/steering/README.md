# Steering

Modifying model behavior at inference time by adding or ablating a direction in activation space.

## Not yet implemented
- **addition_steering** — Add a direction vector to the residual stream to push behavior toward a target concept.
- **ablation_steering** — Remove a direction rather than add one, to suppress a behavior.
- **affine_steering** — Steering with both a learned scale and offset, not just addition.
- **representation_control** — General framework for controlling model outputs via representation-space edits.
- **function_vectors** — Task-specific vectors extracted from in-context examples that trigger the same behavior when added elsewhere.
- **unsupervised_steering_vectors** — Deriving steering directions without labeled contrastive data.
- **multi_layer_steering** — Coordinating a steering intervention across multiple layers simultaneously.
