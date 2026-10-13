# Steering

Modifying model behavior at inference time by adding or ablating a direction in activation space.

## Built in `seesaw.interp.steering`
The computation lives in the library; Lens wrappers are not added yet.
- `steering_vector` — add a direction to one layer's residual stream and sweep its strength
- `feature_clamping` — set SAE features to a fixed value (0 = ablate) keeping the SAE error term
- ablation steering (removing a direction) lives in `seesaw.interp.model_editing.directional_ablation`

## Not yet implemented
- **affine_steering** — Steering with both a learned scale and offset, not just addition.
- **representation_control** — General framework for controlling model outputs via representation-space edits.
- **function_vectors** — Task-specific vectors extracted from in-context examples that trigger the same behavior when added elsewhere.
- **unsupervised_steering_vectors** — Deriving steering directions without labeled contrastive data.
- **multi_layer_steering** — Coordinating a steering intervention across multiple layers simultaneously.
