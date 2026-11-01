"""Visualization — reading intermediate representations without intervening.

Descriptive, not causal: these tools show you what the model is doing at a
given layer or position, but don't prove any of it is load-bearing. That's
what causal_interventions is for.

Built: logit_lens, attention_pattern, direct_logit_attribution.
Planned: tuned_lens, jacobian_lens — see
lens/mcp_server/src/tools/visualization/README.md.
"""
