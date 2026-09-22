# Implementation freeze

Frozen 2026-08-05, before any learned run.

## Hashes

    environment S    d8dd709fd5a55436
    environment P    77b8eeda360dbd82
    criterion        1.1
    preregistration  9d91891
    amendment 1      49ea367

## Arms

Variant S: box_static, projection_static.
Variant P: box_blind_conservative, box_blind_permissive,
box_phase_aware, projection_phase_aware.

## Manipulation checks, measured

| check | result |
|---|---|
| filter divergence | >= 20% over 5000 proposals |
| probe visits both phases | S 155/188, P 887/222 rot/rel steps |
| demanding proposals | ~70% at or beyond the boundary |
| intervention | 74% to 99% by arm |
| executed occupancy, projection | 78.2% (S), 23.9% (P) |
| executed occupancy, box | 24.5% (S), 5.1% (P), reported |
| scripted feasibility | safe on both variants, 10 seeds |
| blind controls | conservative times out, permissive shears, 10 seeds |
| box area of ellipse | 2/pi, about 63.7% |

Nothing above may change. Learned runs follow.
