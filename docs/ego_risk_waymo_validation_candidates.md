# Ego-Risk Waymo Validation Candidates

Date: 2026-06-11 CST

This note records the first real Waymo validation screening for generic ego-risk causal guidance. The sweep used the validation raw directory `/home/anthony/SimAgentJEPA/data/waymo/validation`, checkpoint `checkpoints/causal_diffusion/epoch=00.ckpt`, `target_spec=ego_risk`, `target_event_eta=0.0`, target window `1,8`, `ego_interaction_alpha=4.0`, `near_miss_distance=8.0`, and `ttc_threshold=3.0`.

## Sweep Outputs

- First 100-scene numeric sweep records:
  `outputs/causal_ego_risk_waymo_val_sweep_chunks/records_000_099.csv`
- Ranked candidates:
  `outputs/causal_ego_risk_waymo_val_sweep_chunks/ranked_candidates_000_099.csv`
- First visual batch:
  `outputs/causal_ego_risk_waymo_val_visual_top6/`
- Second visual batch:
  `outputs/causal_ego_risk_waymo_val_visual_top4_extra/`

## Recommended Visual Candidates

| Rank | Scene index | Scenario | Figure | Why it is useful |
|---:|---:|---|---|---|
| 1 | 72 | `b3571d47ebd0f177` | `outputs/causal_ego_risk_waymo_val_visual_top4_extra/idx_00072_b3571d47ebd0f177_guidance_modes.png` | Clear target shift into the ego/SDC future corridor; nonzero ego-risk and near-miss signal; no hard collision/offroad. |
| 2 | 87 | `e0413edd0d2fabb6` | `outputs/causal_ego_risk_waymo_val_visual_top4_extra/idx_00087_e0413edd0d2fabb6_guidance_modes.png` | Visually clear target trajectory alignment with the ego path; small target distance and nonzero near-miss/risk signal. |
| 3 | 31 | `e7a82fd0743a5093` | `outputs/causal_ego_risk_waymo_val_visual_top6/idx_00031_e7a82fd0743a5093_guidance_modes.png` | Clear trajectory edit near the ego lane with low TTC signal; no hard collision/offroad and small token change. |
| 4 | 17 | `cc416279a39e9123` | `outputs/causal_ego_risk_waymo_val_visual_top6/idx_00017_cc416279a39e9123_guidance_modes.png` | Strong low-TTC/risk signal with a visually readable lateral/parallel target edit near the ego future. |
| 5 | 41 | `ebc2a6590cff7cb6` | `outputs/causal_ego_risk_waymo_val_visual_top6/idx_00041_ebc2a6590cff7cb6_guidance_modes.png` | Very close target-ego geometry and strong near-miss signal; useful as a near-miss example even though risk success is zero. |
| 6 | 33 | `14ea3dc438a415f6` | `outputs/causal_ego_risk_waymo_val_visual_top6/idx_00033_14ea3dc438a415f6_guidance_modes.png` | Backup candidate with nonzero risk and near-miss, but the plot is horizontally compressed and dynamics energy is higher. |

## Metric Snapshot

| Scene | risk | near | min distance | min TTC | collision | offroad | token change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 72 | 0.2083 | 0.3229 | 3.02 m | 78.07 s | 0.0 | 0.0 | 0.0072 |
| 87 | 0.1563 | 0.3229 | 2.62 m | 67.99 s | 0.0 | 0.0 | 0.0199 |
| 31 | 0.1771 | 0.0208 | 8.40 m | 1.83 s | 0.0 | 0.0 | 0.0062 |
| 17 | 0.3177 | 0.0000 | 10.28 m | 1.32 s | 0.0 | 0.0 | 0.0089 |
| 41 | 0.0000 | 0.5000 | 2.18 m | 7.68 s | 0.0 | 0.0 | 0.0064 |
| 33 | 0.1458 | 0.1719 | 6.89 m | 6.36 s | 0.0 | 0.0 | 0.0089 |

## Rejected Despite High Metrics

- Scene 20 (`8c20c8003aa1b842`) has the strongest risk score in the first 100 scenes, but the ego-risk edit is visually too subtle in the current plot.
- Scene 22 (`a9a65a8071f1a441`) has strong near-miss signal, but it reads more like a static close-distance case than a clear edited trajectory.
- Scenes 61 and 52 have strong numeric scores, but the rendered edits are too overlapped/static to be strong paper figures.

## Notes

- This is a first 100-scene screen, not a full validation-set search.
- The current default target selection still often picks low-index agents. A better next step is scene-specific target selection by ego proximity, path intrusion, and future crossing potential before running wider sweeps.
