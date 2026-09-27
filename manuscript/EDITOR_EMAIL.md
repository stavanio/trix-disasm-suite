Subject: COMMSENG-26-0216-T: revised manuscript and title change

Dear Editor,

For manuscript COMMSENG-26-0216-T, submitted as "TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly", we have changed the title to "TRiX: Task-conditioned execution governance for robotic disassembly". The revision evaluates governance of recorded foundation-model proposals, but no autonomous agent loop was evaluated. The new title makes that scope explicit. The same declaration appears in the response to Reviewer 4.

We also thank Reviewer 3 for contributing through Communications Engineering's early-career peer-review initiative; the decision letter contains a co-review acknowledgement and no separate substantive requests.

The revised main manuscript and separate Supplementary Information report the policy experiments, actual VLM proposals, physical execution evidence and recorded implementation timing. The point-by-point response identifies the accompanying evidence and manuscript locations.

The exact-arm audit corrects Stage 1 BATTERY SAC/TRiX safe completion to 20.0%, superseding the intermediate draft's 59.8%. The audit is documented in `docs/battery_aggregation_correction.md`; its complete shard linkage is `verification/policy_record_linkage.json` in the evidence archive, under `exact_arm_group_means["stage1/BATTERY/sac/trix"]`. Original records remain retained. The separate matched preventive comparison remains 100.0% versus 100.0%.

The Supplementary Information subsection "Domain Randomisation" and Supplementary Table S4 now make the reset parameters, distributions, ranges and training/validation/test use explicit. "Prospective evaluation under shifted reset distributions" reports the prospectively declared evaluation of 42,000 episodes under widened reset distributions and an outside-support shell, with selected checkpoints and governors fixed. PRY is marked not applicable because its physical reset is deterministic. In the SNAP shell, safe completion is 59.9% under TRiX versus 3.5% under static in a comparison of separately trained policy-and-governor systems that does not isolate the governor's causal contribution; fractures occur in 401 versus 468 of 1,000 episodes per arm. The command and damage attribution in "Post-hoc SNAP command and damage attribution" is explicitly post hoc.

Figure 4 has been regenerated from the retained 80 seed-level values using a replacement plotting script supplied with the source; the historical plotting script remains unavailable.

Sincerely,
Stavan Dholakia, Shivani Shukla, Abhishek Singh and Aditya Gazta
