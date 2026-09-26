Subject: COMMSENG-26-0216-T: revised manuscript and title change

Dear Editor,

For manuscript COMMSENG-26-0216-T, submitted as "TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly", we have changed the title to "TRiX: Task-conditioned execution governance for robotic disassembly". The revision evaluates governance of recorded foundation-model proposals, but no autonomous agent loop was evaluated. The new title makes that scope explicit. The same declaration appears in the response to Reviewer 4.

The revised main manuscript and separate Supplementary Information report the policy experiments, actual VLM proposals, physical execution evidence and recorded implementation timing. The point-by-point response identifies the accompanying evidence and manuscript locations.

The exact-arm audit corrects Stage 1 BATTERY SAC/TRiX safe completion to 20.0%, superseding the intermediate draft's 59.8%. The audit is documented in `docs/battery_aggregation_correction.md`; its complete shard linkage is `verification/policy_record_linkage.json` in the evidence archive, under `exact_arm_group_means["stage1/BATTERY/sac/trix"]`. Original records remain retained. The separate matched preventive comparison remains 100.0% versus 100.0%.

Supplementary Section S4.6 and Table S4 now make the reset parameters, distributions, ranges and training/validation/test use explicit, including the absence of a widened OOD reset evaluation. Figure 4 has been regenerated from the retained 80 seed-level values using a replacement plotting script supplied with the source; the historical plotting script remains unavailable.

Sincerely,
Stavan Dholakia, Shivani Shukla, Abhishek Singh and Aditya Gazta
