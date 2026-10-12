Subject: COMMSENG-26-0216-T: submission of the revised manuscript

Dear Dr Wang,

Thank you for your letter of 20 July 2026 and for the opportunity to revise our manuscript, "TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly", for Communications Engineering. We are grateful to you, Guest Editor Matteo De Marchi and the reviewers for their careful reading and constructive comments, which have substantially improved the manuscript, and we thank Reviewer 3 for the co-review contribution acknowledged in the decision letter. We are pleased to submit the revised manuscript together with a point-by-point response.

Following Reviewer 4's review of our code, the policy comparisons now come from recorded PPO and SAC experiments, replacing the benchmark stubs used for the submitted comparisons. The revision also provides more detailed task implementations and outcome definitions, and evaluates proposals from three vision-language models. The response to Reviewer 4 details these modifications, and we thank Reviewer 4 for strengthening the work.

Because the Methods are central to the study, the manuscript follows Introduction, Methods, Results, Discussion and Conclusions, an order the journal's formatting guidelines permit for Communications Engineering. The Introduction has no subsections, and its final paragraph summarizes the findings. Methods and Results use two levels of subheadings; Discussion and Conclusions have none. Figure legends are grouped at the end of the manuscript, and each figure is supplied as a separate file.

During revision we also added a frozen-policy study of the projection metric. With the admissible set fixed, changing only the metric that selects the nearest admissible command changes task outcomes for frozen policies while every forwarded command stays admissible. The response to Reviewer 4 (R4.2) and the Supplementary Information section "SCREW correction-metric sensitivity of frozen policies" report it in full.

The checksum-indexed evidence package and the current source snapshot are deposited in Zenodo (record 23031294, DOI 10.5281/zenodo.23031294). The record remains private during peer review and will be made public on publication. The confidential read-only reviewer link, supplied only in the correspondence sent to the editor, has been tested without account authentication, including representative file downloads.

The code is publicly available at https://github.com/stavanio/trix-disasm-suite (release v1.0).

The clean manuscript, marked manuscript, Supplementary Information and point-by-point response accompany the resubmission. Because the manuscript was restructured and rewritten throughout, the marked version highlights all of its text in blue as revised, places margin tags beside the passages that answer each comment, and closes with an index that links every comment to the change and its evidence; the marked Supplementary Information follows in the same file.

Thank you again for considering our work. We look forward to hearing from you.

Sincerely,
Stavan Dholakia (corresponding author), Shivani Shukla, Abhishek Singh and Aditya Gazta
