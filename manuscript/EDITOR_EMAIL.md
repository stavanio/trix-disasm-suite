Subject: COMMSENG-26-0216-T: revised manuscript, evidence correction and title change

Dear Editor,

The submitted headline policy comparisons came from random-action stubs with fixed transformations labeled as learning algorithms; presenting them as trained-policy evidence was incorrect. We withdraw those results and their policy-learning interpretation, replacing them with recorded PPO and SAC experiments, corrected task implementations and outcome definitions, and actual vision-language-model proposals. The response to Reviewer 4 details the defects and withdrawals; we thank the reviewer for identifying these problems.

We have changed the submitted title, "TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly", to "TRiX: Task-conditioned execution governance for robotic disassembly". No autonomous agent loop was evaluated; the revised title reflects the study of execution governance over recorded model proposals and learned-policy commands.

The manuscript follows Introduction, Methods, Results, Discussion and Conclusions, an order the journal's formatting guidelines permit for Communications Engineering. The Introduction has no subsections, and its final paragraph summarizes the findings. Methods and Results use two levels of subheadings; Discussion and Conclusions have none. Figure legends are grouped at the end of the manuscript, and each figure is supplied as a separate file.

During revision we also added a frozen-policy study of the correction metric. With the admissible set fixed, changing only the metric that selects the nearest admissible command changes task outcomes for frozen policies while every forwarded command stays admissible. The response to Reviewer 4 (R4.2) and Supplementary Section S9.5 report it in full.

The checksum-indexed evidence package and the current source snapshot are deposited in Zenodo (record 23031294, DOI 10.5281/zenodo.23031294). The record remains private during peer review and will be made public on publication. The confidential read-only reviewer link below has been tested without account authentication, including representative file downloads.

The confidential reviewer link is supplied only in the correspondence sent to the editor.

The code is publicly available at https://github.com/stavanio/trix-disasm-suite (release v1.0-resubmission).

This revision has been prepared within the twelve-week window following the decision of 20 July 2026, which ends on 12 October 2026. The clean manuscript, marked manuscript, Supplementary Information and point-by-point response accompany the resubmission. We also thank Reviewer 3 for the co-review contribution acknowledged in the decision letter; no separate substantive report was supplied.

Sincerely,
Stavan Dholakia (corresponding author), Shivani Shukla, Abhishek Singh and Aditya Gazta
