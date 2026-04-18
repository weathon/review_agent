Now I have comprehensive calibration data. Let me synthesize the final review.

## Summary

ToolDial introduces a multi-turn dialogue dataset of 11,111 conversations for evaluating Tool-Augmented Language Models (TALMs). The dataset spans 16 user/system actions and an average of 8.95 turns per dialogue, generated via a four-step pipeline that constructs an API compatibility graph from RapidAPI, defines action sequences, generates scenario instructions, and uses GPT-4o for utterance generation. Evaluation shows modern LLMs struggle with action prediction and dialogue state tracking on this benchmark.

## Strengths

- **Addresses a genuine gap in TALM benchmarks.** Existing datasets like ToolBench (avg. 2 turns) and API-Bank (avg. 2.84 turns) feature short, simplistic dialogues, while ToolDial averages 8.95 turns with 16 distinct action types covering failure scenarios, clarification, and chained API calls. Table 1 clearly quantifies this gap. The focus on realistic multi-turn dynamics—where systems must request information, handle user failures, and chain APIs—is a real contribution.

- **Technically sound data generation pipeline.** The four-step pipeline (API graph → action sequences → scenario instructions → dialogue generation) is well-structured and systematic. The API graph construction (§3.1) with formalized edge construction criteria (Equation 1) and automated evaluation (MCC=0.868 on 200 human-labeled edges) adds rigor. The 23 action sequences derived from the action taxonomy (§3.2, Figure 3) represent a thoughtful adaptation of TOD concepts to tool-use settings.

- **Insightful evaluation results that reveal specific model weaknesses.** Table 5 shows that GPT models score near 0 on "Clarify" and "Suggest" actions and very low on "Request" (e.g., GPT-4o F1=13.7 on Request with GT), confirming these models rush to answer rather than gathering information—a practically important finding.

- **Demonstrates training utility.** TD-Llama (fine-tuned Llama3-8B on ToolDial) achieves dramatic improvements over its base model, showing the dataset is useful for instruction tuning. The overall dialogue accuracy of only 28.3% even for TD-Llama demonstrates substantial room for improvement.

## Weaknesses

### Fatal
None.

### Major

- **Faithfulness evaluation is circular and provides unreliable evidence.** The faithfulness metric (§4.1) uses G-Eval (itself a GPT-4o-mini-based evaluator) to judge whether model responses reflect API outputs. This creates a circular evaluation: GPT-4o generates the dialogues → model generates responses → GPT-4o-mini judges faithfulness. The >90% faithfulness scores for GPT models (Table 4) are essentially measuring GPT agreement with GPT, not genuine groundedness. This is not an auxiliary metric—it is one of the three core evaluation dimensions. There is no human validation or extraction-based verification of faithfulness. The paper's claim that "small language models are vulnerable to hallucination" (§4.3) based on a 6-point gap (88.4% vs ~96%) measured by an LLM judging LLM-generated text is not well-supported.

- **Ground truth for actions and dialogue states is synthetic and not systematically human-validated.** The 16-class action taxonomy and dialogue states are defined by template rules and GPT-4o simulation (§3.2–3.3), with only 100 dialogues (out of 11,111) evaluated by 3 annotators for naturalness/coherence/faithfulness on 1–3 scales—without inter-annotator agreement metrics. The DST evaluation uses strict exact match on API name, slot names, and string values (§4.1), which makes minor formatting differences indistinguishable from genuine errors. Most critically, the gold actions encode specific design choices (e.g., when GPT models choose "Clarify" where the dataset expects "Response fail," the paper acknowledges this "could be considered somewhat reasonable" but treats it as an error because it "violates the instruction provided in the prompt" §4.3). This means the benchmark evaluates compliance with a particular prompting policy rather than measuring genuine TALM capability.

- **Counterintuitive evaluation results that undermine confidence in the methodology.** TD-Llama achieves 91.0% action prediction accuracy in the "w/o GT" setting vs. 77.5% in the "w/ GT" setting (Table 4), meaning the model performs *better* when ground-truth information is removed from the prompt. This is deeply counterintuitive—providing more accurate prior context should not hurt performance. Looking at Table 5, the "Suggest" action drops from F1=99.9 (w/o GT) to F1=16.8 (w/ GT) for TD-Llama. This suggests the evaluation format or prompt design may have inconsistencies that affect reliability of the reported metrics. The paper does not explain or discuss this anomaly.

### Minor

- **Overclaiming "real-world" and "situation complexity" status.** The paper repeatedly states ToolDial "reflects real-world interactions" (Abstract, §1, Table 1). However, the dialogues are template-driven and GPT-generated, not derived from real user-system logs. The 23 action sequences are hand-crafted, and action frequencies are determined by design, not by empirical distribution. The paper should frame these as "simulated complex scenarios" rather than "real-world interactions."

- **API graph invalid edge filtering is under-specified.** The edge construction achieves only 70.9% precision (§3.1), meaning ~30% of constructed edges are invalid. The paper states "For dialogue generation, we discarded the invalid edges in the subsequent steps" but does not explain the filtering mechanism. Without this detail, it is unclear whether remaining edges were fully validated or whether some invalid chains persist in the final dataset.

- **Limited chaining depth.** Dialogues are restricted to pairs of APIs (§3.3), yet complex real-world tasks often require chains of 3+ APIs. This limits the evaluation of deeper compositional reasoning.

- **No cross-benchmark evaluation for TD-Llama.** The fine-tuned model is evaluated only on ToolDial's own tasks. Without testing on other TALM benchmarks (e.g., API-Bank, ToolBench), it is unclear whether ToolDial training generalizes or merely overfits to the dataset's specific schema and policy.

- **Human evaluation is insufficient for the claimed scale.** Three annotators evaluated only 100/11,111 dialogues (0.9%) on broad 1–3 scales without reporting inter-annotator agreement. This does not adequately validate the quality of the full dataset, particularly the structured annotations (dialogue states and action labels) used as ground truth.

### Trivial
- The "with GT" vs "without GT" naming convention (§4.2) is inverted from common usage—"with GT" means the model sees ground-truth prior labels, which is the *easier* setting, not the default.

## Nice-to-Haves

- Supplement faithfulness evaluation with an extraction-based metric (e.g., whether key API output values appear verbatim in the model's response) to provide an objective signal alongside G-Eval.
- Extend the API chaining to 3+ APIs to evaluate deeper multi-step reasoning.
- Evaluate TD-Llama on other TALM benchmarks to demonstrate cross-benchmark generalization.
- Report per-action-sequence results to identify whether difficulty is concentrated in specific interaction patterns.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"GPT self-contamination invalidates all evaluation results" (Harsh Critic #1):** The harsh critic argues the entire benchmark is contaminated because GPT-4o generated the data and GPT models are evaluated on it. While this is a concern for the faithfulness metric (which uses GPT as judge), DST and action prediction use *exact match* against structured labels—not GPT-based scoring. GPT models do *not* perform near-perfectly on these tasks (DST scores range 33–81%), which contradicts the claim of trivial self-alignment. The self-contamination concern is real for faithfulness (moved to Major) but is overstated for the structured evaluation tasks.

- **"Demand for human annotation or independent verification of all ground truth" (Harsh Critic #2, Spark #1):** While more human validation would strengthen the paper, demanding full human annotation of all 11,111 dialogues and all structured states is impractical for a dataset paper. Many accepted benchmarks (MultiWOZ, SGD) use similar synthetic generation pipelines with limited human validation. This is a nice-to-have improvement, not a fatal flaw.

- **"No cross-benchmark evaluation" (Spark #1):** The reviewer demands evaluation on other benchmarks as a top-3 missing experiment. However, the paper's main contribution is introducing a new benchmark with specific capabilities (multi-turn, chained APIs, failure scenarios) not present in other benchmarks—they cannot be fairly compared on metrics other benchmarks don't define. This is a nice-to-have for future work.

- **"Demand for human performance ceiling" (Spark #4):** For a TALM benchmark evaluation, human performance on structured prediction tasks (exact-match DST and action labels) is not standard practice. This is a reasonable suggestion but not a required experiment.

- **"Demand for more open-source models" (Human Finder #5):** The paper evaluates 7 models (4 GPT variants + 3 open-source). This is comparable to or more than many accepted benchmark papers. Adding more models is a nice-to-have.

- **"Action sequences not validated against real-world logs" (Harsh Critic #5, Human Finder #6):** The paper explicitly scopes itself as creating *designed* scenarios based on TOD literature. The 16 actions are drawn from established TOD taxonomies. While empirical validation against real logs would strengthen realism claims, demanding it goes beyond the paper's stated scope of creating a synthetic benchmark with controlled complexity.

- **"Strict exact-match metric for DST" (Harsh Critor §4):** While exact match has limitations, it is the standard metric for DST evaluation in the TOD community (MultiWOZ, SGD all use exact match). Requesting slot-level F1 or partial credit is a reasonable suggestion but not a methodological flaw.

- **"Unequal comparison between zero-shot GPT models and fine-tuned TD-Llama" (Harsh Critor §4):** The paper uses this comparison precisely to demonstrate the *training utility* of the dataset, not to rank intrinsic model capabilities. The paper states this explicitly ("fine-tuning on our dataset significantly improved the performance"). Treating this as an unfair comparison misreads the paper's intent.

## Novel Insights

The counterintuitive finding that TD-Llama performs better *without* ground-truth prior context (91.0% vs 77.5% for action prediction) is suggestive of an important design issue: when models are given detailed structured reasoning traces as context (the "w/ GT" setting), they may become over-reliant on the structured format and lose the ability to rely on natural dialogue context. This raises questions about how evaluation design choices—specifically, the format in which prior reasoning is presented—can substantially affect results in TALM benchmarks, sometimes in unexpected directions.

## Suggestions

- **Most critical:** Add an objective, extraction-based faithfulness metric (e.g., checking whether API output values appear in the response) alongside G-Eval to provide non-circular evidence for faithfulness claims.
- **Investigate and explain the w/o GT > w/ GT anomaly.** If adding ground-truth prior context hurts performance, it suggests a prompt design or formatting issue that should be illuminated.
- **Reframe claims:** Change "reflects real-world interactions" to "simulates realistic multi-turn scenarios" and "situation complexity" to "designed situation complexity" to avoid overclaiming.
- **Clarify the invalid edge filtering process** in §3.1 or the appendix—specify what mechanism was used and how many edges remained after filtering.

## Score and Decision

**Calibration comparison:**
- **τ-bench** (scores 6–8, accepted poster): Real-world domains, executable evaluation, database state comparison. More grounded evaluation but narrower scope.
- **MTU-Bench** (scores 5–6, accepted poster): Synthetic data, similar GPT-generation concerns, multi-granularity evaluation. Comparable structural weaknesses.
- **ToolTalk** (scores 3–6, rejected): Only 78 dialogues, limited models, small scope. ToolDial is substantially more comprehensive.
- **TaskBench** (scores 3–6, rejected): Similar LLM-generated benchmark with quality concerns. Data quality issues led to rejection.

ToolDial sits between ToolTalk/TaskBench (rejected, with quality concerns) and τ-bench/MTU-Bench (accepted, with similar but better-controlled evaluation). The dataset规模, action taxonomy, and multi-turn focus are genuine contributions. The API graph construction is a real methodological innovation. However, the circular faithfulness evaluation, synthetic ground truth without systematic validation, and the unexplained evaluation anomaly are significant weaknesses that undermine the reliability of the benchmark's core claims. The paper is above the level of ToolTalk/TaskBench in scope and ambition, but below τ-bench in evaluation rigor.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>