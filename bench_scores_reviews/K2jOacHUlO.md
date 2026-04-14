## Summary
This paper introduces **situated faithfulness** — the ability of LLMs to dynamically calibrate reliance on external context versus internal knowledge when the two conflict — and benchmarks it across multiple QA datasets, including a new dataset, **RedditQA**, containing human-written incorrect contexts sourced from Reddit posts. The authors propose two complementary approaches: Self-Guided Confidence Reasoning (SCR), where the model explicitly reasons about source trustworthiness, and Rule-Based Confidence Reasoning (RCR), which uses externally applied confidence-score heuristics. They further propose **Confidence Reasoning Direct Preference Optimization (CR-DPO)**, a fine-tuning method that substantially improves SCR on smaller models by learning from self-sampled paired reasoning traces.

---

## Strengths

- **RedditQA fills a genuine gap.** Existing conflict benchmarks (ClashEval, FaithEval) rely on synthetically perturbed contexts. RedditQA introduces human-authored, naturally occurring factual errors from social media, providing a more realistic test bed for knowledge-conflict robustness. This is a concrete contribution not replicated elsewhere.

- **CR-DPO self-sampling insight.** The finding that self-sampled reasoning paths outperform traces distilled from a stronger model (GPT-4o) is both counterintuitive and well-supported experimentally (Table 4, "-3 Tasks, SS" ablation). The mechanistic explanation — that confidence reasoning is grounded in the model's own knowledge, so importing another model's traces introduces hallucination — is convincing and broadly useful for the community.

- **Thorough failure analysis of RCR.** Section 5.3 goes beyond reporting numbers: it pinpoints three distinct failure modes of RCR (biased/flawed rules, miscalibrated confidence signals, rule-signal misalignment), supports each with targeted experiments (Table 3), and shows why even well-calibrated confidence scores can fail to maximize situated faithfulness. This is the kind of diagnostic analysis that advances understanding rather than just reporting results.

- **Practical model-tier insight.** The finding that strong-reasoner models (GPT-4o) benefit most from prompting-based SCR while smaller models (Llama-3-8B) require fine-tuning (CR-DPO) to match RCR provides actionable deployment guidance that is empirically grounded across two model families.

---

## Weaknesses

### Fatal
None.

### Major

- **CR-DPO is evaluated on a single model (Llama-3-8B).** This is the paper's primary methodological contribution, yet it is never tested on any other open-weight model (e.g., Llama-3-70B, Mistral-7B, Qwen). The core claim — "CR-DPO enhances SCR for smaller models" — rests on a single data point. Without at least one additional model, it is impossible to determine whether the gains are specific to this architecture/scale or reflect a generalizable training principle. This significantly limits confidence in the main contribution.

- **The RCR-vs-SCR size-dependent claim rests on a single small model.** The paper's practical finding that "RCR outperforms SCR for weaker models while SCR outperforms RCR for stronger models" is supported by three models, but the "weaker" side is represented solely by Llama-3-8B. An intermediate model (e.g., Llama-3-70B or Qwen-14B) is needed to establish whether this is a smooth function of model capability or specific to the 8B regime. This matters because the claim is presented as a general principle.

- **RedditQA dataset statistics are absent from the main text.** The construction pipeline is described (Section 3.2.1), but the main paper does not report the final dataset size, number of examples discarded, or annotator count. Given that RedditQA is a headline contribution, these omissions make it difficult to assess benchmark robustness or reproduce the dataset.

### Minor

- **GPT-4o is used in RedditQA construction and is also one of the primary evaluated models.** GPT-4o generates the multiple-choice questions and assesses claim inaccuracy, then is evaluated on the resulting dataset. While direct contamination is unlikely, this methodological asymmetry warrants at least a discussion: does GPT-4o's familiarity with its own output style or prior influence the benchmark's difficulty distribution for that model?

- **No oracle upper bound is reported.** The formulation defines a theoretical upper bound for SF (Section 3.1), but Table 1/2 do not show an oracle that always selects the correct source (internal vs. context). Without this, it is unclear whether the best reported results (e.g., CR-DPO at 62.4% for Llama-3-8B) are close to the achievable ceiling or still far from it. An oracle would let readers contextualize the 8.9% CR-DPO improvement.

- **OOD generalization boundary for CR-DPO is weak.** The model is trained on TriviaQA, NaturalQA, ConflictQA, and RedditQA, then called "OOD" on FreshQA and ClashEval. However, all these datasets are short-form world-knowledge QA of the same type. The OOD claim would be meaningfully stronger if tested on a different domain or question format (e.g., multi-hop, science QA, or domain-specific corpora).

- **The SF metric's equal weighting of Acc_t and Acc_f is unjustified.** In realistic RAG pipelines, correct contexts likely far outnumber incorrect ones. Equal weighting inflates the practical importance of Acc_f. The paper should at minimum acknowledge this design choice and its implications, or report sensitivity under different priors (e.g., 80% correct / 20% incorrect).

### Tiny

- **The pilot experiment justifying question-before-context ordering** (Section 4.1) is mentioned but not shown in the main paper. Since this ordering affects all SCR comparisons, a brief ablation in the main text would increase transparency.

- **Answer correctness operationalization is vague in the main text.** The paper says "e.g., an exact match with the ground truth" (Section 3.1) but the diverse datasets (RedditQA multiple-choice, FreshQA open-ended, etc.) likely use different matching procedures. This should be stated clearly in the main experimental setup rather than delegated to appendices.

---

## Nice-to-Haves

- A SCR failure analysis for GPT-4o, analogous to the Llama-3-8B case study in Figure 2, would provide a more balanced picture of where even the best method breaks down.

- An analysis varying the proportion of correct vs. incorrect contexts at inference (e.g., 80%/20% vs. 50%/50%) would directly assess whether the proposed methods remain practically useful under realistic RAG priors.

- A brief cost-benefit comparison (compute overhead of CR-DPO fine-tuning vs. gain relative to simply using GPT-4o-mini) would help practitioners decide when fine-tuning is warranted.

---

## Removed Points
*These points are flagged for removal — treat them with caution.*

- **The Acc_f upper bound is mathematically incorrect (Harsh Critic).** The critic argues that a false context could still contain useful incidental evidence. In the paper's setup, however, "false context" is defined as one whose entailed answer is incorrect — the bound is an intuitive claim about the setup's design, not a universal theorem. The paper's explanation is clear enough for its purpose.

- **Criticism of the TACS-LR approximation as underestimating TACS baseline.** The paper explicitly justifies using TACS-LR for consistency with proprietary models (Section 4.3). This is a reasonable design choice and noted transparently; the critique does not undermine the paper's conclusions.

- **"Situated faithfulness" is merely a renaming of existing concepts.** While neighboring concepts exist (robust context use, conflict-aware QA, selective retrieval reliance), the paper's unified framing, metric, and dual-class methodology provide a coherent contribution beyond simple renaming.

- **Terminology: "faithfulness" is counterintuitive when it means ignoring the context.** The qualifier "situated" is explained in the introduction and is sufficiently clear; this is a stylistic preference, not a substantive flaw.

- **The paper should compare with more models or a larger model zoo.** Three models spanning proprietary (GPT-4o, GPT-4o-mini) and open-weight (Llama-3-8B) is adequate for the paper's claims. Demanding more models without specific gaps is generic.

- **Pure formatting and style remarks from the harsh critic** (abstract phrasing, section-level writing quality) are removed per policy.

---

## Novel Insights

The most genuinely novel analytical finding in this paper — beyond the proposed contributions themselves — is that **self-sampled reasoning paths are superior to distilled paths from stronger models for confidence reasoning tasks**. Standard wisdom in LLM fine-tuning holds that stronger-model data helps smaller models; this paper shows a principled exception: when the task requires grounding reasoning in the model's own factual beliefs, importing an external model's reasoning traces causes the smaller model to follow reasoning chains that contradict its own knowledge, inducing hallucination rather than calibration. This insight has broader implications for when model distillation is and is not appropriate as a fine-tuning strategy.

---

## Suggestions

1. **Test CR-DPO on at least one additional open-weight model** (e.g., Llama-3-70B or Mistral-7B). This is the most important gap; without it, the main contribution is a single-model result.

2. **Report RedditQA size, discard rate, and annotator agreement in the main text** — even a single sentence with key statistics would substantially strengthen the dataset contribution's credibility.

3. **Add an oracle baseline** to Tables 1 and 2 to contextualize how close the best methods are to the achievable upper bound.

4. **Conduct a brief experiment with a model in the intermediate scale range** (e.g., 30B–70B) to validate the SCR-vs-RCR size-dependence claim as a general trend rather than an artifact of a single architecture.

5. **Address the GPT-4o construction bias concern** with a brief methodological note or, ideally, an inter-annotator agreement statistic from the human verification step.