# Benchmarking Anomaly Detection for Large Language Model Alignment

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Many safety and alignment failures of large language models (LLMs) occur due to anomalous situations: unusual prompts or response patterns that are unforeseen by model developers. Anomaly detection is a promising tool to mitigate these failure modes caused by unknown unknowns; an anomaly detector monitoring a deployed LLM could shut it down or restrict user access in highly unusual situations. We introduce the first anomaly detection benchmark for LLM misalignment, MAAD (Mis-Alignment Anomaly Detection). Benchmarking detection of unforeseen alignment failures is difficult because LLMs are already trained on an extremely broad range of alignment data. Our key insight is that we can force certain known alignment failure modes to remain unseen by explicitly restricting the post-training data that anomaly detection methods can use within MAAD. For example, MAAD tests whether a detector can recognize deception about tool call results without any examples of such deception in the detector's post-training data. We use MAAD to evaluate a number of anomaly detection baselines, including prompting an LLM to ask if a conversation is unusual, measuring the perplexity of prompts and responses, and calculating the Mahalanobis distance of the internal representations of an LLM. We find that perplexity and Mahalanobis distance based detectors perform the best among these baselines, but no method performs at a high level across all failure modes. Our work motivates anomaly detection as an approach to LLM safety and provides a concrete benchmark to measure progress on this important problem. Code and data are available at https://anonymous.4open.science/r/reward-uncertainty-bench-4D66.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper propose to benchmarking the anomaly detection for LLM misalignment, specifically the  Benchmarking detection of unforeseen alignment failures. Thet  find that perplexity and Mahalanobis distance based detectors perform the best among these baselines, but no method performs at a high level across all failure modes.

### Strengths
1. Safety alignment is a very important question

2. Exploring which safety mechanisms are most effective among existing LLM alignment methods is a  timely research question.

3. The idea of adapting anomaly detection techniques from other domains to the context of LLM safety is interesting.

### Weaknesses
1.  Limited scope of anomaly detection in alignment.
The idea of using anomaly detection to safeguard LLMs is not yet a mainstream or widely adopted approach. Most prior works in this area use guard or moderation models directly for safety monitoring. The baselines used in this paper are primarily from non-LLM domains, which limits the significance of the benchmarking results. Since the proposed detector resembles guard models in nature, the research scope could be broadened to include a more direct comparison with established LLM alignment frameworks.

2. Unclear motivation compared to guard or moderation models.

Training an anomaly detector on preference datasets to assess input and output safety is conceptually similar to guard or moderation models (Llama Guard model), which are also trained to classify whether content is safe. The overall intuition and high-level purpose are essentially the same. Also, as the authors note, anomaly detectors tend to yield more false positives, which makes them appear less practical than guard models. Thus, it remains unclear why benchmarking anomaly detectors is necessary

3. Unfair model comparison.

In Table 2, different baselines are built upon different backbone models with varying sizes. Comparing them directly in one table may be misleading, as it is unclear whether the observed performance differences are due to the methodology or simply model capacity.

4. Limited model coverage.

As a benchmarking paper, the experiments would be more convincing if they covered a broader range of models and model sizes. Currently, the evaluation is restricted to Gamma, Llama-3.2, and a few smaller models (1B–8B), which limits the generality of the conclusions.

5. Limited insight or discovery.

A strong benchmark paper typically aims not only to compare methods but also to uncover meaningful patterns or insights that can guide future research. The current version focuses mainly on performance comparisons without offering deeper analysis, interpretations, or directions for follow-up work.

### Questions
Please see above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MAAD (Mis-Alignment Anomaly Detection), a benchmark for anomaly detection targeted at alignment failures in large language models (LLMs). The core claim is that many real safety failures happen in unusual, previously unseen situations where models behave in unsafe, deceptive, or manipulative ways. The paper frames these situations as “unknown unknowns” in alignment: jailbreak prompts that bypass refusals, deceptive tool-use, extreme sycophancy, etc. MAAD is designed to test whether anomaly detectors can flag such situations without having ever seen that specific failure mode in their post-training data.

The authors evaluate several baseline anomaly detectors: (a) prompting an instruction-tuned model to self-rate how unusual a conversation is, (b) perplexity of the conversation, (c) reward models and ensembles of reward models, (d) Mahalanobis distance in hidden representations, and (e) “pessimistic” combinations of a reward model with the other signals. The main empirical result is that Mahalanobis distance and perplexity work best on average (mean AUROC 0.83 and 0.76), but no method is reliable across all seven failure cases, and all tested methods fail badly on at least one case (AUROC <= 0.7). The paper argues that this shows anomaly detection can act as a safety monitor for alignment failures, but current methods are still not reliable enough.

### Strengths
1. Clear problem formulation that focuses on alignment anomalies (normative failures like deception, sycophancy, unsafe compliance), not just factual uncertainty or hallucination.
2. MAAD enforces a “restricted post-training data” regime per failure case, which operationalizes the “unknown unknowns” story instead of assuming access to labeled examples of every safety failure.
3. Practical impact: provides an immediate evaluation target for scalable oversight systems and future safety monitors, and shows that current monitors are still far from reliable (no baseline is >0.7 AUROC on all cases).

### Weaknesses
1. The benchmark’s main claim depends on the “no leakage” guarantee that the allowed training data for a failure case does not already contain that failure mode (or close paraphrases), but this is not quantitatively audited in the main text. Without a leakage audit, it is hard to verify that detectors are truly detecting unseen failures rather than memorizing near-duplicates.
2. The paper merges two operationally different settings under one score: (a) anomalous prompts causing unsafe behavior (e.g., jailbreaks), where you would want a pre-generation gate, and (b) anomalous responses to normal prompts (e.g., sycophancy, controlling behavior), where you would want a post-generation filter. Averaging AUROC across both settings hides different practical requirements.

### Questions
For sycophancy and controlling behavior: are the malign conversations mined directly from real model outputs (e.g., GPT-4o sycophancy (Sharma et al., 2025a), Bing-like persuasive behavior (Roose, 2023)), or were they partially hand-edited / synthesized? If there was editing, what quality control was used to avoid stylistic artifacts that a detector could trivially exploit?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors use anomaly detection as a safety net. In other words, they build methods that can monitor a deployed LLM (i.e., prompt‐response pairs) and detect when something looks anomalous (i.e., very unlike what was seen in training) and thus potentially misaligned, even if the exact failure mode was never seen in training. To this end, they introduced MAAD - a benchmark for evaluating anomaly detection methods for LLM alignment. After evaluation, they found that none of them uniformly succeed across all failure modes, but certain methods (like Mahalanobis distance and perplexity) do relatively well on average.

### Strengths
- The paper is complete

### Weaknesses
- The **challenge and motivation** for this project remain somewhat unclear to me. Specifically, I am not fully convinced by the claim that *“anomaly detection is the key tool for preventing safety failures of LLMs in unusual situations.”* The evaluation results show that existing anomaly detection methods perform poorly, so it is unclear how they can serve as the key tool for safety prevention. More insights and justification are needed to clarify this argument.

- As I understand it, the paper reframes existing *safe/unsafe* categorizations under the concept of *anomaly detection*. However, I find the necessity and novelty of this framing insufficiently explained.

- Moreover, the **writing quality** requires significant improvement. Many sentences are vague and difficult to interpret. For example:  
  > “Our key insight is that we can force certain known alignment failure modes to remain unseen by explicitly restricting the post-training data that anomaly detection methods can use within MAAD.”  
  This sentence is particularly unclear and would benefit from clearer wording and explanation.

- For **Sections 3.1.1 and 3.1.2**, it would be helpful to explain why these specific failure cases were chosen. Any underlying rationale or insights behind the selection would strengthen the paper.

- There exist **many more anomaly detection methods** than those used as baselines in the experiments—for instance, *Self-Inf* [1]. The choice of baselines should be better justified, especially in light of the diversity of existing approaches.


[1] Pruthi G, Liu F, Kale S, Sundararajan M. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems. 2020;33:19920-30.

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MAAD, a new benchmark for evaluating the ability of anomaly detection methods to identify unforeseen alignment failures in LLMs. The central premise is that as LLMs are deployed, they will encounter "unknown unknown" failure modes not covered by their safety training. Anomaly detection is proposed as a key tool to mitigate these risks. The core design of MAAD is to test whether detectors can identify known types of alignment failures (e.g., jailbreaks, sycophancy, tool-call deception) when they have been explicitly excluded from the detector's training data. This setup simulates the challenge of detecting novel failure modes. The benchmark comprises seven distinct failure cases, each with curated training and test sets. The authors evaluate several baseline anomaly detection methods on MAAD, including prompting-based approaches, perplexity, reward models, and Mahalanobis distance on internal representations. The results show that while perplexity and Mahalanobis distance-based methods perform best, no single method achieves high performance across all failure modes, motivating the need for further research in this area.

### Strengths
The paper addresses a critical and timely problem in LLM safety that how to guard against unforeseen alignment failures. The argument that anomaly detection can serve as a crucial safety net for "unknown unknowns" is compelling and relevant to the broader goal of deploying trustworthy AI systems.

### Weaknesses
1.The paper successfully demonstrates that existing anomaly detection methods perform poorly on the MAAD benchmark, but it stops short of providing a deep analysis of why they fail. For instance, why does Mahalanobis distance excel at detecting sycophancy but struggle with function-calling deception? A more detailed investigation into the relationship between failure modes and the shortcomings of specific detection methods is needed. Furthermore, the paper does not propose any new methods or concrete paths forward to address the identified gaps, limiting its impact.

2.The main body of the paper is severely lacking in experimental detail and results. It contains only one primary results table (Table 2), which presents a high-level summary of AUROC scores. Much of the crucial experimental detail, such as the analysis of scaling trends and more granular results, is relegated to the appendix. This makes it difficult for the reader to fully assess the experimental findings without constantly referencing supplementary materials. 

3.The primary contribution of the paper is the collection and reorganization of existing, known LLM failure datasets into a new benchmark structure. While this is a valuable engineering effort, the paper does not introduce any new types of alignment failures or provide fundamentally new insights into the nature of LLM misalignment itself. The work is more of a problem statement and dataset curation than a novel research discovery.

### Questions
1.The results show that no single baseline performs well across all failure modes. Does this suggest that a universal anomaly detector for LLM alignment is infeasible, and that future work should instead focus on ensembles of specialized detectors? What are the authors' thoughts on the path forward, beyond simply stating that "further work is needed"?

2.The Mahalanobis distance method shows surprisingly poor scaling, with performance degrading as model size increases. The paper hypothesizes that this is due to noise from higher-dimensional representations. Have the authors considered or tested dimensionality reduction techniques (e.g., PCA) on the hidden states before calculating the distance to see if this mitigates the issue?

3.The paper's premise is that anomaly detectors can identify "unforeseen" failures. However, the benchmark is constructed from known failure types. How confident are the authors that strong performance on MAAD would translate to effective detection of truly novel, undiscovered failure modes in the wild? Is there a risk that methods developed to succeed on MAAD might simply be better at generalizing across known categories of failure, rather than detecting genuinely new patterns?

### Soundness
1

### Presentation
2

### Contribution
1
