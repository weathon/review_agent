# You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs

- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
Large language models (LLMs) are increasingly deployed in specialized domains such as finance, medicine, and agriculture, where they face significant distribution shifts from their training data. Domain-specific fine-tuning can mitigate this challenge but relies on high-quality labeled data that is expensive and slow to collect in expertise-limited settings. We study label-free test-time adaptation for language models and present SyTTA, an inference-time framework that adapts models on-the-fly without additional supervision. SyTTA couples two complementary uncertainty signals that arise under distribution shift: input-side perplexity, indicating mismatch with domain-specific terminology and patterns, and output-side predictive entropy, indicating diffuse and unstable token probabilities during generation. Unlike prior test-time approaches for LLMs that optimize a single signal, SyTTA integrates both within a unified self-supervised objective that automatically balances their influence, stabilizing generation while improving domain awareness. Across diverse model architectures and domain-specific benchmarks, SyTTA delivers consistent gains. Notably, on agricultural question answering, SyTTA improves ROUGE-Lsum by over 120% on Qwen-2.5-7B with only 4 extra tokens per query. These results show that effective test-time adaptation for language models is achievable without labeled examples, supporting deployment in label-scarce domains. The code will be made available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work is in line with test-time adaptation of LLMs. It proposes a novel label-free test-time adaptation framework (SYTTA) for LLMs. It couples two complementary uncertainty signals from different positions: input-side perplexity and output-side predictive entropy. Existing works typically take only one signal into account. A novel strategy, dynamic importance weighting, is applied to incorporate these two signals effectively. The experimental results demonstrate that the proposed method significantly outperforms strong baselines, verifying its effectiveness.

### Strengths
- The writing and structure are excellent.
- The proposed framework is novel and interesting, especially the dynamic weighting part.
- Two different signals are well exploited via the dynamic weighting.
- Potential collapse and distribution drift are also considered; to avoid such issues, KL is applied.
- The experiments are comprehensive; a concise case study would further clarify the improvement over baselines.
- Strong and consistent gains over strong baselines across the experiments.

### Weaknesses
- From my point of view, no major weakness.
- The discussion of prefix token length could be deepened. More detailed analysis of the generated prefix tokens and other perspectives (e.g., attention sink) would be helpful and interesting.
- It would be beneficial to also compare the computational overhead and latency.

### Questions
**Questions and Suggestions**

- I am curious what the exact generated prefix tokens are. I would appreciate a discussion regarding the generated prefix tokens to analyse their contributions from a semantic perspective. 
- Does attention sink also affect the prefix token length setting?
- It would be better to have a concise case study section to compare different models straightforwardly.


Lastly, I would be happy to increase the rating if my questions are addressed.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes SYTTA, a test-time adaptation framework that jointly (i) reduces input-side mismatch via perplexity-driven Input Distribution Adaptation and (ii) sharpens output confidence by minimizing predictive entropy with a reverse-KL constraint to the base model; a dynamic importance weighting balances the two losses.

### Strengths
1.	Clear and well-structured writing.
2.	The paper targets test-time learning/adaptation for LLMs, a rapidly emerging direction with clear utility for real-world deployments that face distribution shift and scarce labels.

### Weaknesses
1.	**Limited novelty over TLM.** The paper directly reuses input perplexity minimization from prior TLM and mainly adds an output entropy term. Conceptually, this reads as an incremental extension that couples two known self-supervised signals rather than a fundamentally new objective.
2.	**Overstated “Contributions.”** Of the three listed contributions, #2 (“consistent performance gains”) and #3 (“extensive empirical analysis/ablations”) summarize results and diagnostics rather than introduce additional methodological innovations. This weakens the crispness of the claimed contributions.
3.	**Unsubstantiated claim around L191–193.** The statement that “Input Distribution Adaptation reduces input perplexity but does not ensure coherent or confident generation; models may still exhibit high predictive entropy or drift” is plausible but lacks a formal theoretical justification or a targeted empirical test explicitly verifying this claim. 
4.	**Benchmark coverage is narrow.** Experiments follow AdaptEval (DomainBench and InstructBench) as in TLM, but do not include ReasoningBench, leaving uncertainty about the method’s behavior on reasoning-heavy evaluations.
5.	**Reproduction gap relative to TLM.** Given that official TLM checkpoints are publicly available, please clarify any cases where reproduced results fall below TLM.
6.	**Computational cost.** Please report latency and memory cost to assess feasibility in realistic serving settings.
7.	**Effectiveness on quantized LLMs.** Are the gains preserved for common INT8/INT4 deployments?
8.	**Bibliography quality issues.** While minor in isolation, the number of bibliographic errors raises concerns about the manuscript’s overall rigor. For example: (i) the author list and venue for Test-time Learning for Large Language Models are incorrect; (ii) the author list for Efficient Test-time Model Adaptation without Forgetting is incorrect.

### Questions
see Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a technique for test-time LLM adaptation (SyTTA) that relies on minimizing the input prompt perplexity on the one hand, and learning a 4-16 token output prefix on the other hand. The two objectives are weighted dynamically.

### Strengths
- Consistent gains across multiple model families and datasets are demonstrated
- Clear writing style

### Weaknesses
Since the two driving forces here seem to be input distribution adaptation and output confidence shaping, the two obvious ablations are doing only one of them. While the first one (input distribution adaptation) is provided in form of the TLM baseline, an ablation with isolated output confidence shaping is missing.

Having ROUGE-Lsum as the single metric in the main paper is limiting. Table 4 in the appendix does contain BERTScore results, but the gains are less conclusive.

I don't have a background in test time adaptation, so this may be my ignorance. But I have a hard time buying the motivation behind the output confidence shaping. According to the authors, it "introduce an output-oriented objective that regularizes the next-token distribution" (L194). But IIUC the algorithm is actually the opposite of traditional regularization as it makes the model accumulate more probability mass on its own prediction.

Figs. 3-5 are too tiny, especially Fig. 5. I feel that this has become more common recently, but imo it should be a reason for desk-rejection if there is absolutely no chance of reading it on a print-out.

Minor comments:
The Hu et al. (2025) citation is for ICLR, but it should be ICML.

### Questions
- The approach seems quite expensive in terms of test-time compute - can you say more about the compoutational complexity?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes SYTTA (Synergistic Test‑Time Adaptation), a label‑free test‑time adaptation framework for LLMs that couples 

(i) input‑side perplexity minimization (IDA) with 
(ii) output‑side confidence shaping via entropy minimization plus a reverse‑KL “trust region” (OCS). 

A lightweight Dynamic Importance Weighting scheme balances the two signals on the fly. 
Two deployment modes are offered: Static‑Ref (compute/caches a short base‑model prefix once) and Dynamic‑Ref (updates while generating the prefix). 
Experiments on LLaMA‑3.1/3.2 and Qwen‑2.5 (7B/14B) show consistent ROUGE‑Lsum improvements, often with k = 4‑token prefixes and LoRA updates to q_proj and v_proj.

### Strengths
- Static‑Ref’s single forward pass per sample (during adaptation) is a practical and well‑motivated design. The cohort‑level, transductive “question‑only” setting matches common multi‑tenant deployments and makes the compute constraints explicit
- Principled way to sharpen predictions without collapse and to anchor updates near the base model. The mode‑seeking property of reverse‑KL makes it a reasonable “trust‑region” surrogate for generation.
- Analysis provides actionable choices and relates gains to model post‑training intensity (Qwen vs LLaMA), which will help practitioners.

### Weaknesses
- ROUGE/BERTScore/BLEU are surface overlap metrics. For domain QA they correlate imperfectly with factual correctness and safety. The paper lacks human or verifier‑based factuality/faithfulness evaluation (even sampled) and error analysis to ensure gains aren’t mainly stylistic (e.g., matching domain phrasing) rather than correct content.
- Does not report out‑of‑cohort generalization (e.g., adapt on batch A, test on disjoint batch B from the same target distribution), nor sensitivity to cohort size or streaming/batch arrival patterns
- All tasks are text‑generation QA/instruction following. Prior TTA results suggest entropy signals behave differently in code, math, or structured reasoning tasks with verifiers

### Questions
- Use paired bootstrap for ROUGE/BLEU and report 95% CIs.
- Since domains include health/finance, show a toxicity/hallucination sanity check (or cite one) to confirm KL+entropy do not over‑sharpen into confident but wrong outputs.
- Have you tried adaptive k (e.g., stop once average entropy drops below a threshold)? This may retain the benefits of k=4 while saving tokens when the signal is even earlier.
- Why do you restrict the lora to just q_proj and v_proj? Does more modules not yield gains?
- Can you add small‑scale human evaluation or automatic verifier checks (where feasible) to ensure that ROUGE improvements reflect correct content, not just stylistic alignment? Even random 100‑sample audits on Agriculture/Wealth would be informative.

### Soundness
3

### Presentation
3

### Contribution
3
