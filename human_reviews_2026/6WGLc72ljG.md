# Better, Faster: Harnessing Self-Improvement in Large Reasoning Models

- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
While large reasoning models (LRMs) trained with explicit reasoning trajectories have demonstrated impressive performance, obtaining high-quality trajectories is often costly and time-consuming. Hence, recent literature introduces a self-improvement paradigm that enables LRMs to improve themselves by self-generating reasoning trajectories as training data without external supervision. However, we find that this method often falls short in complex reasoning tasks and even leads to model collapse. Through a series of preliminary analyses, we reveal two shortcomings of self-improvement in LRMs: (1) data imbalance, where most training samples are simple, but the challenging yet crucial samples are scarce; (2) overthinking, where many undesired samples with redundant and repetitive reasoning steps are used for self-training. To this end, we propose HSIR, which effectively Harnesses Self-Improvement in large Reasoning models via two simple-yet-effective approaches. Specifically, HSIR introduces a verify-then-exit sampling strategy to mitigate data imbalance by efficiently collecting more accurate solutions for difficult queries, and designs an Intrinsic Diversity score to quantify overthinking and filter out the undesired solutions. We apply HSIR to various post-training paradigms, among which we further propose H-GRPO, an enhanced GRPO algorithm that leverages the intrinsic diversity as an external reward to encourage concise and diverse reasoning via reinforcement learning. Extensive results show that HSIR not only effectively enhances the reasoning performance, i.e., bringing up to +10.9% average performance gains, but also significantly improves the reasoning efficiency by reducing up to 42.4% relative inference overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles two limitations of self-improvement in reasoning models: data imbalance, where training data skews toward simple problems, and overthinking, where models generate redundant and repetitive reasoning steps. The authors propose HSIR, a framework featuring two core components to make models "better and faster". First, a VeriExit sampling strategy addresses data imbalance by efficiently salvaging correct intermediate reasoning from failed solutions for difficult queries . Second, an Intrinsic Diversity score, derived from an attention-aware eigenvalue analysis of the model's internal hidden states, quantifies and filters out "overthinking" solutions. HSIR is applied to both supervised fine-tuning and preference learning, and its InDiv score is also used as a reward in a new RL algorithm, H-GRPO. Experiments on medical and math reasoning tasks show HSIR boosts performance and improves efficiency by reducing inference overhead.

### Strengths
Writing is clear to follow. Visualizations are well designed.

It is good that this approach works on both (SFT, DPO) and RL

Diverse model sizes and model families are used for experimentation.

### Weaknesses
The work should mention how it compares and contrasts with “AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners” (NeurIPS 2025) as their method also related to:

> Through a series of preliminary analyses, we reveal two shortcomings of self-improvement in LRMs: (1) data imbalance, where most training samples are simple, but the challenging yet crucial samples are scarce; (2) overthinking, where many undesired samples with redundant and repetitive reasoning steps are used for self-training. 

The ❶ Data Imbalance section in the proposed paper has been introduced in this previous work.

Unclear how this is a good set-up to analze self-improvement. I.e., T=1 has almost no self improvement?
> For the implementation of self-improvement training, the number of iterations T is set to 1, and the total sampling times K is set to 10.

The paper should clearly mention all compute cost overhead incurred by the method over existing methods.

> sentence embeddings using the BGE-m3 model

The training data scope is limited to only two: gsm8k and medqa.

Is the seed data from a frontier model necessary? What are the compute overheads incurred due to this. Does any frontier model suffice? 

Are all the hyperparameters systematically set? Any sensitivity tests? It is especially unclear if T=1 is reasonable as mentioned above.

> The filter threshold τ is set to -0.5. Notably, for the post-training of Qwen2.5 models, the self-improvement iteration T is set to 3, but for the other LLMs, it is set to 1 due to limited computational resources.

Although E.4 PARAMETER ANALYSIS is available, it would be nice to double check if any more hyperparameters are newly introduced and might be sensitive. If it is sensitive it may be a pain for practitioners. 

What is the unit of train budget in e.g. table 1. Is it reflective of wall-clock time as well?

In industry labs we would want to see that peak performance, since we can checkpoint and go back to this one. However, it is not very convincing to see a fixed iteration budget performance, i.e., 3. Any experiments were we can see peak val set performance comparisons?

In table 2, SFT-Oracle seems to perform consistently better. In what scenario would we not have access to this? I.e., if there is already labeled data available, what is the point of using this method? Any experiments to show orthogonality? E.g. SFT-oracle + proposed method?

The gains on GRPO seem negligibly small in Tab. 4.

### Questions
Why call Iterative Reasoning Preference Optimization RPO? IRPO seems like the most appropriate abbreviation.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on self-improvement scenarios for llm reasoning.
The study identifies several key factors that affect a model's self-improvement capability, such as overly simplistic problems and model overthinking.

To address these issues, the paper proposes the Harnesses Self-Improvement in large Reasoning (HSIR) method:
1. verify-then-exit: Instead of directly discarding incorrect reasoning trajectories, this approach detects whether potential correct answers exist within them, thereby expanding the data instances..
2. Intrinsic Diversity for overthinking: The method analyzes the model's internal states to identify repetitive or redundant reasoning patterns, filtering such data to ensure conciseness.

Building on this, the authors integrate HSIR with GRPO and evaluate it on medical reasoning and arithmetic tasks, demonstrating the effectiveness of their approach.

### Strengths
1. This paper identifies key factors that influence performance improvement in model self-evaluation and proposes the HSIR method to enhance the quality of synthetic data generated through self-improvement. The authors subsequently integrate HSIR with GRPO for optimization.

2. Experiments conducted on the MedQA and GSM8K datasets demonstrate that the proposed approach outperforms several naive self-improvement baselines.

### Weaknesses
1. The contributions of this paper are limited, as the proposed methods rely heavily on existing approaches and lack core novelty.
    - The "verify-then-exit" strategy—which involves identifying and utilizing correct segments from failed reasoning trajectories—is a relatively common technique. Moreover, determining which parts of a failed trajectory are correct remains a challenging issue. If we solely rely on the appearance of the correct final answer as the criterion, this approach can only cover a narrow range of cases and is inadequate for most scenarios.
    - InDiv Score: Although labeled as "intrinsic," the idea of using hidden model states to quantify diversity is not novel. Simply incorporating attention weights does not constitute an insightful contribution. Furthermore, using attention weights as an intrinsic metric is a strong prior assumption, and its validity requires more thorough analysis and justification.
    - H-GRP: This is an incremental modification of GRPO that merely incorporates the InDiv score as an additional signal. The rationale behind this design and the reliability of the resulting reward signal remain questionable.

2. The experimental evaluation in the paper is insufficient. The main experiments only use MedQA and GSM8K—two relatively simple datasets. For out-of-distribution evaluation, the authors included Medbullets and MATH 500 datasets. Overall, the selection of datasets is limited and skewed toward lower difficulty. This lack of comprehensive experiments raises doubts about the method's effectiveness in realistically challenging reasoning tasks.

### Questions
please refer to weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes HSIR, a self-improvement recipe for reasoning LLMs that tackles two pain points of iterative self-training: data imbalance and overthinking. It introduces VeriExit, a verify-then-exit trajectory recycling strategy that truncates failed traces at the first correct step to harvest additional correct solutions on hard queries, and InDiv, an intrinsic diversity score computed from hidden states to filter verbose or redundant rationales; the pipeline is used for SFT and DPO, and extended to RL via H-GRPO that adds InDiv as a reward to encourage concise reasoning. Across benchmarks, the approach reports substantial gains in both accuracy and efficiency, with up to +10.9 percent average performance improvement and up to 42.4 percent reduction in inference overhead.

### Strengths
The paper is strong on originality by combining an intrinsic diversity criterion from hidden states with a verify-then-exit policy and a hierarchical GRPO variant, yielding a coherent recipe that reduces redundant reasoning without sacrificing accuracy. Methodologically the work is careful and high quality: it isolates contributions of InDiv and VeriExit with targeted ablations, tracks verifiable coverage to explain gains, and reports efficiency with detailed token accounting alongside accuracy. Clarity is high due to a clean problem framing, modular decomposition of the pipeline, and intuitive figures that map the training and inference loops to concrete decisions.

### Weaknesses
The paper’s core ideas are promising, but several issues limit clarity, generality, and fairness. 

- The “verify-then-exit” procedure depends on ground-truth answers to scan intermediate steps and insert an exit token, which weakens the claim of fully self-improving training and risks overfitting to answer formats rather than robust reasoning; it also resembles prior verifier or early-exit style decoding and needs a tighter novelty delineation relative to answer-guided sampling and verifier-augmented methods. 
- The Intrinsic Diversity score is computed from internal hidden states and attention weights, may be architecture and layer dependent, and lacks sensitivity analyses for layer choice, token selection, and thresholding; please report ablations on layer depth, normalization, and calibration across models, along with any failure modes when attention is diffuse. 

- Efficiency is measured primarily by token count, not wall-clock latency or energy, so it is unclear whether HSIR reduces real inference cost; add throughput and latency on standardized hardware and include the overhead of InDiv and VeriExit. 

- Comparisons could be fairer: VeriExit uses answer access while some baselines do not, iteration budgets and sampling counts differ across methods, and variance or significance is missing; please control total decoding tokens, report seeds and confidence intervals, and add stronger baselines such as verifier-trained STaR or RLVR variants with non-length rewards.

### Questions
How does VeriExit operate when gold answers are unavailable or noisy, and can you show robustness with imperfect verifiers or distant supervision? 

What are the key sensitivity factors for InDiv (layer choice, token pooling, normalization, thresholds), and do results hold across architectures and model sizes? 

Can you report wall-clock latency, throughput, and energy including the overhead of InDiv and VeriExit, not just token counts? 

Are K and J chosen under a fixed decoding budget, and how do performance and stability vary with these hyperparameters across multiple seeds with confidence intervals?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes HSIR, a self-improvement framework for Large Reasoning Models that targets two observed issues in self-training: data imbalance and overthinking. HSIR adds (i) VeriExit, a “verify-then-exit” decoding that scans previously failed traces and, once an intermediate step reaches the gold answer, truncates and resamples the final answer; and (ii) an Intrinsic Diversity (InDiv) score computed from model hidden states to filter repetitive traces. On MedQA and GSM8K with multiple LLMs, the authors report up to +10.9% average accuracy gains and –42.4% relative inference “overhead” . They also extend InDiv as an auxiliary reward in H-GRPO.

### Strengths
+ The paper crisply diagnoses imbalance/overthinking, and proposes two pragmatic fixes (trajectory recycling + intrinsic diversity) that are easy to add to SFT/DPO pipelines.

+ HSIR improves Qwen2.5 1.5B/3B/7B under SFT and DPO; it also outperforms STaR/ReSTEM and reduces tokens per output; broader one-iteration SFT results on Qwen3-1.7B, Phi-3.5-mini, Mistral-7B, and LLaMA3-8B support generality.

+ Useful ablations for VeriExit vs. answer-driven sampling and InDiv vs. length penalties; extension to H-GRPO shows accuracy preserved while keeping responses shorter relative to GRPO baselines.

### Weaknesses
- The pipeline relies on gold final answers to verify intermediate steps in VeriExit; robustness of the “step reaches the gold” heuristic (string match like “answer is {y}”) is unclear.

- Seed data are distilled from DeepSeek-R1 (and an alternative QWQ-32B), which may bias comparisons and complicate conclusions about self-improvement vs. teacher quality. Stronger decontamination/controls would help.

- Tasks are only MedQA and GSM8K; OOD checks are small (Medbullets/MATH). Broader reasoning (coding, multi-hop QA, planning) would strengthen generality.

- Baselines are not representative: key inference-time (SC/best-of-N/early-stop), verifier-rerank, and “SC→finetune” curation baselines are missing.

### Questions
as in weakness

### Soundness
3

### Presentation
3

### Contribution
3
