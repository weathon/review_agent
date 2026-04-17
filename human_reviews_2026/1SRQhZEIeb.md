# Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the “aha moment”, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce **Double-Checker**, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, *Double-Checker* empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until their solutions are evaluated as correct under self-generated critiques.
We validate the efficacy of *Double-Checker* across various reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes Double-Checker, a data distillation framework designed to synthesize Chain-of-Thought (CoT) data with reflective behavior. Specifically, the study employs probing techniques to reveal that reflection may not always yield beneficial information. The authors synthesized 1,730 instances of data exhibiting reflective behavior and fine-tuned the DeepSeek-R1-Distill-7B and 32B models. Experimental results demonstrate the importance of self-critique in the process.

### Strengths
1. The data synthesis and distillation methodology is reasonably designed and demonstrates potential for industrial applications.

2. The paper is clearly written and easy to follow.

### Weaknesses
1. The probing technique employed in this work does not align with the conventional understanding of probing methods. The authors simply designed a prompt to ask the LLM whether it considers reflection informative. However, to the best of my knowledge, standard probing techniques [1] typically involve accessing the model's internal representations and training an additional linear classifier to detect whether specific information is encoded in a particular layer.

2. The novelty of this work is somewhat limited. The authors' first claimed contribution—that reflective behavior "may not help the model generate informative critiques"—is already an established consensus in the field [2]. Their second claimed contribution, synthesizing a dataset with reflective behavior, is more reasonable but lacks technical innovation and demonstrated potential for large-scale industrial application (the dataset comprises only 1,730 instances, which is insufficient for thorough validation).

3. Fine-tuning a language model to demonstrate reflective behavior may not be an optimal approach. I have not observed the use of methods such as reinforcement learning to enable the model to spontaneously discover problem-solving pathways and generalize this capability to other domains. Conversely, the current approach may introduce risks of overfitting and data leakage.

[1] Liu et al. Probing Language Models for Pre-training Data Detection. ACL 2024

[2] Huang et al. LARGE LANGUAGE MODELS CANNOT SELF-CORRECT REASONING YET. ICLR 2024

### Questions
Please refer to Weaknesses 1, 2, 3

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Double-Checker, a framework designed to improve the reasoning abilities of "slow-thinking" large language models. The authors first observe that while these models exhibit reflection-like behavior (an "aha moment"), they struggle to effectively critique their own solutions and refine them. To address this, Double-Checker fine-tunes a long-CoT LLM on a small, curated dataset of 1,730 self-critical instances. This training teaches the model to perform explicit self-critique and iterative refinement. During inference, the Double-Checker model first generates a solution, then repeatedly critiques and refines that solution until it evaluates its own answer as correct. This "reflect-and-refine" loop  allows the model to catch and correct its own errors.

### Strengths
1. Identifies a Gap: This paper demonstrates that the "aha moment" in long-CoT models does not automatically translate into an effective self-critique mechanism. Probing experiments showed that baseline models often fail to generate informative critiques even when prompted.
(This validation approach is quite insightful. However, I have some concerns regarding the specific models chosen for this experiment (DeepSeek-R1-Distill-7B and 32B). See weakness 1)
2. Proposes the Double-Checker Framework: This framework enables LLMs to iteratively critique and refine their outputs by fine-tuning on a mix of direct inference data 1.7k critique-refine examples.
3. Achieves Performance Gains: The method shows substantial improvements on various reasoning benchmarks. Average pass@1 performance on DeepSeek-R1-Distill-7B: 57.4% to 61.5%; DeepSeek-R1-Distill-32B: 68.8% to 75.4%.
4. Demonstrates Data Efficiency: The paper highlights that significant improvements can be unlocked with a very modest amount of curated critique data (1,730 instances), in contrast to other methods that use much larger datasets.

### Weaknesses
1. Baseline for Probing "Aha Moment": The paper's initial premise—that long-CoT models lack self-critique—is tested on SFT-distilled models (DeepSeek-R1-Distill-7B/32B). Models trained via supervised fine-tuning  are often prone to exhibiting "superficial self-reflection"[1][2]; they learn to mimic the style of reasoning or critique without possessing the underlying capability. The paper's conclusion about long-CoT models would be much stronger if this probing experiment (Sec 3.1) were also conducted on models trained with different paradigms, such as reinforcement learning (like the full DeepSeek-R1) or other powerful reasoning models. The observed weakness might be an artifact of SFT distillation rather than a fundamental limitation of all long-CoT models.
2. Diminishing Value of Iteration: The value of iterative refinement (the core loop of the framework) seems to diminish rapidly. In the 32B model analysis (Figure 3), almost all of the performance gain is achieved at the first refinement step ($N=1$), with performance flatlining at $N=2$ and $N=3$. This suggests the model may not be learning a general, multi-step refinement process, but rather just learning to perform a single-step correction (especially at larger scales).
3. We alse see a significant performance drop at $N=2$ (from 64.6% to 62.7%) before recovering at $N=3$. This non-monotonic improvement raises concerns about the method's robustness.

[1] Training Language Models to Self-Correct via Reinforcement Learning. ICLR 2025

[2] Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards. NeurIPS 2025

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Double-Checker teaches long-chain-of-thought (long-CoT) LLMs to iteratively critique and refine their reasoning using supervised fine-tuning on ~1,730 curated self-critique instances. The model generates structured critiques with correctness judgments, enabling inference-time refinement loops with early stopping. Experiments show substantial gains on mathematical reasoning benchmarks (AIME24: 56.7% → 66.4% for 7B), with ablations isolating the value of critique data.

### Strengths
Section 3.1 "Aha moment does not equate to effective self critique" is a great inisght into the limitations of current reasoning models.

Prior "slow-thinking" models like DeepSeek-R1 show long reflective chains-of-thought but often fail to generate precise, actionable critiques and then actually fix their own earlier answer. Double-Checker encodes that as a supervised behavior: the model learns to (a) summarize its own reasoning, (b) judge correctness, and (c) propose concrete fixes — then immediately attempt those fixes.

Instead of needing hundreds of thousands of RL rollouts or massive synthetic corpora, authors fine-tune on ~1.7K curated critique/refinement traces plus a subset of direct-inference data. Authors start from distilled long-CoT variants of DeepSeek-R1 (7B, 32B) and continue full-parameter SFT using DeepSpeed ZeRO, FlashAttention, and a 32K context window. This is a comparatively lightweight recipe to get iterative self-correction.

The 7B Double-Checker model beats the same 7B base model (DeepSeek-R1-Distill-Qwen-7B) by ~9–10% average absolute accuracy across benchmarks, and especially large boosts on AIME24. The 32B Double-Checker model outperforms its 32B distilled base by ~11% average absolute accuracy and delivers a ~18.2% jump in pass@1 on AIME25. 

"Naive SFT" (fine-tune on the same questions without explicit critique data) improves only mildly over the distilled baseline.

Adding explicit critique + refinement data yields large gains, especially after just one critique round (N=1).

Removing direct-inference data hurts (model forgets how to do first-pass reasoning), and removing Qwen3-generated critiques also hurts, showing both components are doing useful work.

Inference-time behavior is analyzed. The authors measure per-round vs cumulative token usage for up to 3 refinement rounds. Most of the compute is in round 0 (the first long reasoning trace); later critique/refine rounds are shorter and add relatively small incremental token cost. Cumulative context grows with each round, but each additional round is cheaper than the first, and sometimes even cheaper than a naive single-shot baseline that lacks targeted critique.

### Weaknesses
Dataset is limited to the math domain. This may limit the evidence of reasoning beyond symbolic reasoning. Authors may want to write about this in limitations. 

The paper does not analyze the quality or diversity of the critiques generated by the two teacher models. This introduces the risk of teacher bias propagation. Authors may want to write a line or 2 in limitations about this. Authors may want to write about this in limitations. 

Stopping is triggered when the model's own critique says the prior answer is correct. Does this self certification introduce any confounding / limitations to the work? 

The paper does not quantify "false green lights": cases where the model incorrectly certifies a wrong answer as correct and thus prematurely stops improving. Authors may want to consider adding these in the revised draft. 

Double-Checker is framed as an SFT-only alternative to RL-heavy methods like DeepSeek-R1 (pure RL to elicit long reasoning) and LightR1 (SFT + RL). Are there other alternatives? Can authors mention this in the discussion? 

Paper does not provide compute-normalized comparison, i.e. how much total RL rollout cost vs Double-Checker's ~1.7K curated traces?  Without normalizing cost, isn't hard to claim Double-Checker is strictly "more efficient."? 

Paper does not provide full qualitative traces for physics/biology-style questions to confirm that the model is truly doing scientific consistency checks rather than just math-style arithmetic audits. Authors may consider adding these. 

The paper gives qualitative examples of good critiques and refinements, but less about persistent failures: e.g., when repeated critique rounds still don't fix an algebraic slip, or when the model hallucinates constraints not in the problem. Authors may want consider adding these. 

Authors may consider providing breakdown of the hardest unsolved AIME/Olympiad problems, which would help future work. 

Authors may consider providing more details about GPU / training, etc. For example, paper mentions using a 32K context and full-parameter fine-tuning (DeepSpeed ZeRO, FlashAttention), but not exact GPU-days or hardware for training.

Paper does not report Pass@1 vs Pass@k tradeoffs in a way that makes it easy to compare to "test-time sampling with reranking" baselines like ThinkTwice beyond what's in Table 1. 

Authors provide token-per-round and cumulative token curves (Fig. 4) and note that later rounds are cheaper than the first long-CoT pass.
Please translate that into average wall-clock solve time for AIME-style questions on (a) 7B and (b) 32B, with N=0 vs N=1 vs N=3 rounds. This may help for deployment vs RL-style long-CoT models such as DeepSeek-R1.

### Questions
Have authors tried scaling critique data beyond 1.7 k examples—do returns diminish quickly?

Can this framework also include RL Finetuning? Can self-critique be incentivized/learnt through verifiable rewards?

Can Double-Checker be extended beyond math reasoning to tasks like STEM, legal etc?

How trustworthy is the "I'm correct " stopping signal?

Can authors quantify cases where the model declared correctness but the final answer was still wrong? (This is essentially a self-verification failure rate.) 

Can authors report latency / throughput?

Is one refinement round (N=1) basically the sweet spot?

Could you ship a "fast mode" that always does exactly one critique+refine round (regardless of correctness judgment) and stops? How close is that to full Double-Checker performance?

For AIME25 problems that still fail after N=3, what actually goes wrong? Arithmetic slips that never get fixed? Misinterpreting geometry wording? Hallucinated assumptions? A short categorized breakdown would help future researchers target those weak spots.

Could authors provide representative GPQA critique traces to show that the model is, for example, checking factual consistency in biology/physics like "is this premise consistent with known laws," instead of just doing algebra sanity checks?

Authors mention starting from ~8K questions filtered for verifiability and difficulty, then distilling them with Qwen3 and DeepSeek-R1 to produce critique+refine trajectories, and ending with 1,730 training instances. Can you release the filtering rules and prompt templates used for (i) correctness judgment, (ii) structured critique, and (iii) refinement, so others can reproduce the dataset generation pipeline?

Can the authors provide an analysis on the quality/fidelity of the critiques? For the problems that remain unsolved after $N=3$, does the critique fail (i) by not identifying the primary error, or (ii) by identifying the error but providing an insufficient refinement?  

Given the critical finding in Section 3.1, can the authors elaborate on the specific reasoning error patterns (e.g., computational error vs. logical flaw) that the long-CoT models typically fail to self-critique, even when probed?  

The results show $N=1$ provides most of the gain for the 32B model, while the 7B model benefits more from $N=2$ and $N=3$. Can the authors discuss if this suggests larger models acquire the self-critique capability faster and thus require fewer iterations in production?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper first conducts experiments demonstrating that current Long-CoT LLMs lack strong self-critique capabilities, which prevents them from improving output reliability through multi-round iterative reasoning.
To enhance the self-critique ability of Long-CoT LLMs, the authors propose a synthetic data generation framework that produces critique–refine pairs, resulting in 1,730 training instances covering both critique–refine and direct-inference modes. They then apply supervised fine-tuning (SFT) to strengthen the model’s self-critique behavior.
Through comprehensive experiments across multiple benchmarks and detailed analyses, the paper validates the effectiveness of the proposed approach.

### Strengths
- The paper proposes a framework for synthesizing high-quality critique–refine data. Empirically, supervised fine-tuning on the 1,730 synthetic instances significantly improves the model’s self-critique ability.

- The authors evaluate their method on a diverse set of benchmarks and provide detailed analyses regarding factors such as the number of refinement iterations and the composition of training data.

- The paper is well-written and easy to follow.

### Weaknesses
- The paper shows limited novelty. Several prior works [1,2,3] have already explored similar ideas — constructing a pipeline to synthesize critique–refine data, training LLMs to enhance self-critique ability, and leveraging this during inference to improve reliability. The paper does not clearly articulate what distinguishes its approach from these existing studies.

- The paper lacks an analysis or report on the computational cost required for synthetic data generation. This omission raises concerns about the scalability and practicality of the proposed framework, as high generation costs could hinder broader adoption.

- The experiments are conducted solely on the DeepSeek model. The absence of results across other base models limits the generalizability of the findings and weakens the empirical validation.


[1] Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate. COLM 2025
[2] Critic-CoT: Boosting the Reasoning Abilities of Large Language Model via Chain-of-Thought Critic. ACL 2025 findings 
[3] Self-Evolving Critique Abilities in Large Language Models. COLM 2025

### Questions
- Compared to previous studies with similar ideas, what are the key differences and core innovations of this work? 

- Is there any analysis or evaluation regarding the quality of the synthesized data? 

- In the Introduction, the paper mentions that prior methods (e.g., Wang et al., 2025; Tian et al., 2025) fail to leverage self-critique effectively during inference or yield only marginal improvements. Is there any detailed analysis of why these prior approaches underperform?

### Soundness
2

### Presentation
3

### Contribution
2
