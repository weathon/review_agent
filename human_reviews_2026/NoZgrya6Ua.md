# References Improve LLM Alignment in Non-Verifiable Domains

- Decision: Accept (Poster)
- Scores: 2, 10, 4, 4

## Abstract
While Reinforcement Learning with Verifiable Rewards (RLVR) has shown strong effectiveness in reasoning tasks, it cannot be directly applied to non-verifiable domains lacking ground-truth verifiers, such as LLM alignment. In this work, we investigate whether high-quality reference outputs can be effectively leveraged to bridge this gap.  First, we design evaluation protocols that enhance LLM-based evaluators for LLM alignment using reference outputs. Through comprehensive experiments, we show that a reference-guided approach substantially improves the accuracy of less capable LLM-judges using references from frontier models; stronger LLM-judges can also be enhanced by human-written references. We then demonstrate the utility of high-quality references in alignment tuning, where LLMs guided with references are used as judges to self-improve. We show that reference-guided self-improvement yields clear gains over both SFT distillation and reference-free baselines, achieving performance comparable to training with finetuned reward models. Specifically, our method achieves scores of 73.1% and 58.7% on AlpacaEval and Arena-Hard with Llama-3-8B-Instruct, and 70.0% and 74.1% with Qwen2.5-7B. These results highlight the potential of using reference-guided LLM-evaluators to enable effective post-training in non-verifiable domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies how high-quality reference outputs can be used to improve LLM alignment. The authors first design prompting strategies that allow LLM-based evaluators to compare candidate answers with a reference, showing consistent accuracy gains across 11 open-source models on 5 datasets. They then extend the idea to training, where the model generates multiple outputs, ranks them with a reference-guided judge, and performs preference optimization (DPO). Experiments on AlpacaEval and Arena-Hard show that reference-guided self-improvement outperforms reference-free methods and reaches performance comparable to reward-model-based approaches, without requiring additional human feedback.

### Strengths
**Strength 1**: The study evaluates 11 open-source LLMs across five human-annotated datasets (LLMBar-Natural/Adversarial, MTBench, InstruSum, HREF), covering diverse architectures and capability tiers. This broad and systematic evaluation strengthens result credibility and supports reproducibility.

**Strength 2**: Reference-guided approaches such as RefEval and RefMatch consistently outperform both reference-free methods and prior reference-based baselines. Evidence from Table 1, Table 2, and Table 4 shows notable gains in both judgment accuracy and downstream alignment tuning, highlighting the practical effectiveness of references.

**Strength 3**: The analysis reveals stronger relative gains for smaller models and demonstrates that higher-quality references (e.g., human-edited) further improve judge performance. Results in Table 3 and Figure 4 provide helpful insights into when reference guidance is most beneficial.

### Weaknesses
**Weakness 1**: While the paper offers a more systematic and larger-scale study, the core concept of reference-guided judging remains closely related to prior work. In addition, recent studies like Self-Rationalization Improves LLM as a Fine-Grained Judge[1] similarly explore LLM-as-a-Judge and preference-based self-improvement. A clearer discussion of methodological differences and contribution boundaries would help clarify the incremental novelty introduced by leveraging external references.

**Weakness 2**: The pipeline uses frontier LLMs to generate references and also as strong evaluators in benchmarks like AlpacaEval and Arena-Hard. This may cause circular style or preference bias, making it difficult to confirm whether improvements reflect better human alignment or closer imitation of the reference model family.

**Weakness 3**: The paper does not provide clear criteria or measurements for selecting or validating high-quality references. This makes it difficult to assess generalization, especially when references may vary in coverage or factual reliability across datasets.

[1]Trivedi, Prapti, et al. "Self-rationalization improves llm as a fine-grained judge." arXiv preprint arXiv:2410.05495 (2024).

### Questions
Q1: What criteria are used to judge that a reference output is "high quality," and how sensitive is the method to different levels of reference quality?

Q2: Since GPT-4o is used to generate reference outputs and the evaluation benchmarks (e.g., AlpacaEval, Arena-Hard) are also judged by GPT-4 family models, how can we be sure that the improvements indicate stronger human preference alignment rather than closer imitation of GPT-4o-style responses? Incorporating human evaluation or cross-family judging would better support the alignment claims.

Q3: Can the proposed method work when references are unavailable or noisy in new domains, and what are the expected failure conditions when references deviate from correct or complete content?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper studies using references from frontier LLMs for LLM alignment. Specifically in domains where verifiable rewards are not available, the quality of less capable LLM evaluators can be much improved by guiding those with references from frontier LLMs.

### Strengths
1. The proposed approach is a step towards something similar to RLVR for non-verifiable domains, which makes this work interesting to the broad community.
2. Comparisons with strong baselines from literature are provided. In addition the authors design a strong reference-free baseline that is directly comparable to their referenced based approach in terms of prompt quality.
3. Experiments with sources from different frontier LLMs are conducted so that the results are not specific to a particular frontier LLM. This validates the the generalization of the proposed approach.
4. Finally, the paper shows results for actual preference alignment training using the reference guided evaluator. The results conclusively show that the proposed approach is superior.

### Weaknesses
1. The baselines in table 4 are not as strong. What I mean is that some of the baselines for table 4 should probably have been based on the baselines in table 1. If we are to choose a evaluator approach based on results from table 1, we would like to know whether the results in table 1 can serve as a robust tool for choosing a llm evaluator.

### Questions
In section 4.3 you wrote

 (3) V3-Distill is the SFT model
distilled from DeepSeek v3 references. The following models are finetuned from V3-Distill: (4) ROUGE,
which uses ROUGE scores as the reward5
; (5) BERTScore, which uses
the BERTScore metric (Zhang et al.,
2020; Zhao et al., 2025); (6) ArmoRM, which uses the ArmoRM reward model;

What does it mean to finetune ROUGE, BERTScore etc to be finetuned from V3-Distill?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper asks whether giving judges and students a good reference answer helps align smaller LLMs. It introduces three prompt styles for judging—Ref-Free (no reference), RefEval (compare to a reference), and RefMatch (pick the candidate closest to the reference)—and then uses references twice: (i) SFT to distill style/quality from strong references, and (ii) DPO where a reference-guided judge labels on-policy pairs for preference training. Across roughly 5 datasets and 11 judges, reference-guided prompts improve judge reliability, and the SFT→DPO pipeline lifts 7–8B models on common leaderboards (e.g., AlpacaEval, Arena-Hard). Overall, it’s a practical recipe: simple prompts, references as grounding, and a two-stage training loop that competes with a reward-model baseline without extra human feedback.

### Strengths
The paper asks a clear, practical question—can we ground LLM judges and training with strong references—and answers it with simple, reproducible tooling (Ref-Free, RefEval, RefMatch) and a clean SFT→DPO pipeline. The experimental setup is broad (many judges, several datasets) and the gains are consistent: reference-guided judging improves agreement/utility, and reference-distilled SFT followed by preference optimization moves mid-size models meaningfully on common leaderboards. I also like that prompts and templates are easy to reuse, and that multi-reference variants are explored rather than just a single “gold” answer.

### Weaknesses
Methodologically, the novelty feels incremental: reference-guided evaluation and reference-distilled training have both appeared before, and the paper mostly scales and systematizes them rather than introducing a new objective or learning signal. The comparisons also underplay strong preference-optimization baselines (e.g., SimPO, ORPO, KTO) and fine-grained supervision relevant to credit assignment (token/segment-level DPO; span-supervised MT like TWA). 

Robustness is not fully stress-tested: results rely on high-quality references, but we don’t see sensitivity to noisy, weaker, or stylistically different references, nor clear evidence against style coupling to the reference generator. Finally, statistical reporting is light (few CIs/permutation tests), and most end metrics are still LLM-judge based; more targeted human spot-checks would help address circularity.

Missing citation and baseline:
RevisEval: Improving LLM-as-a-Judge via Response-Adapted References — Qiyuan Zhang, ICLR 2025.

### Questions
Reference robustness: How do results change if you paraphrase, shorten, or inject minor errors into references, or use weaker LLMs as references? Any curve of judge accuracy vs. reference quality?

Baseline coverage: Can you add at least one strong SimPO/ORPO/KTO line and one fine-grained baseline (e.g., token/segment-level DPO or span-supervised MT like TWA) under the same data and compute?

Training schedule: Beyond two-stage SFT→DPO, can you try mixed SFT+DPO (interleaved or joint loss) and report its effect vs. your current recipe?

Judge calibration: How sensitive are outcomes to the particular judge and prompt? Any comments on the inter-judge agreement? Any human-correlation study for Ref-Free vs. RefEval/RefMatch?

Multi-reference: (1) I don't see the multi-reference results in Figure 3. (2) Do multi-reference votes always help? What’s the marginal gain of 1→2→3 references? How did you create the multiple references?

Significance: Please add 95% CIs (bootstrap) and a simple permutation test for key deltas in main tables.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether high-quality reference outputs can improve LLM alignment through enhanced evaluation and training methods. The authors develop reference-guided LLM-as-a-Judge protocols (RefEval and RefMatch) and demonstrate their effectiveness in both evaluation accuracy and self-improvement training scenarios.

### Strengths
* Evaluation across diverse datasets (Natural, Adversarial, MTBench, Instrusum, HREF, AlpacaEval and ArenaHard) and models (Qwen, Llama, GPT, Gemma, GLM, etc)
* Useful settings and approaches. RefEval provides explicit guidance on using reference outputs as benchmarks for instruction-following quality. RefMatch provides a good baseline to compare.

### Weaknesses
I mostly concern experiment design.
* Missing comparison to recent reference-based methods like RevisEval beyond a brief mention
* No systematic study of how reference quality affects downstream performance
* Unclear how to obtain high-quality references in domains where frontier models struggle
* No analysis of how reference diversity affects evaluation robustness, especially for aspects that reference responses miss.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
