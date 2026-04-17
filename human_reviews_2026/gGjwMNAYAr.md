# Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback from a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0\% ASR on Claude-3.5-Haiku and 100.0\% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AMIS (Align to MISalign), a meta-optimization framework for jailbreaking large language models that jointly evolves both adversarial prompts and the scoring templates. The approach employs bi-level optimization: an inner loop that iteratively refines jailbreak prompts using fine-grained continuous scores from a scoring template, and an outer loop that optimizes the scoring template itself using an "ASR alignment score" that measures how well the template's continuous scores correlate with binary attack success outcomes. The paper demonstrates strong empirical results, substantially outperforming baselines such as PAIR, TAP, and AutoDAN-Turbo.

### Strengths
- **Well-motivated problem.** The paper identifies an important and under-explored aspect of jailbreak research: the evaluation mechanism itself. While prior work has focused primarily on how to generate adversarial prompts, this paper makes a compelling case that how we evaluate and score jailbreak attempts significantly impacts optimization effectiveness. The observation that different scoring templates yield substantially different results is  valuable  and motivates the meta-optimization approach.

- **Novel meta-optimization perspective.** The bi-level optimization framework that jointly evolves both jailbreak prompts and scoring templates is conceptually interesting and represents a novel perspective in the jailbreak literature. The idea of treating the evaluation mechanism as a learnable component rather than a fixed heuristic is a reasonable contribution to the field, even if the empirical gains are modest.

- **Strong empirical results on challenging models.** The paper demonstrates impressive attack success rates on recent, well-aligned models including Claude-3.5-Haiku (88% ASR) and Claude-4-Sonnet (100% ASR), significantly outperforming the considered baselines. These results on state-of-the-art commercial models are noteworthy and suggest the method can expose vulnerabilities in production systems, which is valuable for the AI safety community.

- **Comprehensive experimental evaluation.** The paper includes experiments across five target models (both open-weight and commercial), two datasets (AdvBench and JBB-Behaviors), and two evaluation metrics (ASR and StrongREJECT). The inclusion of both white-box (Llama) and black-box (GPT, Claude) settings, along with transferability analysis (Figure 4), provides a reasonably thorough empirical assessment.

- **Ablation study.** Table 3 systematically examines the contribution of different components, which helps readers understand which design choices matter.

### Weaknesses
- **Weak inner loop baseline without critical ablations.** The inner loop uses modest hyperparameters (K=5, M=5, L=5), and the paper does not include ablations demonstrating that the outer loop's benefits persist with stronger inner loop configurations (e.g., K=20, M=10, L=10). Given that the outer loop provides only a +2.0% ASR improvement over the fixed-template baseline, it remains unclear whether scaling up the inner loop alone would achieve comparable results. 

- **Missing comparisons with recent state-of-the-art methods.** While the paper claims "state-of-the-art performance," it does not include experimental comparisons with Rainbow Teaming (Samvelyan et al., 2024) and the natural distribution shifts approach (Ren et al., 2025), both of which are cited in the related work section and address the same black-box jailbreaking problem. Including these baselines would provide a more complete picture of where AMIS stands relative to the current state of the art and would strengthen the paper's claims.

- **Insufficient evidence of template evolution.** The paper claims that templates "evolve" and become "stronger," but provides limited empirical support for this progression. Only the initial template (π₀, Listing 3) and final template (π₅, Listing 1) are shown; intermediate templates and their corresponding alignment scores are not presented. Providing this information—such as a table showing Align(π₀) through Align(π₅)—would demonstrate that the meta-optimization is indeed improving template quality monotonically and help readers understand what changes occur during optimization. This evidence would substantially strengthen the paper's core contribution.

### Questions
**Feasibility of Rainbow Teaming/distribution shift methods as inner loop**

Are Rainbow Teaming and natural distribution shift methods architecturally compatible with serving as drop-in replacements for your inner loop, or do they have fundamental design differences (e.g., their own evaluation mechanisms) that prevent integration? If integration is feasible, would the outer loop still provide meaningful improvements when combined with these stronger methods?

Understanding whether these methods can be integrated would clarify if they should be compared as standalone baselines or if architectural incompatibilities legitimately prevent their use as inner loop variants. If they can be integrated and still benefit from template optimization, it would significantly strengthen your contribution's generality.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AMIS, a novel meta-optimization framework for automated jailbreaking of large language models (LLMs). Unlike prior work that treats scoring functions as fixed, AMIS introduces a bi-level optimization architecture that co-evolves both jailbreak prompts and their evaluation criteria. In the inner loop, prompts are refined using dense, fine-grained feedback from a fixed scoring template. In the outer loop, the scoring template itself is optimized based on an ASR alignment score that measures how well its continuous outputs correlate with binary attack success outcomes. Extensive experiments show that AMIS achieves state-of-the-art attack success rates, significantly outperforming existing baselines.

### Strengths
(1) Introduces a new framework to treat scoring templates as trainable components in jailbreaking.

(2) Demonstrates empirical results across multiple datasets and target models.

(3) Highlights the importance of how we evaluate attacks, shifting focus from pure attack engineering to meta-evaluation design.

### Weaknesses
(1) Limited analysis of template evolution dynamics: Do templates converge? Are they interpretable?

(2) Potential for overfitting specific model behaviors or datasets; cross-dataset generalization needs more exploration.

(3) Reliance on powerful attacker LLMs for both prompt and template optimization may limit accessibility or realism in some threat models.

(4) No discussion of computational cost or query efficiency, which matters for black-box settings.

(5) Lack of failure case analysis (e.g., when high-scoring prompts still fail).

### Questions
1.Can you provide examples of how the scoring template changes across outer-loop iterations? Does it become more nuanced or simply more aggressive?

2.Have you tested whether scoring templates trained on AdvBench generalize to JBB-Behaviors (or vice versa)? What factors influence transferability?

3.How sensitive is AMIS to the choice of judge model? Would weaker judges degrade performance disproportionately?

4.Could this framework be inverted to improve defensive monitoring systems—e.g., training better detectors via similar co-evolution?

### Soundness
2

### Presentation
2

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
The paper proposes AMIS, a meta-optimization framework for automatic LLM jailbreak. AMIS jointly evolves (i) jailbreak prompts using dense, fine-grained scores on a 1–10 scale, and (ii) the scoring template itself using an ASR alignment score that measures how well template scores correlate with binary ASR across queries. Reported headline numbers include 88.0% ASR on Claude-3.5-Haiku and 100% ASR on Claude-4-Sonnet on AdvBench, with similarly strong results on JBB-Behaviors.

### Strengths
- Co-optimizing the rubric itself is a meaningful advance over fixed-template approaches and addresses evaluation misalignment.

- Clear wins on ASR and StR vs. strong baselines across multiple targets

- Benchmarks, metrics, model roles, and hyperparameters are well documented, which is helpful for reproducibility.

### Weaknesses
- AMIS still relies on LLM-as-judge for both optimization (inner loop) and ASR labels, which may introduce biases and overfit to judge characteristics. Stronger evidence via human evaluation or multiple heterogeneous judges would strengthen claims.

- The outer loop’s objective is an ASR-alignment score; it is not fully clear how sensitive AMIS is to the choice and calibration of that alignment objective. More analysis on stability across different score ranges/targets would help.

### Questions
See Weaknesses.

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
4

### Summary
This paper targets a critical bottleneck in optimization-based LLM jailbreaking: the quality and reliability of the reward signal used for optimization. It argues that existing signals are suboptimal, being either too sparse (e.g., binary Attack Success Rate) or misaligned and biased (e.g., manually-crafted, fixed scoring templates).   

To solve this, the paper introduces AMIS (Align to MISalign), a sophisticated bi-level meta-optimization framework.   

- Inner Loop (Query-level Optimization): An attacker LLM iteratively refines jailbreak prompts. These prompts are evaluated by a judge LLM (e.g., GPT-4o-mini) using a fine-grained, dense scoring template (e.g., 1.0-10.0 scale) to provide a stable, continuous optimization signal.   

- Outer Loop (Dataset-level Meta-Optimization): This is the paper's core innovation. The framework optimizes the judge's scoring template itself (which is an instruction prompt). It does this by introducing a novel metric, the "ASR Alignment Score,"  which measures how well the judge's dense scores (from the inner loop) correlate with the true binary ASR. An optimizer LLM then generates a new scoring template, guided to maximize this ASR Alignment Score.   

This co-optimization process yields progressively stronger jailbreak prompts and more calibrated reward signals.

### Strengths
- Novelty: The core contribution—using meta-optimization to evolve the instruction prompt of the LLM-as-a-judge —is novel and intelligent solution to the known problem of reward signal quality. This "meta-prompt optimization"  of the evaluator is a significant advance.   

- Framework: The bi-level optimization structure  is sound. The "ASR Alignment Score"  is a key innovation, providing a principled objective to align the dense, inner-loop reward with the sparse, ground-truth outer-loop objective.   

- Excellent SOTA Results: The framework demonstrates outstanding practical effectiveness, achieving near-perfect or perfect ASRs against even the most robust proprietary models available (Claude series, GPT-4o).

### Weaknesses
Computational Cost & Complexity: The primary weakness is the practical barrier to adoption. The bi-level, meta-optimization loop  is inherently complex and computationally expensive. It requires running multiple full inner-loop iterations (each of which is already an iterative optimization process) for each outer-loop step, significantly increasing the total query cost. While Appendix E  provides some cost analysis, this remains a significant consideration.

### Questions
1. The computational cost is a significant practical concern. The analysis in Appendix E is helpful, but could the authors include a brief discussion of the total query cost (e.g., number of LLM calls) of AMIS relative to baselines like PAIR in the main paper?

1. To visually confirm that the meta-optimization (outer loop) is working as intended, could the authors add a plot (e.g., in the appendix) showing the "ASR Alignment Score" of the best-so-far template as a function of the outer-loop iterations ($L'$)? This would demonstrate that the outer loop is indeed converging and finding progressively "better-aligned" scoring templates.

### Soundness
3

### Presentation
3

### Contribution
3
