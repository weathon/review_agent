# Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Recent advancements underscore the significant role of GRPO-based reinforcement learning methods and comprehensive benchmarking in enhancing and evaluating text-to-image (T2I) generation. However, (1) current methods employ pointwise reward models (RM) to score a group of generated images and compute their advantages through score normalization for policy optimization. Although effective, this reward score-maximization paradigm is susceptible to reward hacking, where scores increase but image quality deteriorates. This work reveals that the underlying cause is illusory advantage, induced by minimal reward score differences between generated images. After group normalization, these small differences are disproportionately amplified, driving the model to over-optimize for trivial gains and ultimately destabilizing the generation process. To this end, this paper proposes PREF-GRPO, the first pairwise preference reward-based GRPO method for T2I generation, which shifts the optimization objective from traditional reward score maximization to pairwise preference fitting, establishing a more stable training paradigm. Specifically, in each step, the images within a generated group are pairwise compared using preference RM, and their win rate is calculated as the reward signal for policy optimization. Extensive experiments show that PREF-GRPO effectively differentiates subtle image quality differences, offering more stable advantages than pointwise scoring, thus mitigating the reward hacking problem. (2) Additionally, existing T2I benchmarks are limited to coarse evaluation criteria, covering only a narrow range of sub-dimensions and lacking fine-grained evaluation at the individual sub-dimension level, thereby hindering comprehensive assessment of T2I models. Therefore, this paper proposes UNIGENBENCH, a unified T2I generation benchmark. Specifically, our benchmark comprises 600 prompts spanning 5 main prompt themes and 20 subthemes, designed to evaluate T2I models’ semantic consistency across 10 primary and 27 sub evaluation criteria, with each prompt assessing multiple testpoints. Using the general world knowledge and fine-grained image understanding capabilities of Multi-modal Large Language Model (MLLM), we propose an effective pipeline for benchmark construction and evaluation. Through meticulous benchmarking of both open and closed-source T2I models, we uncover their strengths and weaknesses across various fine-grained aspects, and also demonstrate the effectiveness of our proposed PREF-GRPO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PREF-GRPO, a pairwise preference reward-based GRPO method for text-to-image generation, and UNIGENBENCH, a unified benchmark for fine-grained evaluation. The authors claim their method addresses reward hacking by shifting from reward score maximization to pairwise preference fitting. While the paper tackles an important problem in T2I reinforcement learning, there are significant concerns regarding the experimental setup and claims validation.

### Strengths
1. The paper identifies an important problem in T2I reinforcement learning - reward hacking caused by illusory advantage.
2. The introduction of UNIGENBENCH with fine-grained evaluation dimensions is valuable for the community.
3. The pairwise preference approach is conceptually interesting and aligns with human evaluation processes.

### Weaknesses
1. Unfair Baseline Comparisons: The experimental setup raises significant concerns:
- PREF-GRPO uses UnifiedReward-Think while baselines use UnifiedReward without the thinking mechanism, creating an unfair advantage
- HPS is outdated (the community now primarily uses HPSv2, ImageReward, and MPS)
2. Insufficient Evidence for Reward Hacking Mitigation: The paper fails to provide convincing evidence that PREF-GRPO actually alleviates reward hacking:
- No analysis of training dynamics beyond basic reward curves
- Missing monitoring metrics on established benchmarks like GenEval throughout training
- The comparison in Table 1 only shows final performance, not the stability of the training process
3. Limited Ablation Studies: The paper lacks sufficient ablation studies to understand the contribution of different components of PREF-GRPO.

### Questions
- Have you conducted any analysis to show that PREF-GRPO actually reduces reward hacking behaviors rather than just achieving better final scores?
- How does the computational cost of PREF-GRPO compare to baseline methods, especially as group size increases?

### Soundness
2

### Presentation
3

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
The paper presents a GRPO method for tuning diffusion models with rewards along with a comprehensive benchmark for evaluating text-to-image models. Compared to previous GRPO methods for diffusion/flow models (e.g. {Flow,Mix,Dance}-GRPO), the proposed approach (Pref-GRPO), does not directly compute a reward per sample, but rather computes the reward over the full batch. This appears to make the advantage computation more reliable, leading to better performance. 
Additionally, the paper also introduces "UniGenBench", which has several categories for comprehensive evaluation is evaluated using the LLM-as-a-judge methodology with Gemini 2.5 Pro.

### Strengths
I think the paper proposes an interesting mechanism to deal with the issue of reward-hacking in the reward optimization of diffusion models, which is backed by fairly strong results on several benchmarks. 

The proposed UniGenBench also appears to be a useful addition in terms of evaluating newer models. While I'm not entirely sure that this would become a widely used benchmark given that the field has several benchmarks already (GenEval, T2I-Compbench, DPG-Bench, GenAI-Bench, TIFA etc.), it does have its merits.

### Weaknesses
[Major]

Comparisons with Previous GRPO work: I think the most important question is regarding the performance of Pref-GRPO compared to other formulations (Flow-GRPO, DanceGRPO etc.). The key claim being made in the paper is that the pairwise reward formulation is better suited to compute advantages for optimizing the model compared to existing work. While there seem to be some results in Fig. 2, Tab. 4-6, I'm not exactly sure how these settings compare to the previous GRPO methods. 


[Minor]

A slightly curious aspect of the paper is that the benchmark and mehtod are quite orthogonal; i.e the benchmark on its own can provide useful analysis, while the method could also be validated on existing benchmarks, and to some extent the paper feels like 2 disjoint works stitched together.

### Questions
The only question I'd really like to have full clarity is the differences and advantages over other GRPO frameworks for diffusion models. While I'm leaning to accept the paper, I think answering this comprehensively would be ideal.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces PreF-GRPO, a pairwise preference reward-based Group Relative Policy Optimization (GRPO) method designed to address the reward hacking problem in text-to-image (T2I) reinforcement learning. By shifting from pointwise reward maximization to pairwise preference fitting, the method aims for more stable learning and better alignment with nuanced human preferences. Additionally, the authors present UniGenBench, a new benchmark for T2I generation that evaluates models across comprehensive primary. They also fine-grained sub-dimensions using an automated pipeline based on Multi-modal Large Language Models (MLLMs). Through extensive experiments, the authors demonstrate that PreF-GRPO achieves notable improvements in both image quality and semantic consistency compared to existing baselines.

### Strengths
[+] For the reward hacking problem in existing GRPO methods, this paper identifies “illusory advantage” resulting from minimal score differences and their amplification during normalization

[+] It is well motivated that replace pointwise rewards with pairwise preference-based win rates, which is illustrated in Figure 1.

[+] The description and motivation of UniGenBench as a benchmark is solid, with Figure 3 and Figure 4 substantiating its comprehensiveness.

### Weaknesses
[-] There is a lack of formal analysis on the illusory advantage phenomenon. It would be better to have a more rigorous analysis quantifying the expected change in variance between score-based and pairwise win-rate rewards.

[-] Most ablations focus on comparing PreF-GRPO to existing pointwise methods or naive adaptations, without the adversarial preference model errors analysis.

[-] The assertion that PreF-GRPO produces more “human-aligned” or “faithful” outputs is plausible but not decisively demonstrated without human assessment.

### Questions
1. Can the authors provide a more formal mathematical analysis of reward variance and its amplification?
1. How well does  MLLM-based automated evaluation (via Gemini2.5-pro) agree with human annotators?

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
4

### Summary
This paper brilliantly identifies and solves "reward hacking" in T2I models. It argues the problem is "illusory advantage": standard reward models give similar images very similar scores, which, when normalized, create huge, noisy, and fake "advantages" that make training unstable.

The fix, PREF-GRPO, is elegant. Instead of using flawed absolute scores, it makes images in a group compete. It uses a pairwise model ("A is better than B") and gives each image a "win rate." This signal is far more stable and stops the model from hacking. As a huge bonus, the paper also introduces UNIGENBENCH, a new, super-detailed benchmark for T2I evaluation.

### Strengths
- The "illusory advantage" diagnosis is brilliant.

- The "win rate" solution is an elegant and effective fix.

- The visual proof is undeniable; the qualitative images show this method works and the others don't.

- It also contributes a fantastic new benchmark.

### Weaknesses
The main potential issue is computational cost. A "win rate" for a group of 8 images requires 28 pairwise comparisons ($O(G^2)$), versus just 8 for the baseline ($O(G)$). This seems significantly slower per training step, which the paper doesn't heavily focus on.

### Questions
How much does the $O(G^2)$ complexity of pairwise comparisons slow down the actual wall-clock training time compared to the $O(G)$ baseline?

### Soundness
3

### Presentation
3

### Contribution
3
