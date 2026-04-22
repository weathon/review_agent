# Mitigating Visual Hallucinations via Semantic Curriculum Preference Optimization in MLLMs

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) have significantly improved the performance of various tasks, but continue to suffer from visual hallucinations, a critical issue where generated responses contradict visual evidence. While Direct Preference Optimization (DPO) is widely used for alignment, its application to MLLMs often fails to capture fine-grained semantic differences and encourages shortcut learning. To address these challenges, we propose Semantic Curriculum Preference Optimization (SCPO), a novel framework for MLLM alignment. SCPO employs a progressive, easy-to-hard curriculum built upon our Semantic Curriculum Preference Pairs dataset, which provides fine-grained semantic contrasts sorted by difficulty. This curriculum is trained with a dynamic reference model and a novel symmetric, bidirectional objective to facilitate simultaneous learning from both textual and visual preferences. To our knowledge, SCPO is the first framework to unify semantics, symmetry, and curriculum for MLLMs alignment, effectively mitigating visual hallucinations. Extensive experiments on LLaVA models across various scales and versions validate that SCPO demonstrates superior performance compared to baseline models on multiple hallucination benchmarks, reducing the hallucination rate by up to 62.9%. Moreover, evaluations on generalized benchmarks show that SCPO improves factuality while preserving general capabilities, with its performance remaining stable across general vision-language benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address visual hallucinations in MLLMs, this paper introduces SCPO, a framework that unifies curriculum learning, fine-grained semantics, and a symmetric loss for model alignment. Its key contributions are a difficulty-ranked dataset of semantic preference pairs and a novel cirriculum training objective. The method is validated by dense experiments that SCPO can reduce hallucination rates compromising general performance.

### Strengths
Overall, this is a solid piece of work with the following strengths:

1. The proposed easy-to-hard curriculum learning strategy is well-designed and highly targeted. It effectively addresses the model's difficulty in effectively learning from challenging samples that are too distant from its current policy model. The three-stage learning approach is logically sound and tackles a significant practical issue in DPO. Although the methodology is relatively straightforward (simply dividing the learning into three stages), in my view, this constitutes the most important contribution of this paper.

2. The experimental section is quite comprehensive. It not only demonstrates the progressive improvement in mitigating model hallucinations across stages but also includes sufficient ablation studies. These experiments effectively validate the advantages and rationale behind the multi-stage learning approach.

### Weaknesses
1. A key limitation is the lack of innovation in the SCPO loss. It seems to be a composite of existing techniques rather than offering a novel theoretical or methodological contribution to loss function design in hallucination mitigation.
2. The "easy-to-hard" issue discussed by the authors is a very interesting topic. However, since difficulty is relative, introducing this issue inevitably raises the question of offline versus online evaluation. Although the authors employ various methods and strategies to categorize samples into easy, middle, and hard levels, determining these difficulty levels online—i.e., from the perspective of the model being optimized—would likely further enhance performance. Moreover, as the model’s capabilities improve after each of the three optimization stages, the criteria for difficulty should also evolve accordingly.

### Questions
1. The evaluation on general capability benchmarks is not sufficiently comprehensive. Beyond LLaVA-Wild and MMBench-CN, results on additional general benchmarks are required to conclusively demonstrate that the method does not harm the model's general capabilities.

2. The results in Table 6 and Figure 4 are somewhat puzzling. For instance, why does the hallucination rate increase (e.g., the "chair" metric performing worse than the original model) simply by changing the optimization order? The authors need to provide a deeper analysis and explanation for these counter-intuitive findings.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of visual hallucination in MLLMs, where model outputs contradict the visual evidence. The authors propose Semantic Curriculum Preference Optimization (SCPO), a new framework for multimodal alignment that has three novelty:
1) A Semantic Curriculum Preference Pairs (SCPP) dataset, providing fine-grained preference pairs with quantified difficulty scores.
2) A symmetric, bidirectional optimization objective combining complementary textual and visual preference losses with a cross-modal symmetry loss  to ensure consistent grounding.
3) A curriculum-based dynamic training strategy, where the model is aligned progressively from easy to hard examples, updating the reference model at each stage to avoid optimization instability.

### Strengths
1) The paper addresses an important and concrete weakness of MLLMs, visual hallucination, and lays out the limitations of existing DPO-style alignment approaches clearly.
2) The use of both symmetric optimization and curriculum learning feels natural and well motivated. The method avoids overfitting to either the language or vision side, and the easy-to-hard schedule with dynamic reference updates is well thought out.
3) The experiments are thorough. The proposed method consistently improves hallucination-related metrics on all tested models. 
4) Good ablations and analysis. The ablations on each component clearly show what matters. 
5) The paper is clear, the equations are easy to follow, and the overall structure makes sense.

### Weaknesses
1) The building blocks are not very new on their own. Semantic difficulty scoring, symmetric alignment, and curriculum learning have each been used in prior work. The novelty lies in how they are put together rather than in a single new idea.
2) Limited validation of the dataset. It is unclear whether “hard” samples are genuinely harder for humans or just confusing for the base model.
3) Narrow experimental scope. All experiments use the LLaVA family. It would be more convincing to show that SCPO also helps models with different architectures or training pipelines.

### Questions
I can't access the files in the anonymous github repo, other than the readme file.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SCPO, a preference-optimization framework for multimodal LLMs that couples (i) a semantics-aware curriculum built from a new Semantic Curriculum Preference Pairs (SCPP) dataset, (ii) a symmetric, bidirectional objective that aligns both visual and textual preferences, and (iii) iterative training with a dynamic reference model to stabilize DPO-style optimization across easy to hard stages. Experiments on LLaVA families show consistent hallucination reductions on datasets including AMBER, MMHal, and AMBER-Discriminative; the abstract claims “up to 62.9%” hallucination-rate reduction, with general capabilities preserved.

### Strengths
- **Good presentation**. The writing and presentation of contents are clear and easy to follow.
- **Well-motivated objective:** The symmetric bidirectional loss directly combats shortcut learning by enforcing both positive and negative cross-modal constraints; the formulation is clearly presented. 
- **Stability via dynamic reference:** The staged π_ref reset neatly addresses off-policy saturation/vanishing-gradient issues in later stages and admits a cumulative-reward view.  
- **Comprehensive evaluation suite** with both generative and discriminative checks and head-to-head against specialized anti-hallucination baselines.

### Weaknesses
1. **Design and Details of Difficulty score.** What is supporting the design of the difficulty scores (H, sCLIP, dOT), i.e., why these factors can measure the difficulty for comprehending the images? Besides, please provide more details on the score calculation process. It's not enough for just "using CLIP and DINOv2".
2. **No related work?** This paper **seems not have the section for related work**, which I believe is a big problem. Please add this section and discuss the related works in detail.
3. **Clarification and comparison with related work**. There is a very related and similar work with this paper - SymMPO[1], which also constructs symmetric image-text pairs for mitigating MLLM hallucinations. However, there is no citation or comparison on the preference optimization design, nor incorporated in experimental comparison as baseline. So please discuss and consider the novelty issue of the design CCO/CSO in this paper and provide experiments.
4. **New benchmarks in Experiments.** There still lack some newest benchmarks such as HallusionBench[2] and MMStar[3] in the main experiments, which will help identify the effectiveness in both generative and discriminative settings.

References:

[1] Liu et al., “Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization” In NeurIPS 2025.

[2] Guan et al., "HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models" In CVPR 2024.

[3] Chen et al., "Are we on the right way for evaluating large vision-language models?" In NeurIPS 2024.

### Questions
1. **λ scheduling.** Is λ in SCPO fixed or scheduled across curriculum stages (e.g., increase CSO weight on harder data)? Any evidence of over-regularization when λ is large? 
2. **Compute/throughput.** Please report training time, GPUs, batch sizes, and hyperparameters related.
3. **Comparability across evaluation protocols.** MMHal uses GPT-4-based assessment, which can be sensitive to prompt/aggregation choices; please document exact configs to ensure reproducibility. 
4. **“Up to 62.9%” claim.** Please anchor this to a specific model/benchmark/metric cell in the main tables (not only the abstract) and report absolute baselines alongside relative deltas.

### Soundness
2

### Presentation
2

### Contribution
2
