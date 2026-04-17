# ProteinZero: Self-Improving Protein Generation via Online Reinforcement Learning

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Protein generative models have shown remarkable promise in protein design, yet their success rates remain constrained by reliance on curated sequence-structure datasets and by misalignment between supervised objectives and real design goals. We present ProteinZero, an online reinforcement learning framework for inverse folding models that enables scalable, automated, and continuous self-improvement with computationally efficient feedback. ProteinZero employs a reward pipeline that combines structural guidance from ESMFold with a novel self-derived ddG predictor, providing stable multi-objective signals while avoiding the prohibitive cost of physics-based methods. To ensure robustness in online RL, we further introduce a novel embedding-level diversity regularizer that mitigates mode collapse and promotes functionally meaningful sequence variation. Within a general RL formulation balancing multi-reward optimization, KL-divergence from a reference model, and diversity regularization, ProteinZero achieves robust improvements across designability, stability, recovery, and diversity. On the CATH-4.3 benchmark, it consistently outperforms state-of-the-art baselines including ProteinMPNN, ESM-IF, and InstructPLM, reducing design failure rates by 36-48\% and achieving success rates above 90\% across diverse folds. Importantly, a complete RL run can be executed on a single 8$\times$GPU node within three days, including reward computation and data generation. These results indicate that efficient online RL fine-tuning can complement supervised pretraining by allowing protein generative models to evolve continuously from their own outputs and optimize multiple design objectives without labeled data, opening new possibilities for exploring the vast protein design space.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces ProteinZero, an online Reinforcement Learning (RL) fine-tuning framework for inverse protein folding models. The goal is to optimize multi-objective protein design metrics. Experiments on the CATH-4.3 benchmark demonstrate significant improvements over state-of-the-art baselines like ProteinMPNN.

### Strengths
**Originality**: I like the idea of an online RL Framework for this problem. Furthermore, the authors incorporate multiple novel elements to the architecture, such as the ddG predicted, and the regulizer

**Quality**: The results in the experiment section are very impressive, and carefully checked, through extra steps like the AlphaFold3 validation

**Clarity**: The paper is very well explained. The text is well organized and clear.

**Significance**: It is important to improve inverse folding, as the community has been reliant on PMPNN for a very long time. The insights into RLHF in general are also very interesting.

### Weaknesses
- The structural designability reward relies solely on ESMFold. Although this is briefly discussed in the paper, and external validation is provided, over-optimization toward the biases or error modes of a single folding model remains a subtle risk. It would be interesting to add multiple oracles, to prevent such biases.
- While the model is very interesting, the limitation of being used only for monomers, does impact its usefulness for drug design. 
- There is a growing trend in the field towards fully atomistic 3D design, as opposed to backbone design, which overpasses the need for inverse folding. Of course, this does not mean inverse folding is not valuable (SOTA protein design models still design backbones) but the existence of full-atom design models is not even mentioned in the paper. 
- It would have been great to see a comparison of quality metrics for a 3D backbone design model (like RFDiffusion or FoldFlow), doing inverse folding with this method, or PMPNN; then folding and comparing true and predicted structures. This is, as far as I know, the main usage of inverse folding models, so it would have been good to see this model put to use in that way.

### Questions
- On the first point of the weaknesses, while it is unrealistic to ask the authors to add multiple oracles during the review period, it would be great to see some discussion about this in the paper, and even better to get an estimate of the computational cost of such an addition.
- What is the correlation between your Fast-ddG predictions and actual experimental stability measurements?
- Why specifically use cosine similarity in embedding space rather than other distance metrics?
- How sensitive is performance to the choice of $\alpha_{div}$?
- You report means $\pm$ standard deviations, but over how many runs?
- Why are other RL methods such as ResiDPO not include in the experiments?

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
4

### Summary
This paper introduces ProteinZero, an online RL framework for improving inverse folding using ESMFold self-consistency and a ddG predictor as rewards. Employs embedding-level diversity regularization to prevent mode collapse and analyzes RL design space.

### Strengths
1. Many prior protein foundation model + RL post-training works focus on offline RL, but online RL makes more sense for tasks with reliable reward functions (e.g., self-consistency). I strongly support exploring online RL.
2. Strong results in designability metrics. Experiments are thorough and well-executed. Extensive ablation on reward functions and RL algorithms (offline vs. online).

### Weaknesses
1. Thermal stability reward is interesting but a bit unclear to me in both formulation and interpretation.
    1. Validation: There’s no analysis showing how well the proposed reward correlates with true ddG values. 
    2. Formulation: Why introduce two models $p_\theta$ and $p_\varphi$ (prior) instead of using same model (like $\log p_\theta(y|x) - \log p_\theta(y)$ or $\log p_\varphi(y|x) - \log p_\varphi(y)$? Having both $p_\theta$ and $p_\varphi$ seems to make the optimization more complex (or even inaccurate) without clear justification. The paper mentions (line 227) that "backbone-conditioned likelihoods reflect folding stability," implying that $\log p_\varphi(y|x) - \log p_\varphi(y)$ can do the same job? 
    3. Interpretation: The ddG improvement from the base InstructPLM (Table 1) looks minimal. Given that the objective somewhat resembles DPO, I wonder if this formulation is truly improving stability or just adding some kind of a regularization effect.
2. I have a more fundamental question—Do we actually want to minimize ddG, or do we want to match the target (wild-type) ddG? Accurate inverse folding and stabilization feel like slightly different objectives. Both matter: finding stable, well-expressing sequences is important since all-atom generation and sequence co-design are still imperfect. But if the model is biased toward minimizing ddG too aggressively, it might fail for naturally flexible proteins where some instability is functional.
    1. What is the ddG of wild-type structures? Could you add that as a reference row in Table 1?
3. Can you provide some related works or context for the diversity regularization part for the readers? I like the embedding-level diversity idea, but I’m curious what similar approaches have been explored in the broader RLHF literature (not protein, I saw authors mentioned hamming distance idea in Line 143-144) for preventing mode collapse or promoting output diversity.

### Questions
See Weaknesses.

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
3

### Summary
ProteinZero introduces a novel online Reinforcement Learning (RL) framework designed for the continuous self-improvement of protein inverse folding models. To achieve scalable optimization, the methodology employs efficient proxy reward models, including an ESMFold-based TM-score for fast structural fidelity assessment and a fast ΔΔG (ddG) predictor for thermodynamic stability. The authors demonstrate a technically sound approach to multi-objective optimization, effectively balancing conflicting goals such as designability, stability, KL-regularization, and sequence diversity. The framework shows compelling computational improvements on the CATH-4.3 benchmark.

### Strengths
ProteinZero's primary strength lies in its innovative concept of linking protein generation with self-improvement via online Reinforcement Learning. This paradigm enables continuous model iteration, effectively overcoming the limitations of static, supervised training data. The methodology demonstrates diverse and practical attempts within the online RL framework itself, including a novel embedding-level diversity regularizer, and features a well-designed approach to multi-objective optimization effectively balancing designability, stability, KL-regularization, and diversity. Furthermore, the framework exhibits efficient execution and scalability; specifically, the use of the ESMFold + TM-score proxy reward makes the RL training loop computationally feasible while demonstrating strong performance gains on the computational benchmark environment.

### Weaknesses
1. Computational predictors like Pred-ddG and FoldX ΔΔG inherently have limited correlation with ground-truth wet-lab experimental stability metrics. Whether used as a reward model for optimization or as a metric for evaluation, reliance on these tools introduces an unavoidable systemic bias, meaning the model may optimize for computational artifacts rather than true biophysical stability.

2. The paper's review of related work is incomplete, which complicates the assessment of the necessity and unique contributions of ProteinZero. The application of online RL for optimizing protein generation has been explored previously [1].The methodology and formal approach of these prior RL works appear conceptually very similar. Therefore, a comprehensive review and comparison to such existing RL-based efforts are required.

[1] Wang C, Uehara M, He Y, et al. Fine-tuning discrete diffusion models via reward optimization with applications to dna and protein design[J]. arXiv preprint arXiv:2410.13643, 2024.

### Questions
The content in this section is identical to the points raised in Weaknesses. I would be very willing to increase my score if the authors successfully address these concerns.

### Soundness
2

### Presentation
3

### Contribution
2
