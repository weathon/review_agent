# Towards Real-world Debiasing: Rethinking Evaluation, Challenge, and Solution

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Spurious correlations in training data significantly hinder the generalization capability of machine learning models when faced with distribution shifts, leading to the proposition of numberous debiasing methods. However, it remains to be asked: Do existing benchmarks for debiasing really represent biases in the real world? Recent works attempt to address such concerns by sampling from real-world data (instead of synthesizing) according to some predefined biased distributions to ensure the realism of individual samples. However, the realism of the biased distribution is more critical yet challenging and underexplored due to the complexity of real-world bias distributions. To tackle the problem, we propose a fine-grained framework for analyzing biased distributions, based on which we empirically and theoretically identify key characteristics of biased distributions in the real world that are poorly represented by existing benchmarks. Towards applicable debiasing in the real world, we further introduce two novel real-world-inspired biases to bridge this gap and build a systematic evaluation framework for real-world debiasing, RDBench. Furthermore, focusing on the practical setting of debiasing w/o bias label, we find real-world biases pose a novel Sparse bias capturing challenge to the existing paradigm. We propose a simple yet effective approach named Debias in Destruction (DiD), to address the challenge, whose effectiveness is validated with extensive experiments on 8 datasets of various biased distributions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper scrutinises whether current debiasing benchmarks faithfully reflect the complexity of real-world biases. The authors (i) present a fine-grained analytic framework that decomposes bias into “magnitude” and “prevalence”, (ii) empirically and theoretically demonstrate that real-world data sets often exhibit low-magnitude and low-prevalence biases—properties missing from popular benchmarks, (iii) introduce two new bias types and assemble them (plus existing data sets) into a systematic evaluation suite called RDBench, and (iv) identify a “sparse-bias-capturing” challenge when debiasing without bias labels. To tackle this they propose a simple method, Debias-in-Destruction (DiD), which first “destroys” dominant features and then reconstructs, leading to gains on eight data sets across several bias settings.

### Strengths
S1. Important problem – The work challenges a widespread, implicit assumption (high-prevalence biases) and argues convincingly that it is misaligned with reality.
S2. Novel analytical perspective – The magnitude / prevalence decomposition offers a clear, quantitative lens through which to study bias distributions.
S3. New benchmark – RDBench fills a gap by providing real-world-inspired biases and multi-bias scenarios; releasing code/data will benefit the community.
S4. Practical focus – Concentrating on “debiasing without bias labels” increases the paper’s relevance for industry deployment where bias attributes are rarely annotated.
S5. Methodological contribution – DiD is conceptually simple, easy to integrate into existing pipelines, yet yields consistent improvements.
S6. Thorough experiments – Eight data sets, multiple baselines, ablations, and both empirical and theoretical analyses lend credibility to the claims.

### Weaknesses
W1. Scope limited to image classification – All studied tasks are visual. It is unclear whether the magnitude/prevalence findings (and DiD) generalise to NLP or multimodal settings.
W2. “Real-world inspired” still partly synthetic – The two proposed biases are constructed heuristically; evidence that they mirror true large-scale natural distributions (e.g., via quantitative fitting or user studies) is thin.
W3. Cost to “clean” accuracy – Results mainly highlight robustness under bias; corresponding drops on i.i.d. test sets are not fully reported. Practical users need to understand this trade-off.
W4. Hyper-parameter robustness – DiD introduces new knobs (destruction ratio, masking schedule). Limited analysis is given on sensitivity and tuning without bias labels.
W5. Reproducibility details – Key implementation aspects (random seeds, data split scripts, destruction operator specifics) are relegated to the supplement; including them in main paper would strengthen reproducibility.

### Questions
Q1. How exactly were the two new bias distributions designed? Do you have quantitative evidence showing their closeness to real-world statistics?
Q2. Can the magnitude/prevalence metrics be automatically computed on arbitrary data sets? If yes, will you release a toolkit?
Q3. How does DiD perform when combined with other strong debiasing methods (e.g., GroupDRO, JTT) on RDBench? Are the gains additive?
Q4. What is the computational overhead (training time, memory) introduced by DiD compared with vanilla ERM?
Q5. Have you analysed failure cases where DiD hurts both biased and unbiased accuracy? Understanding such cases would be useful for practitioners.

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
This paper argues that existing debiasing benchmarks fail to reflect the true nature of biases found in real-world data. The authors introduce a fine-grained framework to analyse bias magnitude and prevalence, revealing that real-world biases are typically low and sparse, unlike the strong correlations assumed in synthetic datasets. They further propose RDBench, a systematic evaluation framework, and Debias-in-Destruction (DiD), a simple yet effective method that enhances bias capture under sparse, low-prevalence settings.

### Strengths
(1) Presents a comprehensive empirical and theoretical analysis of real-world bias distributions, introducing the RDBench framework that provides a systematic and realistic benchmark for evaluating debiasing methods.

(2) Proposes a simple yet effective Debias-in-Destruction (DiD) approach that generalizes well across multiple datasets and modalities, demonstrating strong improvements over existing debiasing methods.

### Weaknesses
(1) The clarity of theoretical exposition could be improved, especially regarding assumptions and proofs.

(2) Evaluation on large-scale, high-dimensional real-world data (e.g., complex vision-language models) remains limited.

(3) The DiD method’s simplicity, while appealing, may lack interpretability and deeper theoretical grounding.

(4) Some parts of the framework (e.g., threshold selection for bias magnitude/prevalence) rely on heuristics rather than principled estimation.

### Questions
None

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
4

### Summary
This paper investigates the gap between existing debiasing benchmarks and real-world biased distributions. The authors propose a fine-grained framework for bias analysis that distinguishes between bias magnitude and bias prevalence. Based on empirical and theoretical insights from real-world datasets, the authors introduce RDBench, a new benchmark for realistic bias evaluation, and a simple yet effective debiasing method called Debias in Destruction (DiD). DiD is designed to handle “sparse bias” scenarios, especially under the practical setting of debiasing without bias labels.

### Strengths
- Novel problem framing and fine-grained bias analysis
The paper raises an important question about whether current benchmarks truly represent real-world biases. By distinguishing bias magnitude (how strong the spurious correlation is) and bias prevalence (how common it is in the dataset), the authors provide a meaningful and interpretable framework for bias characterization. This fine-grained view can serve as a strong foundation for future benchmark design. 

- Realistic motivation for bias-agnostic debiasing
The authors focus on debiasing without explicit bias labels, a practically relevant and challenging setting in real-world applications (e.g., MS COCO, COMPAS). The discussion on the limitations of existing auxiliary-model-based approaches (e.g., Nam et al., 2020; Lee et al., 2021) is insightful. 

- Simplicity and generality of the proposed DiD method
DiD is conceptually simple, can be easily integrated into existing methods, and empirically improves performance across several benchmark tasks (LfF, DisEnt, BEL, BED, etc.). The consistent improvements across multiple bias setups demonstrate its general applicability.

- Comprehensive literature grounding
The authors provide an extensive review of existing bias-agnostic debiasing methods, clearly situating their contribution within the field.

### Weaknesses
- Lack of validation on real-world datasets (MS COCO, COMPAS)
Although the introduction emphasizes real-world bias distributions and repeatedly mentions datasets such as MS COCO (for vision) and COMPAS (for fairness in tabular domains), the actual experiments are limited to synthetic or semi-synthetic settings such as Colored MNIST or Corrupted CIFAR-10 (referred to as HMLP BC). The absence of evaluation on these real datasets undermines the claim that DiD or RDBench effectively handles real-world biases.

- Dependence on prior bias knowledge for bias magnitude estimation
Similar to many previous debiasing works, the computation of bias magnitude (Equation 1) assumes that the spurious attribute (or biased feature) is known a priori. This contradicts the fully unsupervised debiasing objective and limits the applicability in settings where such bias attributes are unknown or latent.

- Limited empirical diversity and scalability
The reported experiments (e.g., HMLP BC) are confined to small-scale benchmarks with controlled bias patterns. It remains unclear whether DiD can scale to multimodal or large-scale datasets such as MS COCO or social datasets like COMPAS, which involve complex, overlapping biases.

- Missing discussion on recent relevant works
The paper omits several closely related and contemporaneous studies, such as “Debiasing Classifiers by Amplifying Bias with Latent Diffusion and Large Language Models” (2025), which similarly address real-world bias modeling using diffusion-based augmentation. Including comparisons or conceptual distinctions from these methods would strengthen the paper’s positioning.

- Ambiguity in benchmark naming and description (e.g., HMLP BC)
Some terms such as HMLP BC are not clearly defined or are introduced without detailed explanation of their dataset composition, making reproducibility difficult.

### Questions
ms coco, is there a reason COMPAS wasn't included in the main results or additional experiments?
Can you provide a complexity analysis of the proposed method when implemented in practice?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the assumption that training datasets exhibit severe biases affecting nearly all samples (over than 95%). Analysis of MSCOCO and COMPAS datasets reveals real-world biases are sparse (about 8~15% prevalence) with scattered patterns, contrasting with the diagonal patterns of existing benchmarks. 

Key proposals include the followings:

(1) Fine-grained Metrics: Bias Magnitude (KL Divergence) and Bias Prevalence (proportion of biased samples).

(2) Bias Neutral Category: Expands the Bias aligned/Bias Conflict dichotomy for samples lacking biased features.

(3) Theoretical validation: Propositions 1~2 demonstrate high prevalence distributions require unrealistic matched and uniform marginals, unsupported in reality.

(4) DiD Method: Destroys target features during bias model training, ensuring low loss for bias aligned and high loss for bias neutral and bias conflict by capturing only spurious correlations.

Som this paper mainly claim that existing methods falter on low prevalence real-world data due to misweighting abundant bias neutral samples, a problem addressed by the proposed method DiD.

### Strengths
Strengths are summarized as follows:

(1) Clear problem framing: Figure 1 contrasts diagonal benchmark patterns with scattered real-world biases.

(2) Rigorous theory: Propositions 1 and 2 justify the sparsity of real-world biases mathematically.

(3) Comprehensive experiments: Covers 8 datasets (vision + NLP benchmarks), 9 baselines (e.g., LfF, DisEnt, BEL), multiple bias types.

### Weaknesses
** Critical Weaknesses

(1) Limited real-world evidence: Detailed analysis only for COCO and COMPAS datasets.  CelebA, MultiNLI, and CCW are minimally discussed, appearing mainly in Figure 2 and Appendix. CelebA is real-world, but coverage is limited; medical and social media domains are only motivationally mentioned. Core experiments likely rely on synthetic datasets (Colored MNIST and Corrupted CIFAR-10)

(2) Experimental design flaw: LMLP with threshold 0 result in 0% BN samples, which contradicts the focus on BN prevalence. Only HMLP are LMLP examine the hypothesis.

(3) Unexplained Performance Variation: Table 1 shows improvement from -0.8 to 32.6 across datasets with the proposed algorithm DiD. The paper lacks a predictive model, quantitative feature complexity metric, or failure analysis.

(4) Theory practice gap: Propositions assume binary attributes, while experiments use 10-class problems without a multi-class extension.

(5) Incomplete coverage of related work: Although the paper discusses many relevant studies, several important papers and comparison baselines are missing - for example, PGD [1]

[1] Mitigating dataset bias by using per-sample gradient, ICLR 2023

### Questions
Please refer to the Weaknesses section, which includes my main questions and concerns about the paper.

### Soundness
2

### Presentation
2

### Contribution
2
