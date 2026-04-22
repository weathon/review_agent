# FoSSIL: A Unified Framework for Continual Semantic Segmentation in 2D and 3D Domains

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Evolving visual environments challenge continual semantic segmentation by
introducing the complexities of class-incremental learning, domain-incremental
learning, limiting available annotations, and necessitating the use of unlabeled data.
In this work, we present the framework FoSSIL (Few-shot Semantic Segmentation
for Incremental Learning), which extensively benchmarks continual semantic
segmentation, spanning both 2D natural scenes and 3D medical volumes. Our
evaluation encompasses diverse and realistic settings, leveraging both labeled
(few-shot) and unlabeled data. Building on this benchmark, we introduce
guided noise injection to mitigate overfitting due to novel few-shot classes
from various domains. Furthermore, we leverage semi-supervised learning
for unlabeled data to augment few-shot novel classes. We propose a filtering
mechanism to remove highly confident but incorrectly predicted pseudo-labels,
further improving performance. Results across class-incremental, few-shot, and
domain-incremental scenarios with unlabeled data validate our strategies for
robust semantic segmentation in complex, evolving settings, highlighting both
the effectiveness and generality of our approach. Our findings illustrate that the
proposed framework forms a simple yet powerful recipe for continual semantic
segmentation in dynamic real-world environments. Our large-scale benchmarking
across natural 2D and medical 3D domains exposes key failure modes of existing
methods and offers a roadmap for building robust continual segmentation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents FOSSIL, a unified framework for continual semantic segmentation that simultaneously addresses class-incremental (CIL), domain-incremental (DIL), and few-shot challenges. The method introduces Guided Noise Injection (GNI) to mitigate overfitting in few-shot settings and Prototype-guided Pseudo-label Refinement (PLR) to effectively leverage unlabeled data. Experiments on a new, comprehensive 2D/3D benchmark show FOSSIL significantly outperforms existing methods.

### Strengths
1. The paper tackles a highly practical and challenging problem by unifying CIL, DIL, and few-shot learning. This problem formulation is novel and critical for real-world applications like autonomous driving and medical analysis. 
2. The validation is extensive, testing against various methods on diverse backbones. The ablation studies in Figure 5 clearly demonstrate the necessity and contribution of both GNI and PLR.
3. The paper is well-written, with a clear problem definition , methodology, and analysis, making it easy to follow.

### Weaknesses
1. This principle of using gradient statistics to modulate network parameters seems not new. In my opinion, it is the core mechanic in adaptive optimizers (e.g., Adam, RMSProp), which use squared gradients to normalize learning rates. While GNI's application differs from optimization, I think the underlying concept is related. The paper needs to provide a discussion comparing GNI to other adaptive regularization schemes or existing noise injection methods (e.g., standard weight decay, dropout, or variational dropout) and justify why this specific gradient-based formulation is superior.
2. The paper lacks sensitivity analysis for the thresholds in PLR ($\tau_{conf}$, $\tau_{sim}$). It is unclear how these were "empirically determined" or how robust the model is to their variation.
3. GNI is only applied to the final classifier layer $F$. The paper needs to justify why it isn't applied to deeper feature extractor layers, which are also prone to overfitting in few-shot settings.

### Questions
I hope the authors can address the issues I raised in Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FoSSIL, a unified framework and benchmark for continual semantic segmentation across 2D natural scenes and 3D medical volumes. It targets realistic settings that combine class-incremental (CIL) and domain-incremental (DIL) shifts under few-shot supervision, optionally augmented with unlabeled data. The authors also release five challenging benchmarks (three 3D-medical, two 2D-driving) with multi-session protocols and scarce labels per session.

### Strengths
This paper has the following strengths:

- The paper ambitiously unifies several learning paradigms - continual, few-shot, and semi-supervised segmentation - under one benchmark suite. 

- The proposed benchmarks (Med FoSSIL and Natural FoSSIL) cover realistic multi-session setups and will likely be valuable to the community.  

- The evaluation is broad, including 25+ baselines, multiple backbones (U-Net, DeepLabv3+, SwinUNetr, SAM), and detailed ablations on proposed modules.  

- The paper is generally well organized, with mathematical formulations that are easy to follow.

### Weaknesses
I have recognize these cons:

- The core mechanisms - prototype replay, noise-based regularization, and pseudo-label filtering - are adaptations of known techniques. The contribution is mainly in *integration* and *benchmarking*, not in introducing fundamentally new algorithms.

- The paper does not analyze *why* the guided noise injection helps beyond empirical performance. No theoretical link to stability–plasticity balance is made.

- Some baselines were not originally designed for few-shot or semi-supervised continual segmentation (e.g., MiB, MDIL), which could exaggerate FoSSIL’s relative advantage.

- Details of data splits, unlabeled data sampling, and hyperparameter tuning are missing. For a benchmark paper, this is a major weakness.

- The work lacks qualitative examples or discussions of cases where FoSSIL fails, such as severe domain shifts or noisy unlabeled data.  

- The paper repeats motivation and design explanations across sections, and some figures (e.g., Figure 2–4) are not deeply analyzed.

### Questions
Some questions should be answered:

1. How is guided noise injection different from existing gradient-based regularization (e.g., SAM, weight perturbation)?  

2. Are prototypes recomputed from all sessions or updated incrementally?  

3. What is the additional computational overhead compared to vanilla training?

### Soundness
3

### Presentation
3

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
This paper tackles continual semantic segmentation in a very real-world way: data arrives in sessions, with a fully supervised base session and then a stream of few-shot increments that either introduce new classes, shift the domain, or both, while unlabeled data is abundant. The proposed FoSSIL framework keeps old knowledge without storing raw images by replaying compact class prototypes, steadies training with a gradient-guided noise injection scheme to curb few-shot overfitting, and makes semi-supervision actually work by filtering pseudo-labels using both confidence and prototype consistency in a mean-teacher setup.

### Strengths
1. The paper directly tackles continual semantic segmentation across CIL, DIL, and few-shot learning in one framework. FoSSIL addresses this limitation and leverages unlabeled data to augment scarce few-shot classes.
2. Extensive evidence across 3D medical and 2D autonomous driving benchmarks, with multi-session protocols, shows consistent gains and robustness over strong baselines.
3. FoSSIL integrates cleanly with diverse backbones, indicating strong architecture-agnostic generalization rather than narrow tuning.
4. This paper proposes a readily deployable solution to real-world deployment pain points—privacy, limited annotations, and domain shift.

### Weaknesses
1. The paper lacks a clear pipeline/architecture diagram, which would make the method easier to grasp at a glance.
2. The noise injection module has no analysis of hyperparameters or other strategies (e.g., sensitivity and robustness studies).
3. In several tables (Table 2, Table 4, Table 6, Table 7), multiple methods report identical Session-0 results; the authors should explain why.
4. In Table 3, FoSSIL (U-Net) drops to 0.025 at Session 4 and then rebounds to 0.324 at Session 5; this large fluctuation should be verified (typesetting/statistics) or clearly explained.
5. The paper should report computational costs, including training/inference time and memory/parameter overhead.
6. Although FoSSIL improves over multiple sessions, the absolute Dice/IoU remains low, which may limit practical applicability; this seems at odds with the paper’s motivation and should be discussed.
7. How are τ_conf and τ_sim selected/tuned? Are they shared across datasets? Please provide threshold sensitivity and retention/coverage statistics.

### Questions
Please refer to the points listed under Weaknesses; if the authors can satisfactorily address these concerns, I will consider raising my score.

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
4

### Summary
This paper tries to study continual learning in a complex setting that includes class-incremental, domain-incremental, and few-shot learning. To this end, the authors build a benchmark and propose a method.

### Strengths
Pros:
- Continual learning is indeed an important topic in the field.
- The paper is well written and nicely organized. I can tell the authors put a lot of effort into polishing it, and I appreciate that.
- The experiments cover a wide range of methods, which is nice to see.

### Weaknesses
Cons:
- Honestly, I don't really like this kind of work. It feels like a mixture of everything — several settings thrown together without a clear focus. While this may have some meaning academically, in real-world scenarios (e.g., in industry), people usually prefer to train a specialized model rather than deal with such a complicated continual setup.
- Even if we accept the setting, the experimental analysis is not very systematic. The paper doesn't really explore how different setups affect different models, nor does it provide useful insights for choosing models in practice.
- The paper starts with a quote from Confucius: "I hear and I forget. I see and I remember. I do and I understand." Who is Confucius in this context? Is he a machine learning expert? I don't quite get the connection between this quote and the paper's content.
- The introduction could be smoother in logic. For example, when guided noise injection first appears, I didn't understand why it suddenly shows up or how it relates to the context. Please consider improving the narrative flow there.
- As far as I know, many important medical imaging modalities are 2D. Why aren't they included in the experiments?
- I appreciate that the authors included many methods in the benchmark, but they’re mostly from 2020–2023. There's no mention of newer work (2024–2025), including recent arXiv papers. This weakens the experimental conclusions to some extent.
- Minor issues: e.g., in the introduction, the numbering goes (i)–(ii)–(ii)–(iii), which should be fixed.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
