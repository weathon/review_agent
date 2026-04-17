# Fine-Grained Class-Conditional Distribution Balancing for Debiased Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Achieving group-robust generalization in the presence of spurious correlations remains a significant challenge, particularly when bias annotations are unavailable.
Recent studies on Class-Conditional Distribution Balancing (CCDB) reveal that spurious correlations often stem from mismatches between the class-conditional and marginal distributions of bias attributes. They achieve promising results by addressing this issue through simple distribution matching in a bias-agnostic manner. 
However, CCDB approximates each distribution using a single Gaussian, which is overly simplistic and rarely holds in real-world applications. 
To address this limitation, we propose a novel Multi-stage data-Selective reTraining strategy (MST), which describes each distribution in greater detail using the hard confusion matrix.
Building on these finer descriptions, we propose a fine-grained variant of CCDB, termed FG-CCDB, which enhances distribution matching through more precise confusion-cell-wise reweighting. FG-CCDB learns sample weights from a global perspective, effectively mitigating spurious correlations without incurring substantial storage or computational overhead.
Extensive experiments demonstrate that MST serves as a reliable proxy for ground-truth bias annotations and can be seamlessly integrated with bias-supervised methods.
Moreover, when combined with FG-CCDB, our method performs on par with bias-supervised approaches on binary classification tasks and significantly outperforms them in highly biased multi-class and multi-shortcut scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel method named Fine-Grained Class-Conditional Distribution Balancing (FG-CCDB) to mitigate spurious correlations in deep learning models under the condition of no bias annotations. The authors point out that existing methods (e.g., CCDB) model class-conditional and marginal distributions as single Gaussian distributions, which are too coarse-grained to capture the complex multi-modal structures in real-world data. To address this, they design a **Multi-stage data-Selective reTraining (MST)** strategy that leverages model overfitting behavior to identify "modes" (i.e., bias patterns) through multi-stage data filtering. Based on this, FG-CCDB performs distribution alignment at the mode level, achieving more precise sample reweighting. Experimental results show that the method can rival bias-supervised approaches in binary classification tasks and even outperform them in multi-class tasks, with low computational overhead.

### Strengths
（1）Proposed Multi-stage data-Selective reTraining (MST), which utilizes model overfitting to construct highly biased training sets through multi-stage data filtering. This generates reliable pseudo-bias labels and forms a hard confusion matrix to characterize fine-grained "mode" structures, offering a new perspective for unsupervised bias exploration.  
（2）Proposed FG-CCDB, which performs class-conditional and marginal distribution alignment at the mode level based on MST. This achieves finer and more effective distribution matching than the original CCDB, effectively mitigating spurious correlations.  
（3）The method requires no ground-truth bias annotations, is computationally efficient, and allows sample weights to be computed via closed-form solutions. It achieves superior or comparable performance to existing unsupervised and partially supervised methods on multiple binary and multi-class benchmarks, particularly in strong-bias multi-class scenarios.

### Weaknesses
（1）The definition of "mode" is dependent on specific bias model architectures and training processes, lacking semantic clarity and theoretical guarantees.  
The authors define a mode as (s, y), where s is a "pseudo-bias label" predicted by an auxiliary bias model. However, the physical meaning of s is ambiguous—it may correspond to a single shortcut, a combination of shortcuts, or even entangled uninterpretable patterns. This makes "mode" a black-box concept, with unclear mapping to latent bias factors in the data generation process. Moreover, the quality of s heavily relies on the initial ERM model's overfitting behavior. If the model fails to capture major biases, the entire MST process may collapse.  

（2）Hyperparameter choices in MST (e.g., γ=10%, top 50% high-confidence samples) lack systematic analysis and generalization guarantees.  
While the authors claim γ=10% is a "sweet spot," this conclusion seems empirical, with no sufficient ablation studies to validate its robustness across datasets or bias strengths. Similarly, selecting the top 50% high-confidence samples appears arbitrary, with no discussion of alternative proportions (e.g., 30% or 70%). These critical hyperparameters lack theoretical justification or adaptive mechanisms, reducing the method's universality and reproducibility.  

（3）The method's validity critically depends on the strong assumption that "overfitted models accurately reflect bias structures," which may not hold in complex real-world scenarios. 
The core idea is to exploit ERM models' overfitting to reveal biases. However, in real-world data, biases may be subtle or diverse, and models might overfit to noise or irrelevant features instead of true bias cues. Additionally, when multiple competing biases exist, the model may capture only one, leaving MST unable to reveal the full bias structure. While experiments show strong performance, the robustness of this assumption has not been validated in more challenging scenarios with hidden or complex biases.  

（4）Insufficient comparison with existing methods (e.g., XRM, DebiAN) to highlight FG-CCDB's core innovations. 
The paper includes XRM and DebiAN as baselines but provides brief descriptions of their mechanisms. XRM uses auxiliary models to generate pseudo-group labels, while DebiAN iteratively trains a "discoverer" model to identify biases. MST is essentially another pseudo-labeling strategy. Reviewers expect a more detailed comparison: e.g., does MST's pseudo-label quality significantly outperform XRM or DebiAN? On which bias types does MST excel? Otherwise, FG-CCDB may appear as another variant of existing pseudo-labeling + reweighting paradigms rather than a fundamental breakthrough.  

（5）The experimental evaluation lacks direct quantitative analysis of mode partitioning quality. 
While Figure 2(b) shows a heatmap of the joint distribution J and claims it reflects true biases, there are no quantitative metrics (e.g., mutual information, F1-score with ground-truth biases) to evaluate the accuracy of MST-generated mode partitions. If the mode partitioning itself contains significant errors, subsequent FG-CCDB distribution matching may amplify these errors, leading to performance degradation. The authors claim MST "serves as a strong proxy for ground-truth bias annotations," but this assertion requires stronger empirical or theoretical evidence.

### Questions
（1）FG-CCDB assumes samples within a mode are homogeneous and assigns them equal weights. However, a mode (e.g., a specific background for a class) may still have substructures or diversity. Does ignoring intra-mode heterogeneity affect reweighting effectiveness? Have you considered further subdivision within modes or introducing continuous weights?  

（2）In multi-stage MST, repeating "bias enhancement learning" up to three times improves performance. Is there a saturation point? Would further iterations cause model collapse or overfitting to noise? Could you provide performance curves across iteration counts to demonstrate convergence?  

（3）Although FG-CCDB is computationally efficient, MST requires training multiple auxiliary models, involving multiple training stages. Compared to single-training methods like ERM or uLA, how much additional training time does it incur? Is this computational cost acceptable in practical applications?

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
4

### Summary
This paper builds upon the existing Class-Conditional Distribution Balancing (CCDB) framework and proposes two key components: 
- MST (Multi-stage data-Selective reTraining): a strategy to characterize the bias structure through the hard confusion matrix, serving as a proxy for bias annotations.
- FG-CCDB (Fine-Grained Class-Conditional Distribution Balancing) - a fine-grained extension of CCDB that performs distribution alignment at the mode level, enabling a more detailed representation of multi-modal data distributions. 

The method aims to achieve annotation-free debiasing and robust generalization by modeling complex intra-class variations caused by spurious correlations. Experimental results show that MST can substitute for bias annotations in supervised baselines (e.g., GroupDRO, DFR), and FG-CCDB outperforms prior approaches in multi-class, bias-heavy scenarios.

### Strengths
- Well-structured and clearly written:
The paper is logically organized, and the presentation of ideas—from motivation to methodological formulation—is easy to follow.

- Conceptual improvement over CCDB:
By replacing the single-Gaussian assumption with a multi-modal, mode-based distribution matching framework, the paper effectively extends CCDB to more realistic data distributions.

- Novel use of confusion matrix:
Employing the confusion matrix to infer bias-aligned and bias-conflicting “modes” is elegant and intuitively appealing.
It offers a discrete, bias-agnostic way to describe intra-class spurious correlations.

- Annotation-free contribution:
The approach is promising in scenarios where human bias annotations are unavailable or infeasible, showing comparable results to bias-supervised baselines.

### Weaknesses
- Indirect validation of MST as bias substitute:
The core assumption that MST can replace human-provided bias annotations is only indirectly validated through final task performance. The paper does not report any direct quantitative measure (e.g., F1, ARI, or NMI) of how well MST’s predicted bias partitions align with human-labeled bias groups. Without this, it is unclear whether MST truly captures bias structure or simply produces partitions that happen to improve performance.

- Limited comparison with recent label-free debiasing methods:
Although the paper positions itself within the “annotation-free” literature, it lacks comparisons with the latest label-free or label-free-from-features (LFF) methods, such as those that employ generative modeling or causal data augmentation for debiasing. Including these would strengthen the empirical validity and demonstrate broader applicability.

- Experimental generality:
Most experiments are conducted on benchmark datasets (e.g., Waterbirds, CMNIST) with relatively simple, well-defined bias sources.
The performance of MST and FG-CCDB under multi-bias or entangled bias scenarios remains uncertain.

- Ablation analysis depth:
While ablations for MST and FG-CCDB are presented, the interaction between the two modules is not deeply analyzed. It is unclear how errors from MST propagate to FG-CCDB, or whether FG-CCDB can compensate for imperfect bias predictions.

### Questions
As the paper claims, can the authors experimentally demonstrate how well the MST matches human labels?
Can the authors of the paper present performance comparison results with the latest methods using 'other label free + generative model methods'?

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
This paper addresses debiased learning under spurious correlations without relying on bias annotations. It targets the problem that existing methods mitigate spurious correlations by matching class-conditional to marginal distributions but rely on overly coarse single-Gaussian approximations that fail in multi-modal, multi-class settings. The authors propose MST to exploit ERM overfitting, training on a small split, and amplifying bias via per-class top-confidence selection to derive discrete modes from the final confusion matrix. Then they propose FG-CCDB to transform this matrix into mode-wise weights that align class-conditional to marginal distributions, reducing spurious reliance with lightweight, scalable computation. Experiments that MST can effectively substitute for human bias annotations in supervised methods, and FG-CCDB consistently outperforms or matches state-of-the-art bias-agnostic baselines while maintaining low computational and memory overhead.

### Strengths
-The paper offers a thorough and compelling analysis of the limitations in prior work, clearly diagnosing CCDB’s single-Gaussian assumption and proposing an elegant, scalable remedy via mode-wise matching derived from confusion-matrix–based distributions. 
-The approach demonstrates strong practical value, showing robustness to multi-shortcut scenarios (e.g., UrbanCars) with competitive or superior performance relative to both bias-agnostic and supervised baselines.
-The experimental evaluation is comprehensive. Ablations cleanly disentangle the contributions of MST and FG-CCDB, and the correlation-shift analyses substantiate the mechanism that the method reduces reliance on bias-related features.

### Weaknesses
-The proposed mechanism lacks theoretical grounding. There is no formal analysis of how iterative bias amplification improves minority-mode recall. While the empirical evidence is compelling, a theoretical treatment would substantially strengthen the contribution.
-Several methodological details require further clarification. The selection of top-confidence samples hinges on the biased model’s calibration. Miscalibration could distort mode discovery, yet no temperature scaling or calibration baseline is reported.

### Questions
Why fix the top-50% per-class in bias enhancement? Did you explore adaptive thresholds (e.g., based on class-wise confidence distributions, entropy) or rank-based schedules across iterations? Any results on temperature scaling to improve confidence reliability?

### Soundness
3

### Presentation
2

### Contribution
3
