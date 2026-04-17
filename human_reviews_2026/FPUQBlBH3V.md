# Canonical Latent Representations in Conditional Diffusion Models

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Conditional diffusion models (CDMs) have shown impressive performance across a range of generative tasks. Their ability to model the full data distribution has opened new avenues for analysis-by-synthesis in downstream discriminative learning. However, this same modeling capacity causes CDMs to entangle the class-defining features with irrelevant context, posing challenges to extracting robust and interpretable representations. To this end, we introduce Canonical Latent Representation Identifier (CLARID), a training-free procedure to identify Canonical Latent Representations (CanoReps), latent codes whose internal CDM features preserve essential categorical information while discarding non-discriminative signals. When decoded, CanoReps produce representative samples for each class, offering an interpretable and compact summary of the core class semantics with minimal irrelevant details. Exploiting CanoReps, we develop a novel diffusion-based feature distillation paradigm, CaDistill. While the student has full access to the training set, the CDM as teacher transfers core class knowledge only via CanoReps, which amounts to merely 10% of the training data in size. After training, the student achieves strong adversarial robustness and generalization ability, focusing more on the class signals instead of spurious background cues. Our findings suggest that CDMs can serve not just as image generators but also as compact, interpretable teachers that can drive robust representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CanoRep, a latent code within conditional diffusion models (CDMs) that preserves only the core class semantics. Through a training-free procedure called CLARID, the method estimates extraneous (non-discriminative) directions in the latent variable obtained during reverse diffusion, using the right singular vectors (Jacobian’s right singular vectors). These components are then removed via orthogonal projection to obtain the CanoRep (Eq. (2)). When decoded, CanoReps generate Canonical Samples, which are representative examples with minimal irrelevant context. From these, Canonical Features are extracted and used in the student model for CaDistill, which optimizes three objectives: Alignment loss, CanoRep clustering loss, and CKA-based representation alignment. Experimentally, even when using only 10% of the dataset as CanoReps on CIFAR-10 and ImageNet, the method demonstrates improved robustness and generalization.

### Strengths
- Clear problem definition & concise solution:
The paper clearly identifies the issue that CDMs (Conditional Diffusion Models) encode both class-relevant and class-irrelevant information. It proposes a simple yet general solution — extracting and removing non-discriminative directions via a Jacobian-based linear approximation. The method is broadly applicable regardless of architecture or sampler, and visualizations demonstrate that it conceptually works even under text conditioning and across various samplers/architectures (e.g., DDIM, EDM, U-ViT).

- Improved representation quality:
Using CanoRep improves representation quality, as evidenced by higher NMI scores and consistently more compact and well-separated clusters in UMAP visualizations (Figures 3 and 4). The method also visually reduces class-irrelevant artifacts compared to CFG (Figure 5).

- Practical benefits & data efficiency:
CaDistill achieves consistent improvements in both adversarial robustness (PGD, CW, APGD) and generalization (ImageNet-C, ImageNet-A, ReaL), even when using only 10% of the data as CanoReps (Table 1). In background perturbation experiments, it also shows advantages when the background is randomized or only the foreground is preserved (Table 2).

### Weaknesses
- **No theoretical analysis**: Fundamentally, this study considers the separation between class-relevant and class-irrelevant latent information. I believe that research of this kind would benefit from an information-theoretical explanation (See [1] for fundamentals). In relation to diffusion models, [2] provides an information-theoretic analysis of how latent information is partitioned. It would be helpful if the authors could include a discussion on this aspect, for example, by explaining the changes in mutual information.

- **Experimental scale**: It seems that using a T2I (text-to-image) diffusion model could also improve zero-shot discriminative tasks. I recommend including such experiments as well.

- Too many hyperparameters appear in Eq.6.

[1] (NeurIPS 18) Isolating Sources of Disentanglement in VAEs

[2] (ICLR 25) Diffusion Bridge AutoEncoders for Unsupervised Representation Learning

### Questions
See Weaknesses.

### Soundness
2

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
This paper proposes CLARID (Canonical Latent Representations in Conditional Diffusion Models), a framework designed to extract latent representations of data instances in conditional diffusion models. CLARID identifies directions of largest variation within images of the same class at a specific layer of the diffusion model using the right singular vectors of the Jacobian, and then removes those directions to retain the shared, class-common information as the latent representation.
Building upon these latent representations, the authors further propose CaDistill, a student-teacher training method that leverages canonical samples to obtain robust and generalized representations.

### Strengths
- The method introduces a simple yet effective idea: by utilizing the right singular matrix of the Jacobian, it identifies directions that cause significant local variation on the manifold and removes them, thereby isolating class-common representations.
The approach demonstrates improved adversarial robustness and resilience to distribution shift, showing consistent performance gains over baselines on both CIFAR-10 and ImageNet datasets. 

- In addition, this paper demonstrates that the proposed method effectively removes background information and other class-irrelevant factors (e.g., the background context in the tench class of ImageNet), and this effect is clearly reflected in quantitative metrics.
Furthermore, the latent representations obtained by this method can be directly utilized for sampling in the base diffusion model, which constitutes one of its key advantages.

### Weaknesses
- Numerous hyperparameters:
The overall framework appears to be sensitive to hyperparameter choices. For instance, the process of obtaining Canonical Representations (CanoReps) depends on parameters such as $k$, $l$, and $t_{e}$; moreover, CaDistill involves additional hyperparameters ($\lambda_{align}, \lambda_{cano}, \lambda_{dist}$) during training.
This raises concerns about the practical applicability and generalizability of the proposed method, as the optimal combination of hyperparameters may vary for different base diffusion models, requiring extensive tuning for each setup.

- According to Appendix E, the parameter $l$ seems to be fixed. However, since l may vary with $t_e$, it would be helpful to see results where $k$ is fixed but $t_e$ is varied, showing how the corresponding $l$ changes and affects the outcome.

- It would also be interesting to investigate class-wise robustness. For example, showing the variation of top-$k$ directions across classes or presenting classification performance of the student model for different classes could strengthen the analysis.

- Regarding data fidelity, the paper states that CFG is applied after removing extraneous directions. It would be important to clarify whether the CFG scale used is greater than 1.
If it is, then it becomes difficult to disentangle whether the improvement in representation quality comes from the proposed method (with student network) itself or simply from a stronger CFG scale. Additional experiments verifying this would help clarify the contribution of CLARID.

### Questions
Please refer to the points raised in the Weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to address the issue that conditional diffusion models often entangle class-defining features with irrelevant contextual information. The authors propose a training-free procedure to identify canonical latent representations and further introduce a diffusion-based feature distillation method to enhance the adversarial robustness and generalization ability of a student model.

### Strengths
1. The problem of extracting robust and interpretable representations in conditional diffusion models is important and timely. 
2. The proposed pipeline is conceptually sound: it first identifies canonical latent representations that retain essential categorical information while removing non-discriminative signals, and then leverages these representations to guide the training of a student model with improved adversarial robustness and generalization. 
3. Experiments on real-world image datasets demonstrate the effectiveness of both the canonical representation extraction and the subsequent distillation process.

### Weaknesses
1. The writing and structure of the manuscript could be improved. While the technical content is substantial, the dense presentation and frequent deferrals to the Appendix (e.g., Lines 254–269, 308–320) make the paper difficult to follow. A clearer exposition of the key steps in the main text would improve readability. 
2. Experimental settings are insufficiently described. For instance, it is unclear how the “20 different ImageNet classes” (Line 259) were selected. Providing a rationale for this choice and clarifying the data preprocessing or sampling procedure would strengthen reproducibility. If class selection was random, multiple runs should be conducted and the variance reported (e.g., in Figure 3) to support claims of robustness and generalization.
3. The ImageNet classes used across different subsections are inconsistent, making it difficult to assess the generality of the findings.

### Questions
Please refer to "Weaknesses".

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to improve downstream discriminative learning in conditional diffusion models by extracting robust and interpretable representations. To this end, the paper proposes CLARID, a training-free method for obtaining CanoReps, latent representations that capture conditional information while suppressing irrelvant signals. Through both quantitative and qualitative analyses, the paper demonstrates that these latent representations contain compact, claa-relevant infomration without encoding unnecessary background or non-class features. Building on this, the paper further propose CaDistill, a knowledge distillation framework that leverages these representations.

### Strengths
* The proposed approach of projecting onto otrhogonal directions using the singular vectors of Jacobian is a reasonable and elegant idea. It is also a novel contribution within the context of diffusion-based representation learning for enhancing class-discriminative information.

* The observed performance improvement when applying the method to knowledge distillation demonstrates its practical applicability.

### Weaknesses
* It would be beneficial to provide a clearer methodological or theoretical discussion on why CanoReps work effectively for feature distillation. It remains somewhat unclear why CanoReps sould be preferred over other possible representation extraction methods. For example, how would performance differ if alternative representations were used? In addition to quantiative results, qualitative analyses could further support the claimed effectiveness of CanoReps.

* CanoReps are randmoly selected per category for CaDistill. It would be helpful to include an ablation study examining this choice. For example, what happens if CanoReps corresponding to specific input images are used instead of randomly sampled ones?

* Section 3.5 mentions that CanoReps can be extended to text-conditioned models. In this context, it would be intersting to know whether the CaDistill framework can also be applied to text-image correspondence models, and if so, whether any preliminary results or insights are available.

* In the distillation experiments, comparisons are primarily made with other diffusion-based distillation methods. Includign results from other recent distillation approaches would provide a more comprehensive evaluation. This would also help justify the diffusion-based design by showing that it achieves superior performance despite additional computational cost involved in representation generation.

* The paper currently presents CaDistill as the only application of CanoReps. If the approach could be extended to other downstream tasks, such as transfer learning and image editing, it would strenghten the contribution and highlight broader applicability.

* The description of Figure 3 lacks sufficient details. The meaning of each curve is unclear from the current legend and text description.

* Some important methodological explanations and results are deferred entirely to the appendeix, which hinders comprehension. For example, Section 3.3 is difficult to follow without consulting Appendix K, even though it is presented as a subsection of the main text.

### Questions
Please provide discussions or clarifications on the points raised in the Weaknesses section.

In particular, I would like to understand the fundamental reason why CaDistill achieves superior performance. What underlying property of CanoReps makes it especially effective for distillation?

I am also intersted in how CanoReps could be applied beyond distillation and whether it can extend to other downstream tasks.

### Soundness
2

### Presentation
2

### Contribution
2
