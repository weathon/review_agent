# InfoNCE Induces Gaussian Distribution

- Avg Score: 4.00
- Decision: Accept (Oral)
- Scores: 4, 2, 8, 2

## Abstract
Contrastive learning has become a cornerstone of modern representation learning, allowing training with massive
unlabeled data for both task-specific and general (foundation) models.
A prototypical loss in contrastive training is InfoNCE and its variants. 
In this work, we show that the InfoNCE objective induces Gaussian structure in representations that emerge from contrastive training. 
We establish this result in two complementary regimes. 
First, we show that under certain  alignment and concentration assumptions, projections of the high-dimensional representation asymptotically approach a multivariate Gaussian distribution.
Next, under less strict assumptions, we show that adding a small asymptotically vanishing regularization term that promotes low feature norm and high feature entropy leads to similar asymptotic results.
We support our analysis with experiments on synthetic and CIFAR-10 datasets across multiple encoder architectures and sizes, demonstrating consistent Gaussian behavior.
This perspective provides a principled explanation for commonly observed Gaussianity in contrastive representations. 
The resulting Gaussian model enables principled analytical treatment of learned representations and is expected to support a wide range of applications in contrastive learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the statistical structure of representations learned through contrastive learning with the InfoNCE loss, and demonstrates that such representations closely follow a multivariate Gaussian distribution. The authors provide two theoretical justifications: (1) under alignment and concentration assumptions, high-dimensional embeddings asymptotically approach Gaussianity; and (2) under weaker assumptions, introducing a small regularization term that encourages low feature norms and high entropy leads to similar asymptotic behavior. Theoretical analysis is further supported by experiments on synthetic data, CIFAR-10, and pretrained foundation models, consistently validating the Gaussian property of learned representations.

### Strengths
1.	Theoretical and empirical results align well. The experiments on synthetic datasets, CIFAR-10, and pretrained models convincingly support the Gaussianity of contrastive representations.
2.	The paper is well-organized, with a clear logical flow that makes both the theoretical derivations and empirical validations easy to follow.
3.	The work introduces a novel and meaningful perspective for understanding the contrastive learning objective, potentially offering a new theoretical framework for future studies in this area.

### Weaknesses
1.	While the theoretical findings are solid, the paper could benefit from more discussion on how the Gaussianity of learned representations may translate into practical insights or applications. In the introduction, the authors claim that understanding representation distributions is not only of theoretical insights. However, the main body does not explore how this insight could improve contrastive learning methods or be leveraged in practice  This is my main concern.
2.	The claim that contrastive representations follow a Gaussian distribution offers a novel theoretical view. However, it would strengthen the paper to compare this finding to prior work that also studies the representations of contrastive embeddings—such as [1] and [2], which show that contrastive features tend to form well-separated clusters aligned with supervised labels. These perspectives seem complementary: clustering and Gaussianity both describe structural regularities in learned features. Intuitively, it seems that their results could provide more guidance for our application as contrastive learning (as classification is one of the most important application and metric of learned representations). I wonder what are the advantages of the theoretical analysis in this paper. A clearer comparison could help clarify the added value or distinct implications of the Gaussian model over existing analyses.
3.	The paper focuses on the unnormalized representation space, which is an interesting and often underexplored aspect. Most contrastive frameworks compute losses in the normalized space, while fine-tuning typically uses unnormalized embeddings. It would be valuable if the authors could discuss whether their theoretical analysis provides any insight into this transition—e.g., why Gaussianity might hold or break across normalized and unnormalized spaces, and what this implies for the design of pretraining-finetuning pipelines.
4.	The paper progressively relaxes its assumptions. Nevertheless, it remains unclear how strong or realistic these assumptions are in practice. It would be helpful for the authors to clarify the degree of relaxation in each step and to discuss how far the theoretical conditions deviate from those typically observed in real contrastive learning scenarios. More empirical verifications or discussions of assumption validity could make the theoretical results more convincing and grounded.

[1] Saunshi, Nikunj, et al. "A theoretical analysis of contrastive unsupervised representation learning." International conference on machine learning. PMLR, 2019.

[2]  HaoChen J Z, Wei C, Gaidon A, et al. Provable guarantees for self-supervised deep learning with spectral contrastive loss[J]. NeurIPS, 2021.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors theoretically and empirically analyze representations learned by the InfoNCE contrastive loss, showing they become approximately Gaussian in high dimensions. They present two approaches to demonstrate their claim. First, under two assumptions: an “alignment plateau” assumption and a "thin-shell" assumption, the population-level InfoNCE solution is isotropic, so normalized features are uniform on the sphere and any fixed-$k$ projection is asymptotically Gaussian as dimension $d\to\infty$. Second, even under weaker conditions, adding a vanishing $\ell_2$-norm/entropy regularizer biases the solution toward isotropy, yielding the same asymptotic Gaussian law. Empirically, the authors validate these claims on synthetic data, CIFAR-10, and pretrained models (e.g. CLIP, DINO). They measure thin-shell concentration of feature norms and apply normality tests (Anderson–Darling and D’Agostino–Pearson) to demonstrate that InfoNCE-trained features indeed exhibit Gaussian statistics in practice. The paper argues this formalizes prior observations of “near-Gaussian” features in contrastive learning and justifies Gaussian modeling for downstream tasks (e.g. OOD detection).

### Strengths
- Theoretical rigor: The paper provides a novel and rigorous analysis using high-dimensional probability and spherical CLT tools (e.g. Hirschfeld–Gebelein–Rényi maximal correlation bound, polar KL decomposition, Maxwell–Poincaré CLT). It derives precise conditions (bounded alignment, uniformity on sphere) under which InfoNCE yields isotropic, Gaussian outputs.

- Comprehensive empirical validation: The authors evaluate across diverse settings: a synthetic Laplace dataset, CIFAR-10 with a small encoder, and large pretrained models (supervised ResNet/DenseNet vs self-supervised CLIP/DINO). They consistently observe thin-shell norm concentration and high normality p‑values for InfoNCE-trained features (vs non-Gaussian behavior for supervised models).

- Clarity and organization: The paper is well-structured. Key assumptions and results are clearly stated (e.g. via bullet-point summary of contributions on p.2). The exposition links the theory to concrete phenomena (alignment saturation, feature variance) and shows illustrative figures (e.g. Figure 3) that make the findings intuitive. Definitions and proofs are thorough and mostly easy to follow.

### Weaknesses
- Strong asymptotic assumptions: The theory relies on high-dimensional limits and idealized assumptions (infinite negatives, alignment plateau, perfect norm concentration). These may not hold in all practical cases. The authors acknowledge this, noting the results are asymptotic and “alignment plateau and thin-shell concentration… are not guaranteed to hold universally”. It remains unclear how well the Gaussian approximation holds for moderate $d$ or when these assumptions fail.

- Dependence on regularizer: One route to Gaussianity uses an added regularizer (low-norm, high-entropy) that vanishes asymptotically. It’s not fully clear how sensitive the results are to this term in finite dimensions. Practical guidelines for choosing (or removing) this regularizer without harming performance are not discussed. Besides, it would have been interesting to study the impact of this regularizer on downstream classification performances on one or more different datasets (CIFAR10, STL10, IMAGENET100, IMAGENET1K).

- Scope of experiments: Although varied, the experiments mostly focus on vision models and one dataset (MS-COCO) for pretrained features. It would strengthen the paper to test more domains (e.g. other modalities or tasks or non-natural images) to see if Gaussianity is universal for InfoNCE. Additionally, experiments on how violations of assumptions affect outcomes (e.g. weak augmentations) are not shown.

- Overall, the Gaussianity of the latent space is not so surprising, it has been already shown in Wang and Isola, 2021 that the representations would converge toward a uniform hyperspherical distribution. And it is commonly known that there is a close connection between uniform hyperspherical distribution (von-Mises Fisher with concentration equals to 0) and high-dimensional Gaussian distributions known as the soap-bubble-effect as stated by Vershynin, 2018; Lecture Notes on High-Dimensional Data, Sven-Ake Wegner, https://arxiv.org/pdf/2101.05841; Hyperspherical Variational Auto-Encoders, https://auai.org/uai2018/proceedings/papers/309.pdf Davidson et al. 2018; Farquhar et al, 2021 Radial Bayesian Neural Networks: Beyond Discrete Support In Large-Scale Bayesian Deep Learning, https://arxiv.org/abs/1907.00865. The paper would benefit discussing these results and these papers in order to clearly state their contribution.

- It is unclear why CLIP and DINO (what about v2 or v3 ?) exhibits Gaussian distribution properties even though they were not exactly trained like SimCLR: CLIP positive-pair probably implies different alignment propoerties than SimCLR, and DINO's loss is not exactly the same as SimCLR, the paper would benefit from a better discussion about other self-supervised methods.

### Questions
- In practice, how does one verify the “alignment plateau” condition during training? Can alignment be measured online to check this assumption? Is the plateau different given different images ? Does that mean something about the quantity of information carried by the image ?

- The vanishing regularizer is claimed to ensure isotropy at finite $d$. How should its strength be chosen in practice? Did you experiment with non-vanishing regularizers, and if so, how did they affect downstream performance?

- The work emphasizes normalized (unit) features. Does the Gaussianity hold equally for unnormalized (pre-norm) embeddings? Fig.3 and authors remarks suggests yes, but clarity on this point would be good. It is unclear why would upstream representations (like the one used for downstream tasks, typically before the projection MLP 2 layers) would exhibit a multi-dimensional Gaussian distribution as well.

- Do these results extend to other contrastive losses ? Empirically, on CLIP and DINO it appears to, but what is the theoretical argument to support this behaviour ? Is the Gaussian outcome specific to InfoNCE batch training?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the distribution of the optimal representation obtained by optimizing the InfoNCE objective. It demonstrates, under alignment and concentration assumptions, that finite projections of the high-dimensional representation are Gaussian. The authors then weaken their alignment and concentration assumption to obtain a similar result. Finally, they give empirical evidence to confirm they claim on widely used pre-trained contrastive models (CLIP) and non-contrastive (DINO) models.

### Strengths
-	The mathematical analysis is rigorous, the first claims (Corollary 1 and Proposition 2) are strong (even if they are directly obtained from two well-known results) and the section 4.2 is very technical but sounded. 
-	The empirical evidence given for real-world contrastive-based models gives credit to the theoretical analysis. There are not many works testing the Gaussian assumption on the representations of foundation models while it is often assumed for downstream applications (e.g. OOD).
-	The exposition is clear, concise, and easy to follow (excepted maybe for section 4.2 which requires special attention for the numerous notations…).

### Weaknesses
-	Regarding Assumption 3, the authors mentioned it is weaker than previous Assumption 1. It is still not clear to me the realizability of such assumption and why it would hold in practice. Assumption 1 may be easier to verify empirically (as the authors already mentioned) and it seems more intuitive.
-	In the experiments, you consider DINO, a non-contrastive approach that does not introduce any uniformity term in its loss (which is key in your analysis since you always assume the alignment to be controlled by a constant). It is interesting to note that DINO representation seems to be Gaussian as well, but you do not justify it theoretically. Do you have any theoretical analysis linking the knowledge distillation technique in DINO with the uniformity loss optimized by InfoNCE? 
-	In your experiments, you also consider CLIP, which is a constative approach, but it does not fit well in your theoretical analysis. Indeed, the vision and text encoders are different (they do not share the same weights, and they are not defined in the same space) and the marginal distribution for each modality is different. Therefore, your uniformity term in eq. 5 seems to not fit this case. It is again interesting to note that image and text representations are Gaussian but you do not justify it theoretically.
-	Finally, it is funny to observe that the only foundation model that fits perfectly your analysis (SimCLR) is not present in your experiments!

### Questions
Most of my questions are described in the weaknesses. I believe answering them will improve the quality of the manuscript, in particular regarding the match to the current practice in contrastive learning.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents theory that representations obtained with contrastive learning yield "Gaussian representations". The authors theoretially investigate this claim for normalized and unnormalized contrastive learning techniques and present a small scale theoretical study across popular contrastive learning models in computer vision.

### Strengths
- The authors connect new theory on properties of contrastive representations with empirical experiments covering real world models
- The paper is dense with theoretical results, but still fairly well structured and easy to follow
- The authors release code for reproducibility

### Weaknesses
1. It seems like the authors missed existing identifiability literature for contrastive learning, e.g. https://arxiv.org/abs/1605.06336, https://proceedings.mlr.press/v54/hyvarinen17a/hyvarinen17a.pdf, https://arxiv.org/abs/1805.08651, https://proceedings.mlr.press/v139/zimmermann21a.html, https://arxiv.org/abs/2007.00810, https://arxiv.org/pdf/2410.21869. It would be good to discuss how the presented theory (which uses a different set of tools) relates to this work; the presented work does not talk about the data generating process, but I wonder if the results are fundamentally connected to identifiability; Gaussian representations (or their projections to the sphere) is what we would expect with the considered form of the InfoNCE loss (in particular cf. https://proceedings.mlr.press/v139/zimmermann21a.html) for this.
2. "In particular, the raw unnormalized representations have received little theoretical attention": This statement seems ungrounded, indeed a lot of identifiability theory also studies the unnormalized case (see above).
3. The experiments are not sufficiently rigorous to support the theory. It is unclear to me what the null hypothesis is, and how the authors test for Gaussianity. It would be quite helpful if a study with simulated data would be conducted that precisely illustrates the theoretical claims.

### Questions
1. I might have a fundamental misunderstanding here, but the sentence "Overall, we provide the first principled explanation for Gaussianity in contrastive representations" seems off. Can you comment on how your theory relates to identifiability theory for contrastive learning, which 
2. What is the null hypothesis against the Gaussian distribution that is being observed, what would be the alternative? Will the Gaussian distribution also appear when the underlying data-generating process is non-gaussian distributed, or cannot me mapped into a feature space which is Gaussian-distributed?
3 What is a "Gaussian projection"? (L 191) I assume this is a low-D projection of the full feature space on a 1D axis, where Gaussianity is tested? Doesn't the Gaussianity of a projection then trivially follow from the law of large numbers? If not, could you highlight why this finding is significant?
4. What is the expectation in Eq. 8 over?
5. Consider a mixture of Gaussian distribution, or a vMF mixture where each component represents one class; can you comment if the marginal distribution expected through contrastive learning for this case would be Gaussian? Previous work seems to suggest that a mixture of Gaussians would be recovered, and this seems to conflict with the presented theory here; could you comment?

### Soundness
2

### Presentation
2

### Contribution
2
