# INFER: Embedding Integration with Feature Refinement for Few-Shot Learning in VLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
On general visual recognition tasks, CLIP has demonstrated remarkable few-shot  performance by aligning image and text modalities. However, CLIP's \textit{sole} reliance on the class (CLS) embedding for image representation limits its capacity to capture spatially fine-grained features, which are crucial for fine-grained classification tasks, where subtle differences (e.g., bird species, car models) depend on small, localized variations rather than just the overall object outline. To address this, we introduce INFER, a feature enhancement strategy for CLIP that enhances image embeddings with spatial information intelligently extracted from all patch embeddings as well as the CLS embedding. To this end, INFER leverages attention heads  to compute attention-weighted features of both the patch and CLS embeddings. The most informative heads for each class, identified by their alignment with class text embeddings, are selected to enrich the patch and CLS features, which are then integrated through a fusion module. In the few-shot learning paradigm, INFER establishes new SOTA performance, highlighting the underutilized potential of CLIP’s internal attention mechanisms and providing a generalizable framework for patch-level enhancement in CLIP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a novel framework, INFER, that enhances frozen CLIP embeddings for few-shot image recognition. The authors introduce two main innovations: first, they select class-consistent attention heads from the final layers of the CLIP visual transformer to reweight both class token and patch representations; second, they fuse the enhanced features using a lightweight CNN and MLP module to produce a more discriminative image embedding aligned with textual class semantics.
Through this attention-guided reweighting and dual-path fusion, INFER effectively leverages the latent spatial details already present in CLIP without additional fine-tuning. Experiments across ten benchmarks show that INFER consistently outperforms strong baselines and recent state-of-the-art few-shot methods, especially in fine-grained classification settings.

### Strengths
1. This paper introduces a novel attention-head selection mechanism to leverage class-consistent visual cues without modifying the pretrained encoder.
2. The proposed attention-based feature enhancement and CNN–MLP dual fusion modules provide a clear and interpretable way to reweight and integrate spatial and semantic information, improving representation quality.
3. The method is computationally efficient and training-free, requiring no additional gradient updates while achieving substantial gains in accuracy.

### Weaknesses
1. The paper title is generic. Integration, refinement, and VLMs are all common prospectives of few-shot learning and vision-language models. They struggle to effectively convey the unique contributions or technical highlights of the work. A more specific and informative title would help better position the paper.
2. The motivation is limited. Many existing CLIP-based models not only depend on CLS token for classification, but also leverage remaining token sequence, including exploration about local patch enhancement. The innovation is not powerful compared to relevant works. A more clear and specific comparison with similar works on theory and experiments is important.
3. INFER generates a class-specific image embedding for every candidate class and aggregates similarities, computation and storage grow proportionally to C It would be important to compare inference/storage overhead with a few existing papers on large vocabularies (e.g., ImageNet-1k)
4. Semantic information with a simple template ("A photo of a [class]") is limited helpful on discriminative and class-specific tokens, deviating the attention choice from the actual content needed.  The potential benefits of enriching semantic information deserve further exploration.
5. The effectiveness of INFER on fine-grained scenarios should be further verified on more benchmarks, such as CUB, Cars, and Dogs.

### Questions
yes.

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
This paper introduces INFER, a framework that enhances CLIP's few-shot learning capabilities by addressing its over-reliance on the global [CLS] embedding. INFER leverages rich spatial information from patch embeddings by proposing a two-stage approach: first, it selectively enhances both [CLS] and patch features using a class-specific attention head selection mechanism, which identifies heads that are most semantically aligned with the target class's text embedding. Second, a lightweight fusion module integrates these refined global and local features. By operating on a frozen CLIP backbone, INFER effectively captures fine-grained details, establishing new state-of-the-art performance on various recognition benchmarks.

### Strengths
The paper is well-motivated, and the proposed approach is interesting.

### Weaknesses
1.	The paper lacks an analysis of the method's computational complexity and efficiency (e.g., training/inference time).
2.	The motivation for the two-stage training strategy (CNN then MLP) instead of training the model end-to-end is unclear.
3.	The experiments would be more comprehensive with results for the common ViT-B/16 backbone.
4.	The scope of the few-shot evaluation is somewhat limited.
5.	The model's generalization ability has not been sufficiently tested on out-of-distribution datasets.
6.	A key ablation study is missing, making it unclear what contributes most to the performance gain.
7.	 In the head ablation study, the performance on CIFAR and DTD is very similar when using all 12 heads versus only the top 4, which raises questions about the necessity of the head selection mechanism.

### Questions
1.	Could you provide an analysis of the method's computational complexity, such as its training and inference time?
2.	What is the motivation for the two-stage training strategy instead of training the model end-to-end?
3.	Could you include results for the common ViT-B/16 backbone to make the evaluation more complete?
4.	Could you please include results for 1, 2, 4, and 8-shot settings to broaden the few-shot evaluation?
5.	A stronger test of the model's generalization would be to evaluate the ImageNet-1K trained model on its challenging variants. We suggest including results on datasets like ImageNet-A, -R, V2, and -Sketch.
6.	How much of the performance gain comes from the proposed attention head weighting versus the trainable CNN and MLP components? Could an ablation study be conducted to isolate these effects?
7.	Regarding the head importance metric in Eq. 7: Since α appears to be a scalar, the value of cos(Pαz, M_txt) should be identical to cos(Pz, M_txt). Does this imply that the intermediate layer's attention scores don't actually influence the head selection? Could you please clarify this?
8.	The method section is a bit confusing and hard to follow. Could the method section be rewritten for better clarity?
9.	The citation style is inconsistent throughout the paper (e.g., some with parentheses, some without). Could you please use a consistent citation format throughout the paper?

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
4

### Summary
INFER enhances CLIP's fine-grained recognition through intelligent fusion of spatial information from patch embeddings with the standard CLS token. By identifying semantically relevant attention heads via text alignment and integrating enhanced features through lightweight fusion, it achieves improved capture of subtle visual distinctions.

### Strengths
INFER enhances CLIP's visual representations by intelligently selecting semantically meaningful attention heads to refine both CLS and patch embeddings. While maintaining CLIP's original frozen architecture, this approach improves few-shot classification accuracy and offers interpretable insights into model attention. The lightweight design ensures easy integration with various foundation models and downstream applications.

### Weaknesses
Despite the interesting contributions, I have several significant concerns regarding the manuscript:
1. I recommend adding visualizations of the model's attention maps on real samples. This would offer valuable insights into the discriminative regions it leverages and help validate its decision-making process.
2. The performance gains vary significantly across benchmark datasets. The authors should provide a thorough analysis to explain these discrepancies, potentially relating them to dataset characteristics like granularity, diversity, or attribute complexity.
3. The experimental comparisons should be updated to include more recent state-of-the-art methods, particularly from 2024 and forthcoming 2025 publications where available.
4. The authors should clearly articulate the specific advantages of their proposed module over existing prompt learning methods. Furthermore, a direct performance comparison with these methods is necessary to quantitatively demonstrate its benefits.

### Questions
The authors should provide a more thorough discussion and analysis of the points raised above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes INFER, a novel method for few-shot adaptation of Vision-Language Models (VLMs) like CLIP, with a particular focus on improving performance on fine-grained recognition tasks. The authors identify the key limitation of standard CLIP as its sole reliance on the global [CLS] embedding, which discards crucial spatial information present in patch embeddings. The authors demonstrate that this approach, which keeps the VLM backbone frozen, achieves new state-of-the-art (SOTA) results on 11 few-shot classification benchmarks, outperforming recent methods like 2SFS and MaPLe.

### Strengths
1. Novel and Well-Motivated Mechanism: The central idea of using text alignment to quantitatively select the most relevant visual attention heads is a novel and clever way to bridge the gap between the two modalities for feature refinement. It addresses the well-known "modality gap" and provides a principled way to extract more discriminative features from the frozen VLM backbone.

2. Strong Empirical Performance: The paper reports SOTA results on a comprehensive suite of 11 datasets, including a good mix of generic, scene, and fine-grained benchmarks. The performance gains over strong, recent baselines (e.g., +1.3% over 2SFS on ViT-B/32 and +1.7% on ViT-L/14) are significant and demonstrate the effectiveness of the proposed method.

3. Parameter Efficiency: The adaptation strategy is lightweight. By keeping the entire VLM backbone frozen and only training a small fusion module (a 3-layer CNN and an MLP), the method is highly parameter-efficient, which is a very desirable characteristic for few-shot learning.

### Weaknesses
My main concerns are related to the clarity and justification of the inference mechanism and the training process, as well as the computational cost.

1. Clarity and Scalability of Inference (Eq. 10 & 11): The inference process appears to be computationally intensive and is not fully justified.

Q1: As I understand it, for a single test image $I$ and a dataset with $C$ classes, the model generates $C$ different image embeddings, $M_{img}^{INFER,c}(I)$, one for each class's "enhancement" coefficients. Then, a $C \times C$ similarity matrix is computed. The final score for a text class $c'$ is the sum of similarities across all $C$ image embeddings (Eq 11). Is this understanding correct?

Q2: If this is correct, this approach seems to scale poorly with the number of classes. For ImageNet ($C=1000$), this would require generating 1000 distinct image embeddings and computing a $1000 \times 1000$ similarity matrix for each test image. Could the authors please provide a detailed analysis of the inference-time computational overhead (e.g., FLOPs or latency) compared to baselines like a linear probe or 2SFS, especially on many-class datasets?

Q3: What is the justification for this ensemble-like "sum-of-scores" approach (Eq 11)? It seems to measure which text prompt $t_{c'}$ is, on average, most similar to all class-specific views of the image $I$. Why is this superior to a simpler, more efficient strategy, such as only computing the diagonal of this matrix (i.e., argmax_c [cos(M_img^{INFER,c}(I), M_txt(t_c))])?

2. Justification of Staged Training (Section 3.2): The paper states that the fusion module is trained in two steps: "We first train $f_{CNN}$ with the few-shot samples. With $f_{CNN}$ frozen, we then train $f_{MLP}$".

Q4: Why is this sequential training necessary? Was a joint end-to-end training of $f_{CNN}$ and $f_{MLP}$ attempted? If so, how did it perform? This staged approach seems somewhat arbitrary and would benefit from an ablation study or a clearer justification.

3. Weak Baseline in Qualitative Ablations (Fig. 3): The t-SNE and confusion matrix in Figure 3 compare INFER to standard CLIP. This is a very weak baseline, as standard zero-shot CLIP is not expected to perform well on a dataset like MNIST, and it doesn't represent the strong few-shot baselines that INFER is otherwise competing against. Suggestion: A much more compelling visualization would compare the feature separability of INFER against another adapted SOTA method, like 2SFS or MaPLe. This would more directly demonstrate the benefit of INFER's specific feature enhancement.

4. Limited Comparison Results. The paper only presents a single quantitative table as the main experimental result, which is insufficient for a top-tier conference publication. It is strongly recommended that the authors conduct experiments on additional datasets and include more comprehensive ablation studies to validate the effectiveness of their method.

### Questions
Please see the above questions.

### Soundness
2

### Presentation
3

### Contribution
2
