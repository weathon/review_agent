# From Coefficients to Directions: Rethinking Model Merging with Directional Alignment

- Decision: Reject
- Scores: 2, 2, 8, 4

## Abstract
Model merging is a recently emerging paradigm that aims to integrate multiple independently trained models into a single model without joint retraining. Previous studies have demonstrated the effectiveness of combining parameters from multiple models through strategies such as parameter decomposition, coefficient optimization and subspace learning. These methods significantly reduce the need for expensive joint training and have shown strong empirical performance across diverse tasks.    However, these approaches predominantly treat merging as a problem of parameter space decomposition or fusion coefficient optimization, while largely overlooking the role of directional information in both parameter space and feature space. In practice, naïve merging introduces inconsistencies in dominant parameter directions and disrupts structural coherence across models, which can degrade performance. Moreover, coefficient-based optimization methods implicitly assume compatible feature-space directions across models. However, Neural Collapse indicates that class features follow structured directional patterns, which may differ across independently trained models, making coefficient optimization alone insufficient. In this work, we emphasize the importance of \emph{directional alignment} and introduce a unified geometric framework,  
\emph{Merging with Directional Alignment}~(MDA), which aligns directional structures consistently  
in both the parameter and feature spaces. Our theoretical analysis demonstrates that directional alignment improves structural coherence and tightens generalization bounds. Extensive experiments across benchmarks, model scales, and task configurations confirm the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles the problem of Model Merging. Differently from previous model merging works, which evaluate model merging solely in terms of merging coefficient adjustment or merging task matrices in parameter space, the authors address the problem from a different perspective, focusing on directional alignment in feature space. Building on Neural Collapse insights showing that the final classification layer converges to Equiangular Tight Frames (ETF), the authors propose to align the reconstructed parameters for merging to the ETF. Specifically, they construct a shared space by concatenating the top-k left singular vectors for each task vector at each layer, and then align this shared representation to the ETF.
For feature-space alignment, they introduce a joint optimization framework consisting of three losses: an entropy loss, an alignment loss, and a rotation loss. During optimization, a hyperparameter $\lambda$ is learned per layer through entropy minimization. The features for each task $t$ are rotated via a rotation matrix $R^{t}$ and aligned, using the L2 norm, with the ETF vector of each class in task $t$. This loss is summed over all tasks. Each rotation matrix $R^{t}$  is obtained by minimizing a rotation loss, which encourages $R^{t}$ to be close to the optimal Procrustes rotation computed from the empirical class means and the ETF targets. The method is evaluated on vision tasks in both multi-task settings and generalization to new classes. In the appendix, additional NLP results on LoRA fine-tuned models are provided.

### Strengths
- Aligning the feature space in model merging to the ETF is a novel idea. I believe that feature-space alignment has been largely overlooked in the existing literature, and I appreciate this contribution. 

- I found Figure 2 (right), showing a correlation between performance and deviation from the simplex ETF structure, interesting though more explanation and comparisons with other methods would strengthen the analysis.

- The approach is evaluated on several vision tasks as well as on unseen tasks. Additionally, the appendix reports evaluations on NLP LoRA fine-tuning benchmarks.

- The proposed method shows good overall performance.

### Weaknesses
The presentation of the methodology needs significant improvement. In its current form, it is difficult to understand how the method works as a whole. Moreover, several unclear aspects remain that should be addressed from both theoretical and methodological/experimental perspectives.

1) Section 4.1:  If I have understood correctly, for each task you first retain the top-$k$ principal components $U_t^{(l,k)}$. Then, these are concatenated to form $U_{cat}^{(l)}$ (line 243, left). Then you perform the SVD of $U_{cat}^{(l)}$ (line 243, right) and truncate it  $d_{out}$  to obtain the expression in line 239? The text describing this procedure, if I have correctly understood,  is not clear and rather fragmented. Moreover, this step is not illustrated in  Algorithm 1. $U_u^{(l)}$ seems a typo; It is not defined anywhere. Moreover, truncating to $d_{out}$ seems only useful for matching the target size. I ask the authors to clarify this. 
2) Theorem 1:  Is Theorem 1 proposed by the authors? If not, please add the appropriate reference. If yes, where is its proof provided? Why is it relevant to the proposed method? This is not clear from the text. There also seems to be a discontinuity in the exposition — the connection between the shared subspace construction and Theorem 1 is unclear, and the relevance of the theorem to the method is not well explained.
3) The theoretical bound presented in Section 4.2 appears to be related only to the construction described in Section 4.1. In that case, the two sections should be merged. Moreover, the theorem seems entirely unrelated to how the shared space is constructed using the SVD of each task vector. It appears independent of the specific construction of the shared space and could be applied to any merged task vector that satisfies Equation (3), regardless of how it is obtained. From a theoretical point of view, why is the shared subspace obtained by concatenating the SVDs the appropriate subspace for the bound in Equation (5)?

4) The theoretical exposition assumes that the ETF structure induced by Neural Collapse emerges in all layers of the network. To the best of my knowledge, the ETF structure has been observed only in the final layer of the network (consistent with what is done in Algorithm 2). For instance, the assumption in line 768 of the Appendix – *Under the assumption that $\tau_{ideal}^{l}$ exhibits near-ETF structure, its column space lies essentially in the ETF subspace* –  it is questionable when applied to all layers of the network. The authors must clarify this . From a practical standpoint, why is it necessary to perform the operation in Equation (3) for all layers rather than only for the final layer?

5) From the text description (as well as the algorithms and the figure of the methodology in the appendix), it is not clear whether the authors are proposing two complementary strategies for solving the problem or two strategies that can be combined together.  Algorithm 1 and Algorithm 2 appear to be completely disconnected and it is not clear how they are intended to be used together. Even in the algorithmic formulation, it does not seem that they are used together, since the two algorithms operate on different inputs. In the experimental section, the authors distinguish between MDA-TA and MDA-AM, but no clear definition of these two approaches is provided. Is MDA-TA based solely on Algorithm 1? Is MDA-AM a combination of Algorithm 1 and Algorithm 2? If so, how are they integrated? This is not evident from the current presentation.

6) The notation under Equation (11) in Section 4.2 is overly complicated and should be simplified. Moreover, several derivation steps are not clearly explained, particularly in the construction of $R_{proc}​$; for instance, a determinant term appears abruptly without clarification.

7) How $\alpha$ and $\beta$ are chosen for each setup? In the data-free setting, it is unclear whether Algorithm 2 is employed, there fore not sure if they need to be set for this(clarifying the connection between Algorithm 1 and Algorithm 2, as mentioned earlier, could help here). Similarly, in the Adaptive Merging setup, how are these parameters determined? 


Other relatively minor concerns must also be addressed:

- Line 103: At this point in the paper—near the end of the introduction—the authors introduce an additional layer of complexity that is difficult to follow when reading sequentially. It is not clear how the ETF structure is constructed for model merging.  Moreover, how is the cosine similarity between two matrices defined? Figure 1 (left) is also unclear: what do the fusion coefficients refer to—the classifier weights or the model weights? A better caption for this must be designed.

- The experimental splits appear to follow those used in TSV for evaluation. However, it is unclear how the validation set from the TSV setting is utilized in the current method. Similarly, for adaptive merging, is the unsupervised set used in Algorithm 2 the same as that employed in AdaMerging? These implementation details are missing from the paper.

- Related works: At the end of the related works section, it would be helpful to clearly specify the setting in which the proposed approach operates.

### Questions
I have no further questions regarding the paper. However, in its current form, the paper remains unclear in several aspects, some of which may stem from my own misunderstanding. At this stage, it requires substantial clarification and further development before it can be considered suitable for publication, despite the novelty and interest of the proposed idea.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Existing model methods often ignore directional information in parameter and feature spaces. The authors propose Merging with Directional Alignment (MDA), a method that aims to aligns directions. They motivate their work with insights from neural collapse, showing that traditional model merging disrupts the simplex equiangular frame (ETF). Results on standard benchmarks show promising results.

### Strengths
- method obtains good results on standard vision model mering benchmark dataset, including good generalization. 
- gain of method increases for more difficult 20 tasks scenario and also for larger models (ViT L/14)

### Weaknesses
- I found the presentation lacking. The paper is very hard the follow and I found often unclear and incorrect.. 
(E.g. The introduction of Neural collapse  (section 3.2) needs motivation. In the start of section 4, the main motivation claims that 'Other methods (Gargiulo, wei, Marczak) are based on task-specific subspaces which are claimed to be computationally expensive. They do not only focus on task-specific subspaces, and the paper lacks a computational comparison, etc).   

- The results of ISO-CLS are missing and those of ISO-TA are lower than in their paper (and those are comparable than the ones proposed in this paper). Could you explain that ? 

- the first algorithm, being SVD based has great similarity with Gargiulo et al. The authors should provide a more detailed explanation of the main differences (and similarities). 


minor:
- in related work, authors should clearly state to which strategy they belong data-free/Data-based. 
- typo: Lalign and Lrotation

### Questions
The others should address the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a framework called Merging with Directional Alignment (MDA) for model merging. The key claim is that prior approaches—focused mainly on adjusting coefficients or subspaces—ignore the role of directional information in both parameter and feature spaces. Drawing inspiration from the neural collapse phenomenon, the authors argue that the optimal geometry of a well-trained network approximates a simplex equiangular tight frame (ETF), and that preserving this directional structure during merging can mitigate interference and enhance generalization. MDA thus proposes two main components:

- Parameter-space directional alignment: Low-rank decomposition of task vectors followed by alignment with a constructed ETF basis.
- Feature-space directional alignment: Optimization of task-specific rotation matrices (in SO(d)) and fusion coefficients using a composite loss balancing entropy, ETF alignment, and rotation regularization.

The paper claims theoretical justification via Rademacher complexity bounds and demonstrates empirical gains across vision (ViT-B/32, ViT-L/14) and NLP (Flan-T5-base) tasks.

### Strengths
- The idea of explicitly enforcing directional alignment in model merging is original in its formulation and integration of neural collapse geometry into the merging framework. 
- The empirical evaluation is extensive, spanning 8–20 task vision benchmarks and GLUE-style NLP datasets. The improvements, though moderate (typically +0.5–2%), are consistent and show that directional alignment has a measurable effect. The paper provides clear ablations for the loss weights and module contributions, showing awareness of the internal trade-offs.
- The paper is well-organized and accessible to readers familiar with model merging literature.
- If validated more rigorously, MDA could meaningfully influence the design of geometry-aware merging algorithms, offering a conceptual bridge between theoretical properties of trained models (neural collapse) and practical post-training fusion methods.

### Weaknesses
- Despite focusing on geometry, the paper lacks quantitative or visual analysis of directional statistics: e.g., angular distributions between task vectors before/after alignment, cosine similarity matrices, or CKA correlation trends.
- The distinction between MDA and prior geometric methods (TSV, ISO, DOGE) remains unclear. Many of these already manipulate task subspaces or singular vector orientations—essentially performing partial directional alignment.
- The feature-space optimization involves learning rotation matrices in SO(d), which can be expensive for high-dimensional layers. 
- Neural collapse typically emerges in overparameterized, fully trained networks under strong assumptions (balanced data, vanishing training error). Fine-tuned models and merged networks rarely meet these conditions. The connection to ETF geometry, while intuitively appealing, is speculative.

### Questions
- How is $\Delta ETF$ measured in practice? Could the authors report angular deviations or ETF cosine scores before and after alignment to confirm the hypothesized correlation between geometry and accuracy?
- What is the computational overhead (in FLOPs or training time) introduced by optimizing the rotation matrices and fusion coefficients? Would a simplified or approximate alignment (e.g., diagonal or low-rank rotations) retain most of the benefit?
- Does MDA remain stable when merging models fine-tuned on dissimilar modalities (e.g., text vs. image encoders), or does it assume shared architectures and token embeddings?
- The low-rank truncation in Algorithm 1 is crucial; how does varying the rank k affect both computational efficiency and accuracy? 
- Since many modern fine-tuning schemes are parameter-efficient, can the ETF alignment principle apply to merging low-rank adapters instead of full models?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper theoretically establishes the importance of directional alignment in model merging, and proposes a unified framework, Merging with Directional Alignment (MDA), which aligns direction in both parameter and feature spaces. Extensive experiments on vision (ViT-B/32, ViT-B/16, ViT-L/14 across 8, 14, 20 tasks) and NLP (Flan-T5-base on GLUE) benchmarks show consistent improvements over strong baselines, with ablations validating each component.

### Strengths
- **Well-motivated perspective:** This paper formulates model merging as a problem of geometric alignment (direction) rather than just coefficient optimization (magnitude). Moreover, the paper provides a solid theoretical grounding for why directional alignment should matter by the connection to neural collapse and the simplex ETF. 

- **Comprehensive evaluation and theoretical analysis:** The experiment is well-designed and comprehensive. Its evaluation spans multiple architectures, diverse tasks, and many baselines. The consistency of improvements strengthens the initial claims. Moreover, this paper provides a theoretical analysis to demonstrate that directional alignment improves structural coherence and tightens generalization bounds.

### Weaknesses
- **Questionable assumption about neural collapse in merged models:** The paper assumes that multi-class joint training induces simplex ETF geometry and **that merged models should approximate this structure**. Why? In fact, merged models fundamentally differ from jointly trained models: they aggregate task-specific adaptations rather than training jointly from scratch. No evidence is provided showing that merged models actually violate ETF structure or that imposing ETF structure better approximates joint training than alternatives. Why is ETF specifically the right target structure for merged models? How about other geometric structures (e.g., orthogonal subspaces, common low-rank structure)?

- **Modest performance gains on LLMs:** The NLP results in Table 7 seem to underperform relative to the claims, as MDA TA (82.1%) only marginally outperforms TSV TA (81.8%) and falls short compared to the gains on vision tasks. This raises questions about whether directional alignment via ETF is as beneficial for LLMs as it is for vision models. Why might ETF alignment work better for vision than for NLP? There is a lack of discussion on this point.

### Questions
1. The paper claims the method works in both data-free and data-based regimes, but aren't the two settings quite different in terms of optimization? 
2. How sensitive are results to the rank k in SVD-based decomposition? Is k fixed or task/layer-specific?

### Soundness
3

### Presentation
3

### Contribution
3
