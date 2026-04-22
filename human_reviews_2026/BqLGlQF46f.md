# Beyond the Known: An Unknown-Aware Large Language Model for Open-Set Text Classification

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4

## Abstract
Open-set text classification (OSTC) requires models to correctly classify in-distribution (ID) samples while reliably rejecting out-of-distribution (OOD) inputs—an essential capability for real-world NLP systems. Most OSTC methods train on ID data under the closed assumption that all outputs belong to the known label space and then perform OOD detection with the biased representations, which inherently lack awareness of unknowns and thus yield overconfident predictions on OOD inputs. In this work, we present UnLLM, an Unknown-aware Large Language Model for OSTC. Instead of fixing classification to the entire known label space, we reformulate it into a subset-conditioned text generation task: the LLM is prompted with sampled subsets of known labels, and any instance outside the candidate set is explicitly assigned as “unknown”. This reformulation transforms OOD detection from a post-hoc procedure into an intrinsic modeling capability. More importantly, our approach is the first to explicitly incorporate the unknown into classification, enabling systematic modeling of unknowns through a unified representation–logits–inference optimization, which progressively strengthens the model’s capacity to capture open-set risk. Extensive experiments across six benchmarks show that UnLLM consistently outperforms state-of-the-art (SOTA) baselines. Code and datasets are available at https://github.com/cx9941/UnLLM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles open-set text classification (OSTC) by making “unknown-class” awareness an explicit, trainable behavior for LLMs via UnLLM, a three-stage pipeline.

### Strengths
- The three-stage pipeline (contrastive + orthogonality regularization; K+1 logit calibration; analogy-augmented inference) is coherent and practical.

- Shows consistent improvements on several benchmarks.

- Simple, implementable training recipe that may make LLM classifiers less overconfident under partial label contexts.

- The paper does not over-promise formal guarantees; instead it presents precise objectives/losses and a closed-form calibration projection (Eq. 4) with clear tensor-shape intent.

### Weaknesses
- This paper feels close in spirit to prior OE extensions and recent fine-tuning-with-auxiliary-data methods (WOODS, VOS, NPOS).

### Questions
- Consider a compact numeric ablation table ****(mean±std) and a reliability diagram (ECE).

- What encoder, k, and index size are used for analogy; what’s the per-example latency?

- Nice motivation—could you add one controlled test that separates *withheld in-domain labels* (conditional-OOD) from *cross-domain OOD*, to empirically reinforce generalization.

- The experimental section should take into account advanced OOD detection methods that leverage additional auxiliary OOD data, as referenced in [1, 2, 3, 4].



[1]: Ming, Yifei, et al. "Poem: Out-of-distribution detection with posterior sampling." *International Conference on Machine Learning*. PMLR, 2022.

[2]: Wang, Qizhou, et al. "Out-of-distribution detection with implicit outlier transformation." *arXiv preprint arXiv:2303.05033* (2023).

[3]: Zheng, Haotian, et al. "Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources." *arXiv preprint arXiv:2311.03236* (2023).

[4]: Wang, Qizhou, et al. "Learning to augment distributions for out-of-distribution detection." *arXiv preprint arXiv:2311.01796* (2023).

### Soundness
3

### Presentation
3

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
This paper investigates the open-set text classification (OSTC) problem, in which models must correctly classify in-distribution (ID) samples while identifying out-of-distribution (OOD) inputs. The authors propose UnLLM, an unknown-aware large language model that integrates open-set awareness directly into the fine-tuning process. The method reformulates classification as a subset-conditioned generation task, where random subsets of known labels are provided and instances whose true labels are excluded are treated as 'unknown'. To enhance representation quality and calibration, UnLLM incorporates contrastive learning, orthogonality constraints, and a post-hoc logit calibration step that aligns output weights with OOD-relevant representations. During inference, an analogy-augmented self-reflection mechanism enables the model to reassess uncertain cases and mitigate overconfidence. Experiments on six benchmark datasets demonstrate that UnLLM consistently outperforms both traditional PLM-based methods (e.g. ADB, CLAP, KNNCon) and recent LLM-based baselines (LLM-OOD, VOS, NPO), achieving strong results in both ID classification (K-F1) and OOD detection (N-F1).

### Strengths
* The main contribution of the paper lies in reformulating OSTC as a subset-conditioned text generation task, extending prior work that handled OOD detection only as a post-hoc step. This framing is new in the context of large language models and conceptually straightforward.

* The authors evaluate their approach on six benchmark datasets and a broad range of baselines, covering both traditional PLM-based and more recent LLM-based methods. The experimental coverage is thorough and shows consistent empirical improvements.

* Although the design is primarily empirical, the results suggest that the proposed paradigm can be effectively applied to large models and yields measurable performance gains in OSTC scenarios.

### Weaknesses
The main issue of the paper lies in its limited scientific depth. While the proposed framework achieves promising empirical results, it lacks sufficient theoretical or empirical analysis to explain why existing methods fail and why the proposed reformulation, which frames OSTC as a subset-conditioned text generation task, is essential or fundamentally different. As a result, the paper does not convincingly establish the underlying rationale or mechanisms that support its claimed contributions. The main concerns are summarised as follows:

* **Lack of clear evidence that the proposed reformulation is essential.**  
  The paper provides weak evidence that the subset-conditioned text generation reformulation is fundamentally important for open-set text classification. While the full framework shows improvements, the comparison setup is confusing and does not isolate the benefit of the reformulation itself.

  * In particular, the representation-learning components (contrastive learning and orthogonality constraints) can be readily applied to existing methods, and their effectiveness has already been demonstrated in prior work. As shown in Table 1 and Figure 4, removing these components (‘w/o CL’ or ‘w/o Orth’) significantly reduces performance, often below other LLM-based baselines, suggesting that most of the gains may stem from these well-known techniques rather than from the proposed reformulation.  
  * The Reflective Inference (Analogy-Augmented Self-Reflection) module also appears largely independent of the new formulation. It is introduced as an inference-time enhancement and could be attached to any model trained under different paradigms. Since this component does not interact with the subset-conditioned fine-tuning objective, its inclusion further obscures the specific contribution of the reformulation itself.  
  * To substantiate the importance of the new formulation, the authors should include **control experiments** to isolate its contribution more clearly. *For example*, they could apply the same contrastive, orthogonality, and reflection modules to standard generative fine-tuning or other LLM-based baselines (such as LLM-OOD or VOS) without the subset-conditioning step. Comparing such models with and without access to the subset of known labels would help determine whether the reformulation itself contributes meaningfully beyond these auxiliary components.  
  * Additionally, while the logit calibration component may depend on the new formulation, the paper also lacks a clear, controlled experimental analysis identifying which parts of the framework genuinely benefit from it.


* **Unfair and inconclusive comparison between LLM-based and BERT-based baselines.**  
  The paper emphasises that one of its main contributions is the application of LLMs to OSTC and reports that LLM-based baselines outperform BERT-based methods (Section 5.2). However, this comparison is not meaningful, as the models differ drastically in scale and architecture: the LLaMA-3.1-8B model contains billions of parameters, whereas the BERT-base models are two orders of magnitude smaller. The observed performance gap therefore likely reflects differences in model capacity and pretraining rather than an inherent advantage of the proposed generative fine-tuning paradigm.  

* **Lack of evidence for consistency across model backbones.**  
  The paper reports results on two LLM architectures (LLaMA 3.1-8B and Qwen 2.5-7B), but the baselines are only evaluated on LLaMA. As a result, it is unclear whether the proposed UnLLM framework consistently improves performance across different backbones or whether the observed gains are specific to a single model. To claim general applicability, the authors should include baseline comparisons on Qwen 2.5-7B or at least provide an analysis showing that the observed improvements are robust to architectural variations.

* **Unclear and unsupported motivation, both theoretically and empirically.**  
  The paper’s motivation is difficult to follow at both the conceptual and empirical levels. Several key statements, such as 'LLMs often exhibit overconfidence in their predictions' (Section 4.3) and 'we observe a misalignment between the LLM’s internal knowledge (representation space) and its outputs' (Section 4.2), are presented as empirical observations but are not supported by any systematic study or quantitative evidence. It remains unclear whether these issues actually occur under the presented OSTC setting, and no diagnostic analysis is provided to demonstrate their impact. Furthermore, the paper does not establish a clear theoretical or conceptual connection between these claimed problems and the proposed solutions. As a result, it is difficult to assess whether the introduced components are sufficient or even relevant to address the stated challenges.

### Questions
* At line 246, the authors state that the orthogonality loss minimises the projection of ID representations onto the principal subspace of outliers (Section 4.1, Eq. 8). However, minimising this projection only reduces the overlap between the two subspaces; it does not guarantee orthogonality or decorrelation in a strict sense. Could the authors clarify the theoretical rationale behind this formulation? Why is minimising the projection considered sufficient to ensure separation, rather than explicitly enlarging the null space of the outlier subspace or directly enforcing an orthogonality constraint between the ID and OOD representation spaces?

### Soundness
1

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
3

### Summary
This paper proposes UnLLM, an unknown-aware large language model designed for open-set text classification. The authors introduce a 3-fold method, including:
1) open-set generative fine-tuning to develop a K+1 classifier. 
2) OOD parameter calibration method to align the model’s internal cognitiion of the unknown with its outputs. 
3) analogy-augmented self-reflection mechanism to mitigate overconfidence.

Experimental results across six datasets demonstrate that UnLLM consistently achieves SOTA performance in OOD detection while maintaining competitive accuracy in ID classification.

### Strengths
* This paper targets an important problem (Open-Set Text Classification), and the proposed method is novel.

* Experimental results across six datasets demonstrate that UnLLM consistently achieves SOTA performance in OOD detection while maintaining competitive accuracy in ID classification.

### Weaknesses
1. The writing is not good enough, I find it hard to follow the insights of the proposed method and the details of the equations. For example,
* What are the insights behind adding contrastive learning loss?
* In Equation 2, the authors say $N_{y_{i,j}$ is the number of examples of $y_{i,j}$, but what happens when the model is optimized in batch?
* Why authors could "assume that the representation for each class k ∈ [1,K] follows a class-conditional
Gaussian distribution:"?

2. I have a concern about section `Calibration Direction Construction`, where authors propose to "evaluate the fine-tuned LLM on the label partitioned validation set". I don't think using statistics of validation set is a fair comparison to other baselines.

3. The proposed 3-fold methods seem to have no relation, which makes me feel that LLM is too complicated.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
