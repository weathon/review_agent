# Reversible Primitive–Composition Alignment for Continual Vision–Language Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Vision-language (VL) models are increasingly deployed in non-stationary settings, yet under sequential adaptation they often preserve primitive recognition while losing compositional structure, especially with tight rehearsal budgets and no task IDs. We address this gap by asking how a continual VL system can maintain structurally dependable behaviour while safeguarding zero-shot performance. We introduce Compo-ReAlign, a structure-first recipe built around three components: a reversible composer that maps primitive embeddings to compositions by design, a multi-positive InfoNCE that jointly aligns textual and composed views of the same target, and a spectral trust region that clips updates when alignment sensitivity inflates.  Across compositional DIL and multi-domain MTIL retrieval, Compo-ReAlign sets a new state of the art, improves over the strongest prior by +2.4 R@1, and reduces forgetting by 40%. We provide a compact, reversible alignment head with geometry-aware training for compositionally robust VL continual learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles continual VLM’s tendency to remember primitives (attributes/objects) but forget their compositions. It proposes COMPO-REALIGN: a tiny, plug-and-play head with a reversible composer (orthogonal via Cayley), a multi-positive InfoNCE tying textual and composed views, and a spectral trust-region to stabilize alignment—backbones stay frozen. Across compositional retrieval and continual VQA, it achieves SOTA averages with higher compositional retention and lower forgetting at <1% extra params.

### Strengths
1）  Clear structure-first formulation with explicit reversibility. Turning the composer’s core orthogonal via the Cayley map makes invertibility a design property rather than a penalty—clean and principled. 

2）  Minimal yet effective. The single multi-positive InfoNCE neatly aligns textual and composed views; spectral clipping acts as a geometry safety valve rather than an accuracy crutch. 

3）Practical deployment appeal. Frozen encoders, <1% extra params, and text-centric micro-buffers as structural anchors that are more memory-efficient than images; multilingual and template-form robustness are demonstrated.

### Weaknesses
1） Scope beyond retrieval/VQA? The current streams are comprehensive, yet mainly retrieval/ITM and VQA. How would the reversible composer behave for generation-style continual tasks or dense VL (e.g., referring expressions) where composition interacts with decoding? 
2） The method assumes a stable primitive inventory. If primitives are noisy, overlapping, or evolve over time, do the reversibility guarantees or CRR improvements hold? Could the composer adapt as the primitive set grows? 

3）Spectral thresholding choices. Spectral clipping stabilizes late-layer spikes, but how robust are results to γ and the accuracy of σ̂_max estimation with few power iterations, especially under longer streams? Any auto-tuning strategies? 

4）Decoding primitives from compositions at scale. The identifiability/CRR analysis and top-m readout are compelling; Are there any failure cases where coherence assumptions break down?

### Questions
Please refer to Weakness

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
This paper addresses the critical problem of compositional forgetting in continual vision-language learning, where models retain knowledge of primitive concepts but lose the ability to bind them together correctly. The authors introduce COMPO-REALIGN, a novel and parameter-efficient head for frozen vision-language models. The method is built on three core ideas: 1) a reversible composer that maps primitive embeddings to a compositional embedding via an orthogonal transformation, ensuring invertibility by design; 2) a multi-positive InfoNCE objective that aligns the image with both the textual composition and the synthesized compositional embedding, implicitly enforcing structural consistency; and 3) a spectral trust region that stabilizes alignment geometry by clipping parameter gradients when local sensitivity becomes too high. Through extensive experiments across three distinct continual learning tracks (compositional DIL, multi-domain retrieval, and continual VQA), the authors demonstrate that COMPO-REALIGN achieves new state-of-the-art results, significantly improving performance while substantially reducing forgetting and preserving compositional structure.

### Strengths
1. The paper's primary strength lies in its novel formulation of a significant, real-world problem. It shifts the focus from standard catastrophic forgetting to the more nuanced and practical challenge of preserving fine-grained compositional structure. This is a crucial step towards building more reliable and robust VLMs for dynamic environments. The proposed solution is highly original, particularly the concept of enforcing "reversibility by design" through an orthogonal composer and stabilizing alignment geometry via a spectral trust region. This geometry-first approach is a novel and powerful paradigm for continual learning.

2. The technical quality of the work is outstanding. The proposed method, COMPO-REALIGN, is theoretically well-motivated and elegant in its simplicity. The synergy between the three components—the composer, the objective, and the stabilizer—is clearly articulated and justified. The experimental validation is exceptionally thorough and rigorous. It includes multiple, diverse benchmarks, a comprehensive set of strong baselines, detailed ablation studies (Table 3), and insightful mechanism validation (Section 5.4, Figures 4 & 5). The introduction and use of diagnostic metrics like Compositional Retention Ratio (CRR) provide a much-needed tool for quantifying the specific problem being addressed, adding a layer of depth to the analysis that is often missing in CL literature.

3. The paper is exceptionally well-written and organized. The motivation is clearly established through an exploratory study (Section 3), which effectively diagnoses the problem and guides the development of the proposed solution. The methodology (Section 4) is described with precision, and the mathematical formulations are clear and easy to follow. The figures and tables are informative and well-designed, effectively conveying the key results and insights. The overall narrative is compelling and logically structured.

### Weaknesses
1: The method is exclusively evaluated on a frozen CLIP backbone. While this is a common and practical setup for parameter-efficient continual learning, it limits the scope of the findings. In more challenging scenarios where the representation backbone itself must adapt (e.g., via full-model or partial finetuning), it is unclear how COMPO-REALIGN would perform or interact with backbone-level forgetting. The head might preserve compositional logic, but if the primitive representations from the backbone degrade, the overall system would still fail.

2. The current composer relies on mean-pooling to aggregate primitive embeddings before the orthogonal transformation. While effective for the attribute-object pairs and simple relations tested, this approach might not scale gracefully to more complex, hierarchical, or nested linguistic structures (e.g., "a man in a blue shirt standing next to the car that is parked behind the house"). A discussion on the limitations of mean-pooling and potential extensions would be beneficial.

3. The spectral trust region is a key component, but estimating the Jacobian spectral norm via power iteration, even if efficient, introduces implementation complexity and computational overhead compared to simpler regularization techniques. Although the authors state the overhead is <2%, a more detailed discussion on the trade-offs, especially in the context of much larger models or distributed training setups, would strengthen the paper's practical claims.

### Questions
1. Could you comment on the applicability of COMPO-REALIGN in a scenario where parts of the vision-language backbone are also finetuned? Do you foresee the core principles, particularly the spectral trust region, being adaptable to stabilize the backbone representations themselves, or is the method fundamentally designed as a post-hoc head for a fixed feature extractor?

2. The spectral threshold is a crucial hyperparameter for balancing stability and plasticity. Based on your sensitivity analysis (Figure 7), could you provide more intuition or a general guideline for setting this value? Does it depend more on the model architecture (e.g., ViT-B vs. ViT-L), the task nature (e.g., retrieval vs. VQA), or the data distribution?

3. Have you considered alternatives to mean-pooling, such as an attention mechanism, to create the initial composed representation? While this might add complexity, it could potentially handle more complex sentences and weigh primitives more intelligently. Would it be possible to integrate such a mechanism while preserving the desirable properties of the reversible orthogonal core?

4. Your theoretical analysis in Appendix D provides a compelling lower bound on CRR under certain coherence assumptions. How does the proposed training scheme, specifically the use of a text-centric buffer with diverse examples and the spectral stabilizer, actively help the model satisfy these low-coherence and small-perturbation assumptions in practice?

### Soundness
3

### Presentation
2

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
This paper addresses the challenge of continual vision-language learning. The authors propose a novel method, COMPO-ReALign, designed to improve compositional generalization and structural robustness under stringent memory constraints and in the absence of task identifiers. Both empirical results and theoretical analysis are provided, making the study compelling and convincing.

### Strengths
1. Clear motivation. The paper demonstrates that current vision-language (VL) models tend to forget compositional structures and proposes three diagnostic approaches to address this issue.
2. Strong performance. The method achieves state-of-the-art results across several benchmarks.
3. Theoretical analysis. Under certain assumptions, the paper provides a theoretical analysis to support its findings.

### Weaknesses
1. Weak theoretical assumption. In Line 1056, the paper directly assumes the range of coherence $\mu \lt \frac{1}{2m-1}$ without providing a detailed analysis or justification in the empirical results.
2. Incomplete evaluation of spectral clipping. Since spectral clipping requires the calculation of Jacobian spectra, it is necessary to report detailed training time. The current version only briefly mentions this in Line 684, which is insufficient.

### Questions
1. Can the authors provide evidence or justification to support the theoretical assumption mentioned in Weakness 1?
2. Can the authors report detailed training time for spectral clipping, as highlighted in Weakness 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Brief Summary: The paper tackles the task of vision-language continual learning. The authors show through the exploratory study (section 3) that in continual streams, the primitive understanding (such as attributes/objects) remain stable but compositional aspect particularly zero-shot drops significantly. This the core motivation behind their method Compo-Realign which is essentially an alignment head with reversible composer, trained with info-nce loss, and clipping function. Training is done in a task-free manner. 

Experiments on multiple tracks such as composition, multi-domain retrieval, and continual VQA show proposed Compo-Realign significantly outperforms existing baselines by 2-5 points.

### Strengths
Pros:

1. The investigation and the key problem of losing compositionality but not single-attributes/primitives makes sense to me. The Compo-Realign method is well motivated, albeit in a constrained setting of Lora adaptation. 

2. The proposed Compo-Realign method outperforms existing strong and recent baselines (Table 2), and authors provide single-factor ablation studies (Table 3), showing the benefits of composed positive in both retrieval and vqa settings. A range of datasets are considered including both synthetic and real images. Different downstream-tasks are considered such as composition, vqa, retrieval.

3. The authors have detailed ablation and analysis with visualization graphs in both main paper and supplementary. In particular, detailed ablations on parameter sensitivity (fig7), and order sensitivity (fig8) are appreciated.

### Weaknesses
Cons: 

1. My main concern is that the paper mostly looks at frozen encoders (CLIP) with lightweight heads and not other options such as:

(i) full-fine-tuning (also noted in Conclusion section under future work).

(ii) other backbone other than CLIP, such as SigLIP or SigLIP2

(iii) non-ViT CLIP (such as ResNet based) even though ViT-CLIP is the more commonly used model

2. Similar to above point, while the authors are looking into Continual Vision-Language learning, generative models such as VLMs are not really considered. 

3. Currently, the authors assume the primitive set is already known but that might not be the case for free-form text. To my understanding, the closest would be vg-attr but that is only object + attribute pair and not free form text. Perhaps the COCO/Flickr settings are relevant? But it isn't clear how the primitives are being extracted from entire coco-captions/flickr-captions?

4. The overall improvement over existing baselines is somewhat marginal (+2 points on average recall @1). 

---

Overall Rating: 6/10
The paper provides an interesting setup and investigates the issue of retaining primitives but forgetting composition. The authors provide a very reasonable solution and show empirically it improves over existing baselines. The paper can be significantly strengthened by additional experiments with more backbone models, using VLM generative setup, experiments with image-text continual learning setup.

### Questions
Q1. For ablation in Table 3, instead of spectral trust region clipping, what happens if we do naive fixed clipping?

Q2. In table 3, it seems text-buffer plays a significant role (2.5 points), and noted in L321. Can the authors expand why this is the case? Then it would mean, text replay is the most useful not the reversible mapping as the original claim? Maybe I am misunderstanding something here.

### Soundness
3

### Presentation
2

### Contribution
3
