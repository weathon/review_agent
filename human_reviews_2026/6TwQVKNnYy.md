# Contrastive Vision-Language Learning with Paraphrasing and Negation

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Contrastive vision-language models continue to be the dominant approach for image and text retrieval. Contrastive Language-Image Pre-training (CLIP) trains two neural networks in contrastive manner to align their image and text embeddings in a shared latent space. Recent results evaluating CLIP on negated or paraphrased text have shown mixed performance because negation changes meaning radically with minimal lexical changes, while paraphrasing can create very different textual expressions with the same intended meaning. This poses a significant challenge for improving the evaluation results and alignment of vision-language models. To address this challenge, this paper evaluates the combination of paraphrasing and negation, proposes a new CLIP contrastive loss function accounting for both paraphrasing and negation, and applies LLM-generated training triples consisting of original, paraphrased and negated textual captions to CLIP-like training models. The approach, called SemCLIP, is shown to move paraphrased captions towards the original image embeddings while pushing negated captions further away in embedding space. Empirically, SemCLIP is shown to be capable of preserving CLIP's performance while increasing considerably the distances to negated captions. On the CC-Neg benchmark using an original over negation image-retrieval accuracy metric, SemCLIP improves accuracy from 68.1% to 78.1%. Although results are mixed when compared with CLIP on the Sugarcrepe++ benchmark, SemCLIP's performance is generally better than the models trained with negated captions. This robustness to negation extends to downstream zero-shot classification tasks where SemCLIP pre-trained on Sugarcrepe++ performs better than CLIP on all tested downstream tasks. These results indicate that SemCLIP can achieve significant robustness to semantic transformations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
**Summary:**

This paper proposes SemCLIP, an extension of CLIP that jointly models paraphrasing and negation to improve semantic robustness in vision–language learning. It introduces new paraphrase and negation losses within a low-dimensional projection subspace to align equivalent captions and separate contradictory ones. Experiments on CC-Neg and Sugarcrepe++ show SemCLIP preserves CLIP’s retrieval accuracy while improving robustness to negation and linguistic variation.

### Strengths
**Strengths:**

Authors tackle an important problem of negation in multimodal retrieval.

### Weaknesses
**Weaknesses:**

- CLIP is now outdated and many new multimodal models perform much better than CLIP. See MMEB leaderboard (V1) and the models on it.
	- Most of these models are expected to be very robust to paraphrases.
- Comparison with ConCLIP, NegCLIP and ParaCLIP missing.
- Missing Ablations:
	- What is the need for extra projection layer? Ablations need to be performed.
	- Why not use a contrastive loss with the new (anchor, paraphrase, negative). Why use two seperate losses? Ablation needs to be performed.
- Writing needs to be improved:
	- "However, large multimodal models underperformed relative to LLMs" - needs citation
	- Lines 70-72: Citation/Evaluation missing. Does clip underperform on these examples?
	- Lines 87-88: What above findings?

### Questions
See Weaknesses

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The problem of improving CLIP training is considered. The paper introduces SemCLIP incorporating a dedicated embedding projection space and a combined loss function​, that includes components for paraphrasing (L_paraphrase​) and negation (L_negation​) alongside the standard contrastive loss (L_contrastive​). SemCLIP aims to move paraphrased captions closer to the original image embeddings while pushing negated captions further away, leading to a more robust semantic alignment between text and image. Experimental results, particularly on the CC-Neg benchmark, show that SemCLIP preserves CLIP's original performance while increasing the distance to negated captions, and this robustness extends to downstream zero-shot classification tasks.

### Strengths
N/A

### Weaknesses
1. Lack of technical novelty. It is not a new idea to finetune CLIP with negation data or paraphrasing data.
2. Lack of comprehensive evaluation. The proposed SemCLIP model was only evaluated on  two compositionality benchmarks and 5 classification benchmarks (CIFAR-10, CIFAR-100, FOODS101, FLOWERS102, OXFORD Pet). This is clearly insufficient to evaluate a CLIP model. Evaluation on more benchmarks (e.g. VTAB+ for classification, COCO/Flickr for text-image retrieval) is necessary for a solid paper.

### Questions
What is the cost for collecting synthetic caption data by LLMs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper, SemCLIP, proposes an extension to the CLIP framework to jointly address model robustness against two critical semantic transformations: paraphrasing (equivalence) and negation (contradiction). The approach uses a new combined contrastive loss incorporating $L_{paraphrase}$ and $L_{negation}$ terms applied to LLM-generated training triples.

### Strengths
The paper proposes to jointly model the two opposing yet critical semantic transformations—equivalence (paraphrasing) and contradiction (negation)—within a single unified contrastive learning framework. This approach is intriguing, establishing a necessary research direction for exploring the holistic semantic robustness of multimodal models.

### Weaknesses
* Despite the joint objective, paraphrase robustness does not improve: on SCPP, SemCLIP underperforms the CLIP baseline ($53.1\\%$ vs. $60.0\\%$), and on CC-Neg paraphrase it trails a "Paraphrase-only" variant ($21.0\\%$ vs. $23.0\\%$). This pattern suggests a practical tension between the attractive force of $L_{\\text{paraphrase}}$ and the repulsive force of $L_{\\text{negation}}$.

* Although negation robustness improves, it remains far from CoN-CLIP ($\text{CC-Neg Acc}_{\\text{neg}}$ $78.1\\%$ vs. $99.70\\%$), with substantial downstream zero-shot classification drops ($\approx 20$-$30$ p on Foods-101, Flowers-102, etc.). This questions the competitiveness of the projection-based loss for contradiction.

* The paper lacks a mechanistic account of how the low-dimensional projection reconciles opposing forces in the joint objective. Moreover, restricting loss weights $\\alpha, \\gamma$ to $\\{0, 1\\}$ precludes assessing trade-offs; a continuous search $0 < \\alpha, \\gamma < 1$ is necessary to demonstrate optimality and robustness of conclusions.

### Questions
* The paper needs a mechanistic account (e.g., visual or mathematical analysis) demonstrating how the low-dimensional projection successfully disentangles the competing $L_{paraphrase}$ and $L_{negation}$ forces, as their conflict seems to cause performance degradation on paraphrasing.

* Paraphrasing accuracy dropped below the CLIP baseline. What is the root cause of the conflict between the opposing $L_{paraphrase}$ and $L_{negation}$ forces, and how does the projection space explicitly mitigate this tension?

* Since loss weights were restricted to $\{0, 1\}$, a continuous parameter grid search is warranted to find the optimal balance.

* Given the vast gap to CoN-CLIP ($Acc_{neg}$ 78.1% vs. 99.70%), what is the fundamental limitation of the projection-based loss that prevents achieving competitive performance?

* The substantial performance drop on challenging downstream tasks (e.g., Foods 101, Flowers 102) suggests a failure in generalization. Please provide insight into why the learned semantic robustness does not effectively transfer to more complex, fine-grained visual recognition and compositional reasoning tasks.

### Soundness
1

### Presentation
3

### Contribution
2
