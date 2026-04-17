# Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Machine unlearning—the ability to remove designated concepts from a pre-trained
model—has advanced rapidly, particularly for text-to-image diffusion models.
However, existing methods typically assume that unlearning requests arrive all
at once, whereas in practice they often arrive sequentially. We present the first
systematic study of continual unlearning in text-to-image diffusion models and
show that popular unlearning methods suffer from rapid utility collapse: after only
a few requests, models forget retained knowledge and generate degraded images.
We trace this failure to cumulative parameter drift from the pre-training weights
and argue that regularization is crucial to addressing it. To this end, we study a
suite of add-on regularizers that (1) mitigate drift and (2) remain compatible with
existing unlearning methods. Beyond generic regularizers, we show that semantic
awareness is essential for preserving concepts close to the unlearning target, and
propose a gradient-projection method that constrains parameter drift orthogonal
to their subspace. This substantially improves continual unlearning performance
and is complementary to other regularizers for further gains. Taken together, our
study establishes continual unlearning as a fundamental challenge in text-to-image
generation and provides insights, baselines, and open directions for advancing safe
and accountable generative AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the problem of continual unlearning for text-to-image diffusion models, where forgetting requests are handled sequentally. The paper shows that popular unlearning frameworks suffer from degraded performance in the continual setting. The paper shows that the problem is due to cumulative parameter drift, where the model weights progressively drift from the original weights. To address this, the paper proposes several methods: regularization, selective fine-tuning, model merging and gradient projection. Experiments show that these methods significantly improve the model's ability to retain performance in the continual unlearning setting.

### Strengths
- The continual unlearning setting is a practical and important problem in the field of text-to-image unlearning.
- While the root cause being cumulative parameter drift is unsurprising, the paper provides clear empirical evidence and analysis of the phenomenon.
- Though tested on only two baselines (see weaknesses), I find that the experiments and results are thorough and provide good evidence of the author's claims.
- Proposed methods like regularization and gradient projection are simple yet general, making them easy to integrate with existing unlearning pipelines.
- Overall, the paper is clear, concise and well-structured.

### Weaknesses
- In itself, the proposed methods are not novel (regularization, gradient projection etc. are certainly not new), but applied to the setting of continual unlearning. 

- The experiments are conducted on two methods, ConAbl and EraseDiff. While these are representative, it is unclear how the findings of the proposed methods would generalize to other classes of unlearning algorithms. 

- The authors claim simultaneous unlearning is costly, but their proposed model merging is also costly given independent copies have to be unlearned. Have the authors compared the computational costs and whether model merging is more efficient than simultaneous unlearning?

- On the Taylor expansion of the loss in Sec 5.2, the loss is *upper-bounded* by $||\theta^* - \theta^\dagger||$, thus even if the RHS grows, it does not guarantee the retention loss grows. Hence the conclusion that "loss grows proportionally (up to a constant) to $||\theta^* - \theta^\dagger||$" seems somewhat inaccurate.

### Questions
- Given the combined results in Fig 8, what end-to-end 'default' recipe do the authors recommend for sequential unlearning? A short set of takeaways for practitioners would help solidify the insights.
- Have the authors considered a Fisher-weighted regularization approach from continual learning like in [1]? This would lie between full L2 regularization and Selective fine-tuning.
- How sensitive is performance of gradient projection to the auxiliary set of semantically similar concepts? Could a poorly chosen set harm retention or unlearning?
- Have the authors investigated the effect of the order of unlearning requests? For example, does unlearning a broad concept (e.g. "photorealism") early in the sequence have a different impact than unlearning a more specific one (e.g. "Van Gogh")?
- How do results differ on concepts that can be referenced explicitly (like objects) versus indirectly (e.g., styles by prompting for artist names, or using synonyms)?

[1] Heng, Alvin, and Harold Soh. "Selective amnesia: A continual learning approach to forgetting in deep generative models." Advances in Neural Information Processing Systems 36 (2023): 17170-17194.

### Soundness
3

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
3

### Summary
This work conducts the first systematic investigation of continual unlearning in text-to-image diffusion models, revealing that existing unlearning methods suffer from severe utility collapse due to cumulative parameter drift. To address this, the authors propose a set of regularization-based strategies that mitigate drift while remaining compatible with existing methods. They further introduce a semantic-aware gradient projection technique that constrains parameter updates to directions orthogonal to the target concept’s subspace, preserving related knowledge. Overall, these methods substantially improve continual unlearning stability and establish strong baselines for safe, accountable generative AI.

### Strengths
1. This paper presents an interesting and valuable study on continual unlearning in text-to-image diffusion models.
2. This paper is very well presented.
3. The paper conducts a detailed analysis of the challenges faced by continual unlearning in text-to-image diffusion models through a series of experiments.

### Weaknesses
The author emphasizes that this paper does not propose new algorithms but focuses on the analysis of continual unlearning. I have the following questions about this paper:

1. Compared to regular continual learning, what are the additional challenges of continual unlearning? Parameter drift and conceptual confusion have been extensively studied in continual learning. Results in figure 3 separate the unleaning target from the retention target, but can also be interpreted as follows: as the number of requests increases, the model forgets the required targets, leading to indiscriminate unleaning of all concepts.
2. Experimental findings indicate that object retention and style retention exhibit distinct patterns of forgetting, though further analysis of this phenomenon is lacking.
3. In the experiment shown in Figure 4, for sequential learning, is the sum of the update iterations for multiple requests the same as the update iterations for simultaneous learning?
4. What insights does this paper offer for future research on continual unlearning in text-to-image diffusion models. Given that the methods combined in this paper have long been applied in continual learning, does this imply that regularization techniques and model merging approaches designed to address continual learning issues can effectively tackle continual unlearning in text-to-image diffusion models?

### Questions
Minor concerns:

- The colors in Figure 3 are too similar, resulting in poor readability.

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
4

### Summary
The paper introduces continual unlearning for text-to-image diffusion models where removal requests arrive sequentially. It shows that popular methods degrade quickly in this setting due to cumulative parameter drift away from the pretrained weights, then proposes simple add-on remedies such as L1 or L2 update penalties, selective fine-tuning, model merging, and a semantic-aware gradient projection on cross-attention projections to protect nearby concepts. Experiments on an UNLEARNCANVAS-based benchmark report strong improvements in retention while maintaining unlearning effectiveness.

### Strengths
1. Clear problem definition of continual unlearning with precise requirements for erasing targets, preserving prior removals, and retaining unrelated capabilities, plus explicit metrics for unlearning accuracy and retention accuracy split into in-domain and cross-domain.

2.  Practical plug-and-play remedies that integrate with existing unlearning methods, including L1 or L2 update penalties, selective fine-tuning, and model merging via TIES, which reduce drift and improve retention. 

3. The gradient projection idea operates on cross-attention projections to protect nearby concepts and combines well with other regularizers for further improvements.

### Weaknesses
1. Sensitivity to choices such as the strength of L1 or L2 penalties, the top k percent for selective updates, and the number and selection rule for auxiliary concepts in gradient projection is not fully characterized.

2. Limited cost analysis for independent unlearning plus merging and for importance computation in selective tuning.

3. The paper does not discuss several closely related recent works that address multi-concept and efficient forgetting, such as Sculpting Memory: Multi-Concept Forgetting in Diffusion Models via Dynamic Mask and Concept-Aware Optimization (ICCV 2025) and ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning (ICLR 2025). These studies provide complementary perspectives on dynamic masking and pruning-based forgetting, and should be compared for completeness.

### Questions
1. How sensitive are the results to the regularization coefficient and top k selection used in selective fine-tuning and merging?

2. How are auxiliary concepts selected for gradient projection, and how many are required for stable performance?

3. The evaluation primarily relies on EraseDiff as the base method, which limits the generality of the findings. Incorporating other representative approaches such as ESD, SalUN, and AC (Ablating Concepts in Text-to-Image Diffusion Models) would provide a more comprehensive and convincing demonstration of continual unlearning behavior across different unlearning paradigms.

4. While the paper mentions that continual unlearning may degrade the model’s general generative ability, the current evaluation mainly tests unrelated or random objects and styles to measure retention. A more informative evaluation would consider semantically related concepts to the forgotten target. For instance, when unlearning the concept “cat,” it would be more revealing to measure how well the model retains the ability to generate “tiger,” “lion,” or “leopard,” which are close in the embedding or visual space. Moreover, the paper lacks a broader assessment of general generation ability on a large-scale benchmark such as MS-COCO using metrics like CLIP Score or FID, which are standard in diffusion model evaluation.

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
3

### Summary
Current machine unlearning methods typically assume that all deletion requests arrive simultaneously. However, in real-world scenarios, deletion requests are often sequential, a setting referred to as continual unlearning.
Existing approaches suffer from severe performance degradation under this setting, leading to both ineffective unlearning and the collapse of unrelated generation quality.
To address this, the authors systematically study continual unlearning for text-to-image diffusion models and propose three regularization-based strategies (update norm regularization, selective fine-tuning, and model merging), along with a semantic-aware unlearning method (gradient projection). These methods aim to mitigate parameter drift, improve retention of unrelated concepts, and minimize interference among semantically similar concepts.
Experimental results demonstrate that the regularization methods effectively alleviate the performance collapse problem, while the semantic-aware unlearning method achieves the most significant improvement in in-domain retention. Furthermore, it can be combined with other regularization techniques to achieve a better trade-off between unlearning effectiveness and image quality.

### Strengths
- Formally define and analyze continual unlearning in the text-to-image setting.
- Provides both theoretical and empirical insights into performance collapse due to parameter drift.
- Proposes modular regularization and semantic-aware techniques that can easily integrate with existing unlearning methods.
- Gradient Projection method effectively improves in-domain retention and reduces collateral forgetting.

### Weaknesses
- The study is limited to style and object deletions; it does not evaluate more practically relevant concepts such as NSFW, copyrighted, or identity-based content as previous works.
- All experiments are conducted on a single diffusion model within the UnlearnCanvas benchmark. The paper does not assess whether the proposed regularizers and gradient projection method generalize to other architectures or larger-scale diffusion models
- The benchmark setup relies on a limited base model and a relatively small, templated set of prompts for generation. This constrained setting—previously identified as a limitation of existing unlearning evaluation [1], which may not fully capture the diversity and complexity of real-world unlearning scenarios.
- While the proposed techniques (regularization, model merging, gradient projection) are well-motivated, they are largely adaptations or combinations of existing ideas.

[1] Ko, Myeongseob, et al. "Boosting alignment for post-unlearning text-to-image generative models." Advances in Neural Information Processing Systems 37 (2024): 85131-85154.

### Questions
In addition to the weaknesses,
- What's the computational overhead of different unlearning methods evaluated in the paper?

### Soundness
2

### Presentation
3

### Contribution
3
