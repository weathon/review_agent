# ORION: Decoupling and Alignment for Unified Autoregressive Understanding and Generation

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Unified multimodal Large Language Models (MLLMs) hold great promise for seamlessly integrating understanding and generation. However, monolithic autoregressive architectures, despite their elegance and conversational fluency, suffer from a fundamental semantic–structural conflict: optimizing for low-level reconstructability in generation leads to catastrophic forgetting of high-level semantic understanding. We present ORION, a unified framework that resolves this conflict through Decoupling and Alignment. A non-linear vision head decouples structural pressures from shared representations, while a novel Representation Consistency Loss explicitly aligns semantics during generation. Together with a curated progressive training recipe and high-quality multimodal data, our method enables balanced optimization of both capabilities. Built purely on a monolithic autoregressive backbone without task-specific separate parameters, ORION achieves performance on par with or exceeding recent state-of-the-art unified models that rely on more complex designs. These results validate monolithic autoregression as a simple, effective, and competitive path toward truly integrated multimodal intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces ORION, a unified framework that resolves the conflicts between understanding and generation through two decoupled vision heads for semantical and generative objectives separately.
The core contribution is:
1. The authors explicitly define the "semantic-structural representation conflict" in unified autoregressive MLLMs.
2. To resolve the conflict, the authors decouple semantic-preserving and low-level generation objectives by two vision heads:
+ The semantic-preserving is realized by aligning the semantic prediction distribution using the raw language heads with a frozen MLLM teacher of the same initial weights, with a proposed Representational Consistency Loss, i.e., $\mathcal{L}_{KL}$.
+ The low-level generation objective is realized by aligning the outputs of an additional vision head with the input embeddings of the MLLM, i.e., $\mathcal{L}_{MSE}$.
3. The authors dedicatedly curated a progressive training pipeline to warm up the vision heads and conduct full-parameter training.

The authors conduct experiments on several benchmarks and show that ORION can achieve SOTA generation ability while preserving its base MLLM's understanding ability.

### Strengths
1. The paper writing is very clear and easy to follow.
2. The figures are easy to read.
3. All designs, i.e., the vision regression head and the $\mathcal{L}_{KL}$, are rationalized well.
4. Promising results on both understanding and generation benchmarks.

### Weaknesses
1. **Incremental to SEED-X without discussion**. 
This work adopts nearly the same architecture as SEED-X[1] without discussing the differences. Please correct me if I misunderstand something:
+ Both SEED-X and ORION receive native tokenized texts and ViT embeddings as the inputs.
+ For the text side, they are exactly the same, i.e., supervised by the Next-Token-Prediction objective.
+ For the vision side
    + SEED-X regresses the $N$ learnable queries with $N$ ViT embeddings using MSE to foster generation ability. They ignore the raw $N$ corresponding vision tokens because at that time, MLLMs are not well-developed, and the backbone is purely text-based, i.e., Llama-2. Therefore, the vision tokens do not have a decodable semantic space. 
    + Since this work adopts a pre-trained MLLM as the backbone, the $N$ vision tokens become meaningful (decodable), and the learnable queries are no longer needed. Therefore, ORION switches to directly regress the $N$ vision tokens using the same MSE loss. 
+ Both SEED-X and ORION fed vision outputs to a diffusion decoder to produce pixel output.

I recognize the contribution of the naturally extended $\mathcal{L}_{KL}$ objective because the vision tokens are decodable under the MLLM-as-backbone scenario. But for other parts, it looks like simply switching the regression target from $N$ queries (SEED-X) to $N$ raw vision outputs (ORION), which seems a bit incremental to SEED-X.

[1] SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation

2. **Overclaimed contribution.**
I do not agree with the authors' claim:

*"Conceptually, we are the first to identify and define the “semantic-structural representation conflict” in unified autoregressive MLLMs".*

There is a semantic-structural representation conflict between understanding and generation has become a consensus in the community. Several well-known works have already raised such an issue. For example, in Janus[1] (which is not cited by this paper):

```
The output of understanding task not only involves extracting information from images but also involves complex semantic reasoning. Therefore, the granularity of the vision encoder’s representation tends to mainly focus on high-dimensional semantic representation. By contrast, in visual generation tasks, the main focus is on generating local details and maintaining global consistency in the image. The representation in this context necessitates a low-dimensional encoding that is capable of fine-grained spatial structure and textural detail expression. Unifying the representations of these two tasks within the same space will lead to conflicts and trade-offs.
```

Many other papers regarding unified models have similar statements, so the authors should not claim they are **the first one** to discover it. 

[1] Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation

3. **Inaccurate description of Cascaded Architectures.**

In L115-L116, the authors comment:

*The MLLM acts as a text encoder to guide image synthesis. While leveraging specialized models, this design cannot comprehend its own visual outputs, precluding multi-turn interaction.*

But many MLLM-based unified models receive multimodal input instead of pure text. Though some query-based works, e.g., MetaQuery, cannot directly understand the query, they indeed can understand the output images. By combining the output image with previous conversations and new user instructions as a multimodal input, they can also realize multi-turn interaction.

4. **Misclaimed architecture type.**

The authors claim ORION is a monolithic autoregressive model, but the generation requires a separately trained diffusion model, i.e., Nexus-Gen. From Figure 1, there is no difference between the Cascaded Architectures (Left) and the Monolithic Autoregressive Models (Right). From my perspective, true Monolithic Autoregressive Models should not rely on external diffusion models, e.g., Transfusion[1] and Show-o2[2]. ORION should be categorized as a Cascaded Architecture. 

[1] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

[2] Show-o2: Improved Native Unified Multimodal Models

5. **Does not truly resolve the conflicts.**

+ From Table 3, we can observe that compared to Qwen2.5-VL-7B, the performance on most benchmarks, i.e., MMVet, SEED, and RWQA, shows a clear degradation, indicating that the semantic-structural representation conflict is not truly resolved. 
+ From Figure 4 Image Editing part, the results are quite similar to those of MetaQuery, where the objects and backgrounds are significantly altered. For instance, the background of the robot becomes much brighter, and the robot's shape is changed. For the second image, the buildings on the left are totally different after editing. This evidence suggests that the learned representations do not truly preserve low-level details; instead, they still focus on semantic accuracy, and the model is doing regeneration instead of editing.

6. **Potential benchmark hacking because of the usage of BLIP-3o training data**.

BLIP-3o training data is a highly controversial one in the community, because it directly distills the GenEval benchmark. Training on such a corpus and evaluating on GenEval is unfair for other baselines, e.g., Janus-Pro and Show-o2. Also, as a unified model, evaluating only on GenEval is not sufficient. Please consider evaluating on other benchmarks, including those non-semantic-oriented ones, to show the true generation quality.

### Questions
1. Can you add more discussion about the difference between your work and SEED-X?
2. Can you show results on:
+ Semantics-oriented benchmarks other than GenEval, for instance, WISE and DPG-Bench.
+ Non-semantics-oriented benchmarks, such as MJHQ-30K and MS-COCO, these benchmarks should reveal if ORION learns aesthetic-aligned representations from a low-level view.

**I am willing to raise my score based on other reviewers' comments as well as the rebuttal quality.**

### Soundness
3

### Presentation
4

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
The paper proposes ORION, a unified multimodal framework that keeps a monolithic autoregressive backbone while attempting to reconcile a central tension between semantic fidelity for understanding and structural reconstructability for generation. The method introduces two key pieces. First, a non-linear MLP vision head decouples low-level regression pressure from the shared backbone. Second, a representation consistency loss uses a frozen teacher to align the student’s text-prediction distribution at visual token positions, aiming to prevent semantic drift during generative training. The approach is trained with a progressive three-stage recipe that warms up the vision head, then performs full-parameter pretraining with an understanding-heavy mix, and finally multi-task supervised fine-tuning that includes editing. Experiments report competitive text-to-image scores on GenEval and strong preservation of understanding across MMBench, MMStar, MMVet, SEED-Bench, and RealWorldQA, alongside ablations that attribute gains to the MLP head and the consistency loss.

### Strengths
The paper isolates a concrete failure mode in unified autoregressive models, namely semantic drift during generative fine-tuning, and frames it as a representation conflict. The proposed remedy is conceptually simple yet targeted. The MLP head reduces direct gradient pressure on shared representations, and the representation consistency loss anchors semantics where cross-entropy supervision is absent. The training curriculum is pragmatic and clearly described, including data composition and hyperparameters, which improves reproducibility. Reported results indicate that ORION narrows the gap between monolithic models and more complex unified systems, while maintaining multi-turn interleaving abilities. The ablations are helpful and show that both the warm-up and the consistency loss matter, with measurable gains on understanding and generation metrics.

### Weaknesses
First, the representation consistency loss depends on a frozen teacher distribution at visual token positions. The paper should clarify whether the teacher sees exactly the same tokenization and masking, and whether temperature scaling or label smoothing is used. Without that detail, it is hard to judge stability and potential confirmation bias, since the student is nudged toward the teacher’s pretraining priors.
Second, the training recipe relies on large data mixtures that include regenerated images and recaptions. The paper should detail data licensing, deduplication against evaluation sets, and the exact share of synthetic captions. This is important for fairness and for assessing potential overfitting to benchmarks. 
Third, although the ablation table is useful, it collapses several factors at once. For example, Q-Former vs MLP comparisons change the warm-up corpus size, and the effect of the loss weights is not isolated. A factorial ablation on λ values and warm-up sizes would make the causal story more convincing. 
Finally, some claims about emergent multi-turn and cross-lingual behavior are illustrated qualitatively. These are compelling, but a small, controlled quantitative probe would help, for instance by measuring understanding accuracy on the model’s own generations and by using non-English GenEval-style subsets.

### Questions
1. Teacher details for the consistency loss. What teacher is used exactly, and does it share weights with the initialization of the student backbone. Do you match logits with a temperature or probabilities directly. How sensitive are results to the loss weight λKL and to masking strategies at visual token positions.
2. Data governance. Please specify data sources, licenses, and any filtering for copyrighted or sensitive content. Clarify whether FLUX regenerated images and Qwen recaptions appear in the test distributions and how you prevented leakage into GenEval or understanding benchmarks. 
3. Robustness and safety. Did you assess failure modes like hallucinated text in images, biased generations, or unsafe outputs during editing. Any guardrails applied during training or inference would be useful to report.

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
This paper proposes ORION, a unified autoregressive model that aims to reconcile the inherent conflict between semantic understanding and structural reconstruction in multimodal large language models (MLLMs). The core claim is that a "semantic-structural representation conflict" hinders monolithic architectures, which the authors address via "decoupling and alignment." Specifically, they introduce a non-linear MLP vision head to decouple structural prediction and a Representation Consistency Loss to align semantics during generation. The model, built on Qwen2.5-VL and a Nexus-Gen decoder, is evaluated on standard benchmarks, showing competitive results in both understanding and generation without task-specific parameters.

### Strengths
- Clearly formalizing the semantic-structural conflict in unified autoregressive models.
- The model is evaluated on a wide range of standard benchmarks for both understanding and generation.
- The paper is well-written and easy to follow.

### Weaknesses
- The paper fails to discuss and contrast its approach with other recent works that pursue a similar monolithic, autoregressive vision (such as GLaMM or Emu3).
- Presenting image editing as a result without quantitative evaluation.
- The ablation study does not full isolate the mechanisms of "decoupling" and "alignment."
- The paper lacks of discussion on training cost. The progressive three-stage training, while effective, appears computationally intensive.

### Questions
- Can you provide quantitative results on a standard image editing benchmark to support the claim "certain image editing capabilities."  
- Can you design an ablation that separates  "decoupling" and "alignment?" For instance, applying the KL loss to a model with a linear head to see if "alignment" alone can compensate for poor "decoupling."
- Can you provide a discussion of the total training cost compared to simpler fine-tuning or other unified models (e.g., parallel architectures).

### Soundness
3

### Presentation
3

### Contribution
3
