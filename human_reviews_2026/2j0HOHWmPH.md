# Memory Self-Regeneration: Uncovering Hidden Knowledge in Unlearned Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 8, 2

## Abstract
The impressive capability of modern text-to-image models to generate realistic visuals has come with a serious drawback: they can be misused to create harmful, deceptive or unlawful content. This has accelerated the push for machine unlearning. This new field seeks to selectively remove specific knowledge from a model's training data without causing a drop in its overall performance. However, it turns out that actually forgetting a given concept is an extremely difficult task. Models exposed to attacks using adversarial prompts show the ability to generate so-called unlearned concepts, which can be not only harmful but also illegal. In this paper, we present considerations regarding the ability of models to forget and recall knowledge, introducing the Memory Self-Regeneration task. Furthermore, we present MemoRa strategy, which we consider to be a regenerative approach supporting the effective recovery of previously lost knowledge. Moreover, we propose that robustness in knowledge retrieval is a crucial yet underexplored evaluation measure for developing more robust and effective unlearning techniques. Finally, we demonstrate that forgetting occurs in two distinct ways: short-term, where concepts can be quickly recalled, and long-term, where recovery is more challenging.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Memory Self-Regeneration (MSR), a novel framework for analyzing how unlearned diffusion models can “self-recover” forgotten concepts. The proposed method, MemoRa (Memory Regeneration with LoRA), combines DDIM inversion, spherical interpolation, and LoRA fine-tuning to regenerate erased concepts using only a few images. The authors also introduce the notion of short-term and long-term forgetting to model differences in recoverability. Experiments demonstrate that several existing unlearning methods fail to truly erase knowledge.

### Strengths
1. Prior work mainly focuses on whether forgetting is achieved, not whether forgotten concepts can be reactivated. This paper drawing analogies between human short/long-term memory and model forgetting offers a conceptually rich and interdisciplinary framing.

2. Quantitative results (FID, CLIP, NudeNet detection, ASR) and visualizations consistently show that MemoRa can rapidly restore “forgotten” concepts, verifying the persistence of hidden knowledge.

3. The study addresses a crucial issue in model safety and compliance — whether “machine unlearning” truly eliminates harmful or private content.

4. The paper is well-organized and clearly written. Figures (especially Fig. 1–2) effectively illustrate conceptual and procedural flow.

### Weaknesses
1. The paper primarily demonstrates empirical findings but lacks a formal analysis explaining why the model retains recoverable memory at a representational level (e.g., through weight-space or manifold topology).

2. It remains unclear whether MemoRa reveals true residual memory or simply relearns from scratch given minimal samples. The “self-regeneration” claim would benefit from ablation comparing to training from random initialization.

3. While the paper introduces short- vs long-term forgetting qualitatively, it does not propose a quantitative metric to measure the degree of residual knowledge.

4. Experiments focus on Stable Diffusion v1.4; extension to other architectures (e.g., text-to-video, large-scale T2I foundation models) would strengthen generality.

### Questions
See Weaknesses

### Soundness
2

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
This paper investigates the phenomenon that diffusion models may retain residual knowledge even after being “unlearned.” The authors propose a new evaluation task, Memory Self-Regeneration (MSR), and a corresponding LoRA-based method called MemoRa. The idea is to show that a few samples can “re-trigger” previously removed concepts from an unlearned model. The paper further introduces a distinction between “short-term forgetting” and “long-term forgetting,” drawing an analogy with human memory mechanisms.

### Strengths
The paper explores an interesting topic, forgetting recovery, which is important for safety and compliance of diffusion models.

### Weaknesses
1. Regarding the definition of short-term and long-term forgetting. I don't think this can be defined as short-term and long-term forgetting. When unlearning a diffusion model, if the number of forgetting training steps is sufficient, the model parameters deviate significantly from their initial positions, resulting in forgetting that is difficult to recover. Conversely, if the number of forgetting training steps is small or only the inference process is modified, the movement of model parameters is minimal, making the forgetting easier to recover.
2. Regarding MemoRa, fine-tuning is performed using datasets created with Lora and interpolation. Interpolation increases the number of images related to the concept, and as the volume of image data for the relevant concept grows during the training process, the model becomes increasingly likely to revert to its state before forgetting the concept. This is merely one way of generating data, with a relatively minor contribution.
3. The level of FID indicates the degree to which the model has forgotten its training. Therefore, can FID reflect the extent of "memory recovery"?
4. It appears that the use of spherical interpolation is no different from ordinary latent data augmentation, but the authors did not explain why they chose this method, nor did they provide ablation comparisons.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
The manuscript studies memory self-regeneration, focused on  analyzing knowledge recovery mechanisms in models, with particular 
emphasis on their ability to recall information that has been previously unlearned. The proposed method aims to recover unlearned
information using only a few images containing removed concepts.

Two distinct modes of model forgetting are studied: a short-term form, where concepts can be quickly recalled, and a long-term  form, where recovery is slower and demanding.

The manuscript aims to answer the question whether the unlearned diffusion models are capable of self-regenerating forgotten information?

The manuscript proposes MemoRa, a strategy for recalling knowledge in unlearned models, with a particular focus on approaches based 
on Low-Rank Adaptation (LoRA). Spherical interpolation is used to expand the latent space in order to generate more relevant images.

### Strengths
* The manuscript provides a study in models being able to recall information after unlearning.
* It provides a simple solution for recalling past information.
* It provides extensive experimental results.

### Weaknesses
* The manuscript lacks a theoretical justification that guarantees the recovery of information or provides bonds whether the information can
be recovered.

### Questions
What happens that unlearning and relearning are repeated, like in a process of continual learning?

### Soundness
3

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
This paper introduces Memory Self-Regeneration, a task for evaluating the robustness of machine unlearning methods in text-to-image diffusion models. The authors also propose MemoRa, a recovery strategy that uses DDIM inversion combined with LoRA to relearn supposedly erased concepts using only a few reference images. Through experiments on Stable Diffusion 1.4 with various unlearning methods, they demonstrate that many unlearning approaches exhibit "short-term forgetting" where concepts are easily recovered, while others achieve "long-term forgetting" that is more resistant to recovery.

### Strengths
- The paper provides extensive experiments across multiple state-of-the-art unlearning methods (ESD, FMN, MACE, SalUn, AdvUnlearn, etc.) and diverse concept types (nudity, objects, artistic styles, celebrities), offering valuable insights into the brittleness of current approaches.
- The MSR task provides a concrete framework for evaluating unlearning robustness that could benefit the research community.
- The MemoRa strategy demonstrates that recovery can occur with minimal computational resources (only 6 images, ~15 minutes of LoRA training), which is an important finding for understanding real-world vulnerabilities of unlearned models.
- The AutoMemoRa variant shows consideration of practical trade-offs by addressing the FID degradation issue, and Multi-MemoRa demonstrates the approach's extensibility to multiple concepts.
- The evaluation setup is well-documented with appropriate metrics (FID, CLIP, NudeNet, ResNet-50 classification) and includes both standard generation and adversarial attack scenarios.

### Weaknesses
- Limited The proposed MemoRa strategy is a straightforward combination of existing techniques (DDIM inversion + spherical interpolation + LoRA fine-tuning) without algorithmic innovation. The paper does not explain why this specific combination is optimal or provide ablations justifying each component. Other works have explored more sophisticated approaches to memory extraction in diffusion models, such as textual embedding-based methods [1].
- All experiments are conducted exclusively on Stable Diffusion 1.4. The paper provides no evidence that findings generalize to modern, transformer-based architectures like SDXL, SD 3.0, FLUX or Sana. 
This is a critical limitation given that architectural choices significantly impact both unlearning effectiveness and memory retention.
- The short-term vs. long-term memory analogy borrowed from neuroscience reduces to "easy to recover" vs. "hard to recover" in practice. The paper does not provide mechanistic insights into what causes these differences beyond speculation about manifold displacement (Section 3, lines 116-123). The hypothesis that short-term forgetting corresponds to "moving away from the manifold" while long-term forgetting reflects "displacement along the manifold" lacks empirical validation through representation analysis or ablation studies.
- The paper does not discuss recent work on quantization-based vulnerabilities in unlearned LLMs [2], which shows that forgotten knowledge can re-emerge even without explicit retraining, as well as alternative memory extraction approaches in diffusion models using textual embeddings [1] and maybe theoretical work on why approximate unlearning methods fundamentally struggle with complete erasure.
- Only LoRA-based recover is explored. Comparison with alternative approaches like full fine-tuning, embedding optimization or prompt tuning would strengthen claims about unlearning brittleness.
- The paper identifies a problem but offers limited guidance on how unlearning method developers should respond. Should they optimize directly for MSR robustness? If so, how? The conclusion states that MemoRa "struggles when the erased concept has been more deeply replaced" but doesn't explain how to achieve this "deeper replacement."


While this paper addresses an important question about the reliability of machine unlearning, it suffers from critical limitations and lacks technical novelty, since it is essentially an empirical observation rather than a methodological advance. Additionally, experiments are confined entirely to SD 1.4, providing no evidence of generalization to modern models or different architectural families. The short-term vs. long-term memory framing, while conceptually appealing, reduces to "easy vs. hard to recover" without mechanistic insights or predictive value.
In its current form, this represents a useful but incremental empirical study that highlights known limitations of approximate unlearning rather than providing significant new insights or techniques.


[1] Kowalczuk et al., "Finding Dori: Memorization in Text-to-Image Diffusion Models Is Not Local"  
[2] Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization", ICLR 2025

### Questions
- Can you provide ablations on key design choices like number of seed images or DDIM inversion starting timestep (t=35 vs. other values)?
- The authors hypothesize that short-term forgetting involves "moving away from the manifold" while long-term forgetting involves "displacement along the manifold" (Section 3). Can you provide empirical evidence for this claim? Maybe using latent space visualization (t-SNE/UMAP of representations), distance to nearest training examples or analyzing intermediate layer activations.
- If MSR reveals that a method exhibits short-term forgetting, what should developers do? Can you propose modifications to unlearning algorithms that would make them more robust to MSR attacks?

### Soundness
2

### Presentation
3

### Contribution
2
