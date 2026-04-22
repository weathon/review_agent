# Finding Dori: Memorization in Text-to-Image Diffusion Models Is Not Local

- Avg Score: 4.00
- Decision: Reject
- Scores: 0, 6, 8, 2

## Abstract
Text-to-image diffusion models (DMs) have achieved remarkable success in image generation. However, concerns about data privacy and intellectual property remain due to their potential to inadvertently memorize and replicate training data. Recent mitigation efforts have focused on identifying and pruning weights responsible for triggering verbatim training data replication, based on the assumption that memorization can be localized. We challenge this assumption and demonstrate that, even after such pruning, small perturbations to the text embeddings of previously mitigated prompts can re-trigger data replication, revealing the fragility of such methods.
Our further analysis then provides multiple indications that memorization is indeed *not* inherently local: (1) replication triggers for memorized images are distributed throughout text embedding space; (2) embeddings yielding the same replicated image produce divergent model activations; and (3) different pruning methods identify inconsistent sets of memorization-related weights for the same image. Finally, we show that bypassing the locality assumption enables more robust mitigation through adversarial fine-tuning. These findings provide new insights into the nature of memorization in text-to-image DMs and inform the development of more reliable mitigations against DM memorization.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper shows that memorization in text-to-image diffusion models is not localized. It finds that pruning specific “memorization neurons” (as in NeMo or Wanda) only hides, rather than removes, memorized images—since adversarial text embeddings can still re-trigger those replications.

### Strengths
1. The paper presents compelling evidence that may overturn the previously assumed locality of memorization in diffusion models.

2. The writing is very clear.

### Weaknesses
I agree with some of the authors’ viewpoints — in particular, that memorization is indeed difficult to mitigate through pruning. The concept of locality may need to be considered in two parts: detection and mitigation. When memorization occurs, certain neural activations become significantly stronger, which can clearly be useful for detection. However, pruning these activations is not sufficient to mitigate memorization. We can observe strong compensatory effects — meaning that multiple neural structures (e.g., heads) can contribute to memorization. When one head or neuron is pruned, others can take over. Moreover, these neurons might even be dynamic rather than fixed.

That said, I still believe the authors’ method does not convincingly demonstrate that pruning fails to mitigate memorization. I do not think the second row of Table 1 adequately addresses this concern. The adversarial token embedding itself introduces a large amount of external information. It is possible that the image has already been forgotten, or that memorization has been mitigated, but the adversarial method injects external information, enabling the model to relearn the image during inference and thus generate it again. I recommend the author to refer this paper [1]. This gives a more clear explanation.

Table 4 in Appendix H.2 also supports this argument. For a non-memorized image, after 150 adversarial steps, the generated image’s SSCD score rises from 0.17 to 0.65 (± 0.06), only 0.05 below the threshold defined by the authors. This demonstrates that the proposed method fails to prevent memorization. And the adversarial token embedding privde large information for the model to learn during inference. Meanwhile, the template’s memorization SSCD reaches at most 0.65, which suggests that Nemo might indeed be helpful (although I also believe Nemo is not necessarily effective).

[1] Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models.

### Questions
See weakness.

### Soundness
2

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
In order to prevent diffusion models from generating replication of their training data and cause privacy issues, recent works have applied pruning based memory mitigation methods. This paper challenges the memorization locality assumption which is the fundamental of pruning based methods. They claimed that pruning methods do not fully erase the memory, and an adversarial text embedding may re-trigger the training data. They further show that memorization occurs at various places, and proposed a global fine-tuning method that overall solves the problem.

### Strengths
Strengths:
1. Meaningful observation. This paper provides an insightful questioning about the memorization locality assumption of pruning based mitigation methods, showing that the memory is not fully erased even after pruning. The observations related to memory have the potential to give awareness to later researchers studying memorization of diffusion models. This is the major reason I think it is worth accepting.
2. Clarity of the structure. The story line of the paper is clear, from the method inducing the observation, explaining the reason behind this observation, to the proposed new method based on this observation, which improves the reader’s experience.
3. Methodology. This paper presents an innovative method through adversarial embedding, adversarial embedding optimization and fine-tuning. The approach appears logical from my review, but there may be further work focusing on efficiency improvement.

### Weaknesses
Weaknesses:
1. Experiment fairness. In section 4.4 ‘our mitigation’ is optimized upon Dori’s adversarial embeddings, afterwards, it is compared to baselines again with Dori as in Table 2. I wonder if this experiment setting may cause an unfair situation to baselines. Since only ‘our mitigation’ is optimized on adversarial methods, the baselines have no chance to gain such advantage.
2. Supplementary experiments. if ‘our mitigation’ approach has another version without participation of adversarial embeddings, we may have a better comparison with the baseline similar to the format of Table 1, i.e. comparison with and without Dori, to present performance advantage comprehensively and fairly (which could also solve weakness 1).
3. Diagram issue. Figure 1, is the right part redundant with the left part (2) to (3). If so, we may merge the right part with the left and add the details between (2) and (3).

### Questions
Questions:
1. In 4.4 Approach, do we need to train a set of adversarial embeddings before each fine-tuning optimization? If so, would the time consumption of the training be a concern?
2. In 4.4 Approach, is the text embedding being optimized by both fine-tuning and Dori? Would that be a conflict or redundancy?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the memorization problem in diffusion models, which causes concerns about data privacy using these generative models. Specifically, this paper follows the line of work (NeMo and Wanda) that mitigates memorization by pruning the local weights responsible for triggering verbatim training data replication, and challenges their locality assumption. Using a series of adversarial experiments, the authors demonstrate that even after applying pruning-based defenses such as NeMo and Wanda, small perturbations in text embeddings can re-trigger verbatim generation of memorized training images. Based on this, the paper introduces Dori, an adversarial-embedding optimization procedure, to uncover these vulnerabilities and show that memorization is not localized in either text-embedding, activation, or weight space. Finally, it proposes adversarial fine-tuning, a global mitigation strategy that effectively removes memorized content without degrading model utility.

### Strengths
1. The paper is well-written and easy to follow. 
2. The paper provides a novel and practical perspective on memorization in text-to-image diffusion models by explicitly challenging the locality assumption underlying prior pruning-based mitigation strategies. 
3. The methodology is rigorous and comprehensive. 
4. The experimental setup is sound, and the metrics used are clearly defined and justified. The authors thoroughly compare Dori against NeMo, Wanda, SISS, and concept-unlearning baselines (ESD, Concept Ablation). The results are also impressive.
5. The practical significance of this paper is high, as it contributes towards better privacy-preserving diffusion models and calls into question the reliability of existing “local pruning” defenses.

### Weaknesses
1. **Model Scope.** The analysis focuses exclusively on Stable Diffusion v1.4, the only model with known memorized prompts. While the authors justify this choice, it limits claims of generality. Extending the evaluation to even a partially curated SD v1.5 or fine-tuned variants would strengthen the argument. Could the authors comment on whether non-local memorization might emerge differently in larger or more recent models (e.g., SDXL or FLUX)?
2. **Computational Overhead.** The proposed adversarial fine-tuning requires multiple rounds of adversarial embedding optimization and full-parameter updates, which can be expensive. The paper acknowledges this but could provide clearer quantification of runtime and memory overhead relative to pruning methods.
3. **Limited Theoretical Analysis.** While the paper empirically demonstrates non-locality, it does not provide a deeper theoretical explanation for why memorization becomes distributed in diffusion models.

### Questions
Please see the above sections for details. In addition, do the authors expect similar non-locality in non-text-conditional (unconditional) diffusion models?

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
4

### Summary
This paper investigates memorization in text-to-image diffusion models and argues that such memorization is non-local—that is, it cannot be mitigated by pruning a small set of neurons or weights. The authors introduce Dori, an adversarial embedding optimization method that can recover memorized images even after pruning defenses, and propose an adversarial finetuning strategy as a stronger mitigation. Experiments on Stable Diffusion 1.4 demonstrate that pruning-based defenses merely “hide” memorized content rather than removing it.

### Strengths
1. The paper addresses an important topic, memorization and privacy in diffusion models, which is highly relevant to model safety and responsible AI.
2. The paper is well-written and highlights an underexplored dimension of diffusion model memorization behavior.

### Weaknesses
1. **Unrealistic scenario.** The proposed setting is impractical in the real world. The proposed finetuning-based method seems to align with the model publisher’s side (those who wish to make their models trustworthy). However, the paper assumes a white-box adversary with full access to the source code, as explicitly mentioned by the authors. However, most real-world text-to-image (T2I) systems, such as Midjourney or ChatGPT, only expose text-level APIs, not model internals. Therefore, the claimed adversarial setting does not reflect realistic threat models or deployment scenarios. 

2. **Method is both unrealistic and impractical.** Even if a white-box API model existed, the proposed finetuning approach is infeasible for a model publisher. Finetuning has no practical value unless the memorized images are already known. Existing training-time approaches can prevent memorization from scratch, and inference-time approaches can operate without knowing whether a given prompt corresponds to a memorized sample. In contrast, the proposed finetuning method requires prior knowledge of memorized images—which is itself extremely impractical to obtain. For instance, prior works [1,2] identify memorized images by exhaustively scanning and regenerating huge datasets such as LAION-2B. Thus, using the proposed method would first require massive precomputation to find memorized images, before even performing the fine-tuning itself. The authors themselves acknowledge that they can only test on Stable Diffusion 1.4, precisely because memorized images are public only for that model. Moreover, Stable Diffusion 3 already shows that simple image deduplication in the dataset effectively prevents such memorization during training. Altogether, the method is highly impractical for any real API provider or model publisher.

3. **Questionable performance evaluation.** The authors report only SSCD and MR metrics in the main paper, while omitting another critical metric for memorization—CLIP score. In the appendix, their method actually achieves the lowest CLIP score among the compared methods. This discrepancy raises concern: Table 1 (pruning methods result) includes CLIP scores, but Table 2 (finetuning methods including their own method) omits them. This selective reporting appears intentional and undermines confidence in the claimed performance.

4. **Limited scalability and incomplete experimental coverage.** The paper evaluates only on Stable Diffusion 1.4, claiming that no other models have public memorized-image datasets. However, the dataset provided by Webster [2] (which the authors used) also includes prompts for Stable Diffusion 2—though smaller in number, they are sufficient for at least limited experiments. As someone who has used that dataset myself, I can confirm that while SD 2 has fewer memorized samples, it is by no means impossible to test. The authors’ decision not to include it weakens the generality of their conclusions.

[1] Carlini et. al., "Extracting training data from diffusion models," USENIX'23.

[2] Webster, "A reproducible extraction of training images from diffusion models," Arxiv.

### Questions
The fundamental question is when and where this method would be used. Given Weakness #1 and #2, the authors must clearly describe a realistic scenario in which their approach could be practically deployed.  At present, the paper feels like it constructs a problem that does not exist in real-world T2I systems.

### Soundness
2

### Presentation
2

### Contribution
2
