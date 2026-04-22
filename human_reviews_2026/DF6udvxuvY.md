# From Pixels to Words -- Towards Native Vision-Language Primitives at Scale

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
The edifice of native Vision-Language Models (VLMs) has emerged as a rising contender to typical modular VLMs, shaped by evolving model architectures and training paradigms. Yet, two lingering clouds cast shadows over its widespread exploration and promotion: (-) What fundamental constraints set native VLMs apart from modular ones, and to what extent can these barriers be overcome?
(-) How to make research in native VLMs more accessible and democratized, thereby accelerating progress in the field. In this paper, we clarify these challenges and outline guiding principles for constructing native VLMs. Specifically, one native VLM primitive should: 
(i) effectively align pixel and word representations within a shared semantic space; (ii) seamlessly integrate the strengths of formerly separate vision and language modules; (iii) inherently embody various cross-modal properties that support unified vision-language encoding, aligning, and reasoning. Hence, we launch NEO, a novel family of native VLMs built from first principles, greatly narrowing the gap with top-tier modular counterparts across diverse real-world scenarios. With 390M image-text examples, NEO efficiently develops visual perception from scratch while mitigating vision-language conflicts inside a dense and monolithic model crafted from our elaborate primitives. We position NEO as a cornerstone for scalable and powerful native VLM development, paired with a rich set of reusable components that foster a cost-effective and extensible ecosystem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a novel monolithic large vision-language model, called NEO, supported by an improved rotary positional embeddings (RoPE) mechanism and multi-stage training. NEO is also backed by several existing techniques, such as hybrid attention masking, and shared FFN, attention and norm layers for vision and language modalities. With evaluations reported on several well-established benchmarks, the work aims to demonstrate the performance of the proposed method.

### Strengths
The primary strengths of the work could be listed as the following:

- The particular idea of leveraging an improved RoPE mechanism (called "Native RoPE" in the work) in the context of monolithic vision-language models is novel.
- The work includes a very thorough literature review with more than sufficient citations to contemporary works, even those released within the same year, which is much appreciated.
- Evaluation is performed on a decent number of benchmarks, and there are a number of ablations on the positional encoding strategy, which is also a great to have.

Relatively minor strengths of the work could be listed as the following:

- Overall structure and flow of thought presented in the work is decent.
- The figures are of very high quality and are visually appealing, though a bit crowded (see the minor weaknesses below).

### Weaknesses
The primary weaknesses of the work could be listed as the following:

**W1: Architectural and Training-time Adjustments Similarities with Existing Works:** The work borrows heavily from two existing works, EVE [A] and EVEv2 [B] in both of its architectural and training-time adjustments. In particular, sharing the norm layers, attention blocks and FFN blocks have been explored in [A], patch embedding and word embedding layers are nearly identical to [B], and the _native multi-modal attention_ is the standard practice in large VLMs [C, D]. Furthermore, the overall training strategy is nearly identical to [B], with the Stage 1 pretraining corresponding to [B]'s Stage 1 & 2.1, Stage 2 mid-training corresponding to [B]'s Stage 2.2. and Stage 3s matching. Normally, having these similarities would not be a major weakness if it was not for the narrative of the work, which, in its current form, appears to present these as novelties of the proposed framework.

**W2: Ambiguities in the Narrative:** There are several ambiguities in the narrative of the work. Most importantly, due to the aforementioned W1 the exact contributions of the work are not very clear to read from the text, as the current narrative renames a few well-established practices in the field in a rather ad-hoc manner. To exemplify, these include renaming the standard hybrid attention masking in the literature to "Native Multi-modal Attention" or renaming the monolithic blocks of [A, B] to "Native VLM primite" while the only architectural difference from [A, B]'s blocks is the improved RoPE mechanism and the added Q and K parameters that go along with it. Finally, I found it a bit hard to grasp the exact changes introduced over existing works in Section 3.1 in general, with several terms like "Pre-Buffer" not being well-defined.

**W3: Fairness of Evaluations:** The work indeed includes a good number of benchmarks and a good number of ablations trying out different RoPE variants within the same framework. However, one critical thing that is lacking is a fair comparison between [A, B] and this work. Notably, NEO utilizes a much better LLM compared to the baselines considered (Qwen 3 versus older Qwen 2.5/Vicuna variants) in Table 1 and it was also trained with much more data than many of them. Given NEO does not add much beyond the improved RoPE mechanism over [A, B] architecturally, a more fairer comparison would demand them trained with similar budgets or at least with similar performing LLMs.

Relatively minor weaknesses of the work could be listed as the following:

- In many parts of the text the usage of \citep and \citet commands were used incorrectly. This hinders the readability of the text for the broader audience and fixing them would greatly improve the reading experience.

- Some figures are very crowded with many details, creating potential confusions in grasping their main message. To exemplify, Figure 1 includes many different details regarding the full LVLM pipeline. Although its quality is very high and I can clearly see that much effort went into constructing it, I believe that pruning it greatly would make it easier for the reader to grasp its main message.

*Finally , although I am leaning towards rejection for the work in its current form, I would like to encourage the authors to clarify any potential misunderstandings I might have had.*

---
[A] Diao, H., Cui, Y., Li, X., Wang, Y., Lu, H., & Wang, X. (2024). Unveiling encoder-free vision-language models. Advances in Neural Information Processing Systems, 37, 52545-52567.

[B] Diao, H., Li, X., Cui, Y., Wang, Y., Deng, H., Pan, T., ... & Wang, X. (2025). Evev2: Improved baselines for encoder-free vision-language models. arXiv preprint arXiv:2502.06788.

[C] Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., ... & Zhai, X. (2024). Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726.

[D] Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., ... & Dai, J. (2024). Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 24185-24198).

### Questions
- Given the similarities in performance with Video-RoPE [E] and the proposed RoPE mechanism in this work (Table 3), how well the proposed RoPE mechanism compare against it under smaller training budgets, such as those utilized for [A, B]? 

- Can you comment on the fairness of evaluations raised from above? How do you think the differences in training and architectural settings could be effecting the evaluation results and how do you think you could address these?

---
[A] Diao, H., Cui, Y., Li, X., Wang, Y., Lu, H., & Wang, X. (2024). Unveiling encoder-free vision-language models. Advances in Neural Information Processing Systems, 37, 52545-52567.

[B] Diao, H., Li, X., Cui, Y., Wang, Y., Deng, H., Pan, T., ... & Wang, X. (2025). Evev2: Improved baselines for encoder-free vision-language models. arXiv preprint arXiv:2502.06788.

[E] Wei, X., Liu, X., Zang, Y., Dong, X., Zhang, P., Cao, Y., ... & Lin, D. (2025). VideoRoPE: What Makes for Good Video Rotary Position Embedding?. arXiv preprint arXiv:2502.05173.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel approach to train autoregressive, monolithic vision-language models that omit domain-specific vision encoders in favor of light-weight encoders and a native multi-modal training resulting in the *NEO* models.  The model consists of a small two-layer convolution encoder, a multimodal pre-buffer, and a pretrained LLM (Qwen3). The attention and position encoding procedure is optimized for multi-modality. The model training comprises of 3 stages with 390M image-text examples. The resulting 2.2B/9B models are thoroughly benchmarked against prior modular and monolithic VLMs, and outperform all prior models in the latter category.

### Strengths
- The paper proposes improvements in native, monolithic multimodal LLM training by introducing a pre-Buffer for better alignment (trained separately in the first stage), native rotary position embeddings with modality-specific base frequencies, native multimodal attention (causal for text, full-bidirectional for vision) with decoupled H, W, T processing. 
- Using 390M image-text examples, NEO reaches a high performance and outperforms all previous native VLMs.
- NEO is built on top of a modern LLM (Qwen3) and supports flexible resolution
- The authors provide NEO-2B and 9B intermediate and final checkpoints
- The models are thoroughly evaluated against other models, including modular and native models in 2B and 8/9B categories
- Some design choices (number of layers in the pre-Buffer and attention/embedding methods) are ablated

### Weaknesses
- I keep wondering why NEO-9B uses a 50% smaller pre-Buffer than 2.2B. The paper mentions "mainly due to the good trade-off between performance and efficiency." (L431) but does not provide evidence for that. I am not convinced that the results in Fig. 5 extrapolate to a larger post-LLM.
- Some systematic ablations of design choices are often missing. The number of layers in the pre-Buffer and attention/embedding methods are ablated but nothing else. This left wondering which design choices in NEO actually impact performance: e.g., is it the data (quality/quantity)? The stages? The more modern LLM (related work uses older LLMs)? Or, is it actually the proposed combination of design choices. I understand that providing such controlled ablation experiments might not be economically feasible, but they obfuscate the contribution nonetheless. 
- NEO still significantly lags behind modular VLMs, even older ones like Qwen2-VL (e.g., 16% on InfoVQA). Given the relative improvements in its category and the lower amount of data this is not a big issue itself, however the phrasing in "Comparison with Modular VLMs." (L357ff) is a bit overselling.
- The scaling improvements between 2B and 9B seem modest compared to modular VLMs, "casting shadows" over the scalability of NEO.
- Some parts of the paper feel LLM-generated by overusing (sometimes nonsensical) synonyms, making it hard to follow the paper. I would encourage the authors to manually revise the paper.
- Fig. 1/3 are densely packed and hard to comprehend.

### Questions
- Are there any insights why the performance on HallusionBench and "knowledge-heavy" tasks suffers? Fundamentally, this does not seem like a multi-modal problem to me.
- Please review the LLM written parts for clarity.
- Please consider using \citep to improve legibility

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present Neo, a family of native Vision-Language Models built on top of Qwen3-1.7B and Qwen3-8B. The key elements of the Neo architecture are (1) an attention block that decouples H, W and T dimensions for Query and Key computation; (2) RoPE position embeddings that use separate frequencies for H, W and T dimensions; (3) bidirectional attention for images; (4) added transformer layers ("Pre-Buffer") to project vision and text embeddings to the same embedding space. The authors train Neo models on 390M image-text examples in a process containing three stages and compare against prior modular VLMs using pretrained vision encoders and native VLMs. The trained models obtain competitive performance on various datasets against modular VLMs, while outperforming prior work on native VLMs.

### Strengths
- Each architectural component is motivated with the incorporation of inductive biases that make the processing of images in native VLMs more similar to modular VLMs, which I found to be intuitive.
- The decoupling of H, W and T dimensions in both the Query and Key computation as well for RoPE computation is novel to the best of my knowledge.
- The training process does not assume access to any pretrained vision encoders.
- The various ablation studies in Section 4.3 are very valuable in arguing for the importance of various design choices in the attention blocks, particularly of the suggested adjustment to RoPE. This is especially the case when all native VLM models compared against make use of different datasets or pre-trained LLMs.

### Weaknesses
While I believe the contribution is valid, in large part due to the ablation studies, there are various flaws to the paper:
- Firstly, there are clear errors in the related works section. We do not know the underlying architecture for multimodal GPT models and it is therefore incorrect to claim that they are modular or native. It is also important to note that GPT-4o, being able to both condition on and generate images, is more similar to a model like Chamelon (included in the native model section) than standard modular model counterparts. It is likewise an issue to make claims regarding Claude or Gemini.
- I would argue that saying Neo "rivals top-tier modular counterparts" in the abstract and introduction is overclaiming. For general vision benchmarks, Neo approaches modular model performance, but nonetheless falls short in each case. This is particularly an issue for MMMU and MMVet, where 8B model performance is at least ~20% poorer. This and the bullet point above are the rationale for the soundness score.
- Likewise, any comparison made to prior native VLMs has the confounder of Neo making use of the newer and stronger Qwen3 backbones. This is mitigated by the ablation studies, which are much welcome.
- I found the writing to be unclear, particularly for sections dealing with the "Pre-Buffer" and "Post-LLM." Some added formalism about what computation each of these components perform and which components are initialized from the Qwen3 backbone and which are not would make understanding easier. Figure 2 sadly is unclear as both the Pre-Buffer and Post-LLM components make use of the same primitives but just differ in color.

### Questions
- A majority of the citations should be changed to parentheticals, rather than in-text citations with \citet.
- As a follow-up to my point regarding the Pre-Buffer, I also struggled at understanding what exactly was done in the "Comparison between Pre-Buffer and Vision Encoders" section. Would it be correct to say that training was repeated here with InternViT/CLIP/SigLIP used in place of the Pre-Buffer?

### Soundness
2

### Presentation
2

### Contribution
3
