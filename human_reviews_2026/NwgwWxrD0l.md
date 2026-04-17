# STEER AWAY FROM MODE COLLISIONS: IMPROVING COMPOSITION IN DIFFUSION MODELS

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
We propose to improve multi-concept prompt fidelity in text-to-image diffusion
models. We begin with common failure cases—prompts like “a cat and a dog”
that sometimes yields images where one concept is missing, faint, or colliding
awkwardly with another. We hypothesize that this happens when the diffusion
model drifts into mixed modes that over-emphasize a single concept it learned
strongly during training. Instead of re-training, we introduce a corrective sampling 
strategy that steers away from regions where the joint prompt behavior overlaps 
too strongly with any single concept in the prompt. The goal is to steer
towards “pure” joint modes where all concepts can coexist with balanced visual
presence. We further show that existing multi-concept guidance schemes can operate 
in unstable weight regimes that amplify imbalance; we characterize favorable 
regions and adapt sampling to remain within them. Our approach, CO3, is
plug-and-play, requires no model tuning, and complements standard classifier-free
guidance. Experiments on diverse multi-concept prompts indicate improvements
in concept coverage, balance and robustness, with fewer dropped or distorted concepts 
compared to standard baselines and prior compositional methods. Results
suggest that lightweight corrective guidance can substantially mitigate brittle 
semantic alignment behavior in modern diffusion systems. Code is available at
https://github.com/debottam-dutta7/co3

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed the CO3, Contrasting concepts compose better, to improve muti-concept prompt fidelity in text-to-image diffusion models. Instead of re-training, the authors introduced a corrective sampling strategy, to steer away from regions where the joint prompt behavior overlaps strongly with any single concept in the prompt. Specifically, they analyze and show that composition through weighted sum of Tweedie-means in the Tweedie denoised space, offers a more general framework. Experiments on diverse multi-concept prompts demonstrate improvements in concept coverage, balance and robustness, with fewer dropped or distorted concepts compared to standard baselines and prior compositional works.

### Strengths
1)	The paper is well-written, and the structure is well-organized. 
2)	The proposed CO3 is plug-and-play, model-agnostic, and gradient-free. And the results shows stronger semantic alignment to prompts. 
3)	Combing the strengths of correction-based approaches and composable diffusion seem reasonable and effective.

### Weaknesses
1)	Some very latest and more relevant works are not compared, such as, Magnet (We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function), TWEEDIEMIX ICLR 2025, ConceptWeaver, etc. 
2)	It is confusing that why using arbitrary weights in (10) (i.e., Liu et al. 200) does not lead to a valid Tweedie-mean. There seems a lack of theoretical analysis or empirical validation.
3)	How to connect the equation (12) to (13). On other words, why (13) is representing the samping from the unnormalized probability distribution in Eq. (12).
4)	More experiments are tested on two concept prompts. Can this model generalize to the scenario with the prompts of three or more concepts?

### Questions
1)	Some very latest and more  closely relevant works are not compared, such as, Magnet (We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function), TWEEDIEMIX ICLR 2025, ConceptWeaver, etc. 
2)	It is confusing that why using arbitrary weights in (10) (i.e., Liu et al. 200) does not lead to a valid Tweedie-mean. There seems a lack of theoretical analysis or empirical validation.
3)	How to connect the equation (12) to (13). On other words, why (13) is representing the samping from the unnormalized probability distribution in Eq. (12).
4)	More experiments are tested on two concept prompts. Can this model generalize to the scenario with the prompts of three or more concepts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed a method for better prompt-aligned multi-concept image generation. Specifically, existing models often neglect or mix multiple concepts in a single generation scenario. The proposed method tends to squeeze the distribution, hindering the generation process from walking through some problematic regions. The probability normalization approach is designed for this purpose.

### Strengths
1. The diagnosis of the multi-concept image generation problem is interesting—specifically, the overlap of individual concept distributions and the original multi-concept distribution's regions.
2. The distribution correction idea is theoretically intuitive and good.

### Weaknesses
1. A known post-hoc guidance is called in a new name, “correction guidance”, for a better sound of the paper. It sounds unnatural.
2. Only Subjective metrics were used. What about objective ones, such as CLIP-T, DSG (Davidsonian Scene Graph), or simple classification metrics?
3. The proposed method is based on a modern and relatively stronger model, SDXL, while it is often compared with older methods with weaker diffusion models (2023 methods, such as Attend and Excite or Divide-Bind with SD 1.5 or 2.0). Also, many other modern methods are neglected in the paper ( Not a fair comparison.
4. In the ablation study, the corrector’s contribution is somehow questioned: a) What is the performance of SDXL+Corrector (w/o Re-sampler)? b) In all 6 metrics, it has gains only in 3 metrics, out of 6. The other 3 are either the same or even worse.
5. There is no limitation section.
6. In Table 3, it has the worst speed performance, and even the metrics are not great compared with ToMe.
7. Limited color diversity in qualitative results in Figure 3.
8. No qualitative Ablation study.
9. It drops the image quality in Figure 5.

### Questions
1. Isn’t \epsilon have a similar formulation as \tilda(\epsilon) in Lemma 1 204-205 (CFG, with Lambda)? Is it a typo?
2. What is the "x" in Lemma 1, a) 209? is it x_{t}?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem of multi-concept compositional generation in text-to-image diffusion models. The authors hypothesize such models struggle with compositional generation because the learned joint distribution for composite prompts overlaps significantly with individual concept distributions, leading to dominance by a single concept. To address this, they propose CO3, a plug-and-play corrective sampling method that suppresses individual concepts by re-weighting Tweedie means of each concept, rather than directly re-weighting in the noise (or score) space. Experiments on SD1.5, SDXL, and Pixart-$\Sigma$ show improved prompt alignment and concept coexistence.

### Strengths
1. CO3 requires no retraining or gradient computation and can be applied to any diffusion model. This makes it broadly applicable and practical.
2. The paper provides an intuitive theoretical explanation showing that directly performing weighted averaging of scores does not guarantee a valid Tweedie mean or consistent classifier-free guidance formulation; by instead re-weighting in the Tweedie-denoised space, CO3 ensures mathematical validity.
3. Novelty. While not the first to explore Tweedie-denoised space reweighting for concept composition [1], CO3 offers a clear theoretical formulation and a practical two-stage approach, early resampling followed by later correction, that together provide consistent empirical improvements and strong interpretability.

[1] Kwon & Ye, TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation, ICLR 2025.

### Weaknesses
My main concern lies in the evaluation, particularly regarding benchmark coverage and baseline selection.

1. The evaluated prompts are relatively limited, focusing mainly on compositions involving animals and objects. Incorporating commonly used benchmarks such as T2I-CompBench, which include more diverse attribute and object combinations, would strengthen the empirical validation.
2. A relevant baseline is missing [1]. In addition, comparisons with recent **LLM-guided generation methods** [2, 3, 4] would be beneficial to quantitatively assess the relative performance and effectiveness of CO3 in enhancing compositional fidelity. If the proposed method is orthogonal to LLM-guided approaches, it would be particularly valuable to demonstrate how CO3 complements them when combined.
3. Including comparisons with recent **state-of-the-art text-to-image models**, such as **Stable Diffusion 3.5** or **Flux**, would improve the paper’s relevance and help readers better understand the practical significance of the proposed method.

[1] Yu et al., Improving Compositional Generation with Diffusion Models Using Lift Scores, ICML 2025.

[2] Yang et al., Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms, ICML 2024.

[3] Lian et al., LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models, TMLR 2024.

[4] Hu et al., ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment, arXiv preprint.

### Questions
Please refer to weaknesses section.

### Soundness
2

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
5

### Summary
The paper addresses the problem of multi-concept prompt fidelity in text-to-image diffusion models. The authors propose CO3, a lightweight, re-training-free corrective sampling strategy that adjusts inference to maintain balanced representation among multiple concepts. By analyzing the instability of existing compositional guidance, the method identifies stable composition-weight regions and steers sampling toward them. CO3 integrates seamlessly with classifier-free guidance and demonstrates improved concept balance, coverage, and robustness across diverse prompts compared with prior compositional baselines.

### Strengths
1. Correction‑based compositional guidance with theoretical grounding and broad evidence.
- The method unifies ideas from correction‑based approaches and composable diffusion into a single re‑training‑free sampling framework, and provides a reasonable theoretical justification for composition‑weight choices and stability regions. The paper offers extensive quantitative and qualitative comparisons that demonstrate improvements in multi‑concept fidelity while remaining compatible with standard classifier‑free guidance.

### Weaknesses
1. Comparison on multi‑concepts is incomplete.
- The paper focuses on improving multi‑concept prompt fidelity, especially when one concept is rare or easily dominated by another. A direct theoretical and empirical comparison with R2F [1], ideally on RareBench, would clarify relative strengths on the rare‑concept regime where fidelity typically degrades most.

2. Model coverage could be broadened beyond SDXL.
- To strengthen the claim of generality, evaluations on additional recent open‑source text‑to‑image diffusion models, for example FLUX, would make the comparisons more persuasive and reduce the risk that gains are specific to one backbone.


Reference

[1] Park et al., Rare‑to‑Frequent (R2F): Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance.

### Questions
1. Compatibility with low‑step and high‑order solvers.
- Main experiments use SDXL with a 50‑step DDIM sampler. Can CO3 be applied effectively with high‑order solvers such as DPM++ at around 10 steps, and with distilled text‑to‑image models that operate at 4 steps or fewer? Any guidance on tuning composition weights in these low‑step regimes would be helpful.

2. Human evaluation.
- ImageReward and BLIP‑VQA are useful proxy metrics, but has a human evaluation been conducted to assess concept presence and balance under multi‑concept prompts? If not, a small user study could substantiate the claimed improvements in perceived fidelity.

### Soundness
3

### Presentation
3

### Contribution
2
