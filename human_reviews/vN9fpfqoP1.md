# Fine-Tuned Language Models Generate Stable Inorganic Materials as Text

- Decision: Accept (poster)
- Scores: 6, 5, 6

## Abstract
We propose fine-tuning large language models for generation of stable materials. While unorthodox, fine-tuning large language models on text-encoded atomistic data is simple to implement yet reliable, with around 90\% of sampled structures obeying physical constraints on atom positions and charges. Using energy above hull calculations from both learned ML potentials and gold-standard DFT calculations, we show that our strongest model (fine-tuned  LLaMA-2 70B) can generate materials predicted to be metastable at about twice the rate (49\% vs 28\%) of CDVAE, a competing diffusion model. Because of text prompting's inherent flexibility, our models can simultaneously be used for unconditional generation of stable material, infilling of partial structures and text-conditional generation. Finally, we show that language models' ability to capture key symmetries of crystal structures improves with model scale, suggesting that the biases of pretrained LLMs are surprisingly well-suited for atomistic data.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes an approach to materials discovery by fine-tuning large language models (LLMs) on text-encoded atomistic data. The authors claim that this method can generate materials predicted to be metastable at a higher rate than competing diffusion models.

### Strengths
* The paper is well-motivated, addressing the limitations of existing computational materials databases and the potential of generative models for materials discovery.
* The proposed approach of fine-tuning large language models on text-encoded atomistic data is novel and unorthodox, offering a new perspective on materials generation.
* The paper is well-written, providing clear background information and a thorough explanation of the proposed method.

### Weaknesses
Related concerns are discussed in the questions section.

### Questions
* Can the authors provide a more detailed comparison with existing generative models in materials discovery, discussing the advantages and limitations of the proposed LLM-based approach compared to other state-of-the-art methods?
* Are there any potential drawbacks or limitations of using LLMs for materials discovery, such as computational complexity or interpretability of the generated structures?
* In the paper, the authors mention that "We chose not to augment the ordering of atoms because these variables often contained valuable information". How to ensure that the atomic order will not change the results?
* In the paper, the authors mention that "we only run VASP calculations on materials that have already been predicted as metastable by M3GNet", while in the caption of figure 3, "we only run VASP on structures predicted to be stable by M3GNet." What are the differences between the "stable" and "metastable"?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper propose a new application area for large language models (LLMs), i.e. LLaMA-2. The author leverages parameter-efficient fine-tuning to use LLMs. Based on domain expertise in material sciences, the author evaluated the proper tokenization methods for crystal structures and developed new metric to further include symmetric information into fine-tuning process.

### Strengths
1. the author got good performance in the shown benchmarks.
2. the method seems solid since it has been widely used in many other domains.

### Weaknesses
1. While one anticipates good performance from LLMs on standard evaluation metrics, especially with the likes of LLaMA-70B, the critical matter lies in the practical application in experiments.

2. Given that LLMs can sometimes produce hallucinations, it would be beneficial to comprehensively evaluate this behavior in large models, rather than merely touching upon it in the limitations section. Presenting failure examples could offer valuable insights.

3. Regarding Figure 2, could the authors elucidate why larger models result in poorer coverage?

### Questions
See above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed to use LLMs to model the materials. The main technical contribution comes from 1) a modified tokenization mechanism that is suitable for material data; 2) a data augmentation method that leverages the property of the materials; 3) the prompt designs. Experiments on benchmark datasets show that when tuned on LLAMA2, it is able to improve the performance compared to other baselines.

### Strengths
- Application on the material science domain is interesting
- The adaptation of LLMs for the material science is reasonable
- Empirical results are promising

### Weaknesses
- The technical contribution is relatively limited, where the tokenization, the prompt design and the objectives for pertaining are all well studied in the past.
- Since the nature of the material data is quite different compared to the natural language data, I’m wondering whether the pretrained LLAMA2 is offering any additional value. It would be helpful if one can show the performance with and without loading the pretrained checkpoint from LLAMA2 tasks.

### Questions
I’d like to see the author’s response on the questions I listed in the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
