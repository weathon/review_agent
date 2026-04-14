# FairGen: controlling fair generations in diffusion models via adaptive latent guidance

- Decision: Reject
- Scores: 3, 3, 5, 5

## Abstract
Diffusion models have shown remarkable proficiency in generating photorealistic images, but their outputs often exhibit biases toward specific social groups, raising ethical concerns and limiting their wider adoption. This paper tackles the challenge of mitigating generative bias in diffusion models while maintaining image quality. We propose FairGen, an adaptive latent guidance mechanism enhanced by an auxiliary memory module, which operates during inference to control the generation distribution at a desired level. The latent guidance module dynamically adjusts the direction in the latent space to influence specific attributes, while the memory module tracks prior generation statistics and steers the scalar direction to align with the target distribution. To evaluate FairGen comprehensively, we introduce a bias evaluation benchmark tailored for diffusion models, spanning diverse domains such as employment, education, finance, and healthcare, along with complex user-generated prompts. Extensive empirical evaluations demonstrate that FairGen outperforms existing bias mitigation approaches, achieving substantial bias reduction while preserving generation quality. Furthermore, FairGen offers precise and flexible control over various target distributions, enabling nuanced adjustments to the generative process.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes FairGen, addressing the challenge of mitigating generative bias in diffusion models while preserving image quality. It also introduces a bias evaluation benchmark, HBE, covering diverse domains such as employment, education, finance, and healthcare. Empirical results show that FairGen outperforms existing bias mitigation methods, achieving bias reduction while maintaining generation quality.

### Strengths
1. FairGen strikes a balance between minimizing bias and preserving image quality.
2. This paper proposes the HBE dataset, which is both comprehensive and well-structured, spanning diverse domains, including underexplored areas.

### Weaknesses
1. The writing needs improvement. Paragraph three might be better placed in the Related Work section. The Preliminaries section spends too much space (around 70% of a page) describing the diffusion model, which is not the core content of the paper. Additionally:
- Lines 135 and 223 should use \citep instead of \citet.
- In line 213, prompt_c should be written as prompt c.
- Line 221 contains "mistralai," which should be removed.
- In line 222, a brief description of the HBE dataset should be provided instead of referring directly to Section 4.
- On line 241, the {+1, -1} description is repeated unnecessarily.
2. In Figure 1, the encoder and decoder are not properly described. This is potentially confusing, as the encoder should refer to the text encoder, and the decoder should correspond to the VAE decoder.

### Questions
1. Clarification on orthogonality (line 218):
- What exactly does "orthogonality" mean in this context, and why is it used? The sentence feels confusing and could benefit from rephrasing for clarity.
2. HBE dataset split proportions:
- Why does the HBE dataset allocate 40% for training, 10% for validation, and 50% for testing? This allocation seems unusual, as the training set is relatively small while the test set is quite large.
3. Source of the target distribution in Fig. 1:
- What is the source of the target distribution mentioned in Fig. 1? If the correct ratio of attributes (e.g., educated vs. uneducated) is unknown, it seems problematic to assume a 50:50 split. In reality, such a distribution may differ, and it can be difficult to determine the true ratio.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes FairGen, a method for controlling the distribution of sensitive attributes, such as gender, in text-to-image generation. FairGen introduces a latent guidance module that calculates the difference between scores conditioned on pairs of prompts differing only in the targeted attributes. A language model is fine-tuned to automatically generate these prompt pairs. Additionally, an indicator guidance module is used to control the guidance direction based on historical statistical properties. A benchmark is also introduced to facilitate fair generation evaluation.

### Strengths
The paper tackles fairness in text-to-image generation, a timely and important issue with significant ethical implications. The proposed method is straightforward and shows promising effectiveness. The motivation is clear and well-articulated, and the paper is well-organized and clearly written.

### Weaknesses
Some key details are missing, particularly regarding the design of the indicator guidance model $I(c,M,(a\_1,a\_2))$. This component is the core of the proposed method, but it is not adequately described in the main text. Please see some of my question in the Question section.

Another major concern is the inference burden introduced by the method. Since the latent guidance is prompt-dependent, it must be computed at inference time, resulting in three evaluations of the score model per timestep and, therefore, a 300% increase in inference time. Additionally, there are extra inferences required for the language model. For multi-dimensional attributes (as in Eq. (7)), this increase is even more substantial. This high computational cost is problematic, especially when compared to baselines like Shen et al. (2023), which does not increase inference time. Consequently, the comparison may be unfair. To convincingly establish the method's effectiveness, a comparison under equal inference times (e.g., adjusting sampling steps across methods) is needed.

---

Minor:

- L85: Extra space before the period.
- Misuse of \citet and \citep, e.g., L121 and L135.

### Questions
- What is the detailed formulation of the indicator guidance model $I(c,M,(a\_1,a\_2))$? Specifically, what statistical properties are used within this model? As clusters are updated over time, how are these statistical properties assigned when KNN is used for re-clustering? Does this imply that all individual historical prompts and their generated properties need to be stored?
- In Eq. (7), each pair of attributes forms a latent guidance. For categorical attributes (e.g., different races), does this mean that a squared number of latent guidances will be incorporated during generation?
- According to the notation $I(c,M,(a\_1,a\_2))\\in\\{1,-1\\}$, it seems that latent guidance is always active. What happens if the prompt already specifies the attribute? For example, with a prompt like "A male computer programmer," will the method still cluster this prompt's features $E(c)$ and include this generation in the statistical counts? If not, how is it determined whether the method applies for a given prompt?
- The text suggests that for each cluster, historical property counts are maintained. Are these counts global statistics, or are they calculated separately for each user? This matters because using global statistics without immediate updates could lead to biased generation in a small window for individual users.
- How are the features of past generations obtained (e.g., determining if each generation is male or female)? Are external classifiers used for this purpose?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces FairGen, a novel adaptive latent guidance mechanism designed to mitigate biases in text-to-image diffusion models while preserving image quality. FairGen operates during inference, using an adaptive latent guidance module combined with an auxiliary memory module to control the generated output's attribute distribution, allowing for dynamic bias mitigation. The method demonstrates flexibility and efficacy by enabling controlled generation aligned with specific target distributions, making it superior to existing prompt intervention and fine-tuning approaches. To evaluate FairGen, the paper also introduces a comprehensive benchmark, the HBE dataset, which covers diverse domains and complex prompt structures, providing a robust framework to assess generative fairness. Experimental results indicate that FairGen achieves substantial bias reduction while maintaining high image quality, showing promise as an effective solution for fairness-aware generation in diffusion models.

### Strengths
1. FairGen’s innovative adaptive latent guidance mechanism allows for dynamic and precise bias control in diffusion models, significantly improving upon existing static or fine-tuning methods.

2. The introduction of the HBE benchmark provides a comprehensive and realistic framework for assessing bias across multiple domains, making it a valuable contribution to future research in generative fairness.

3. Experimental results show that FairGen achieves substantial bias reduction while preserving high image quality, demonstrating its effectiveness in balancing fairness and visual fidelity in text-to-image generation.

### Weaknesses
1. The paper does not provide a comprehensive comparison of FairGen with methods beyond the diffusion model domain, limiting insights into its broader applicability in generative fairness control.
2. It lacks detailed error analysis, particularly regarding failure cases or scenarios where FairGen might amplify bias unintentionally.
3. This paper lacks in-depth theoretical analysis.
4. There is minimal discussion on the computational costs associated with integrating the adaptive latent guidance mechanism into large-scale diffusion models.
5. Some intermedium visualization results could benefit the empirical studies.

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper works on reducing bias regarding sensitive attributes such as gender, race, etc., from pretrained Stable Diffusion models. The objective of such a line of work requires reducing bias while preserving the generation alignment with the user prompt. The paper proposes an inference-time guidance strategy (FairGen) that is composed of two components: the adaptive latent guidance module and the indicator guidance module to use the previous generations as the memory to guide the inference process toward producing unbiased images. Specifically, the indicator guidance module is constructed based on the memory of previous generations and the sensitive attributes whose biases are to be removed. The adaptive latent guidance module then takes such output as input, along with other inputs, such as input prompts, to provide guidance for the inference process. For evaluation, the paper proposes a more comprehensive benchmark and showcases superior performance than baselines.

### Strengths
- The task that this paper tackles is well-motivated and has attracted the attention of many recent works.
- The proposed inference-time guidance method can be efficient as it does not require modifying the pre-trained Stable Diffusion’s weights and has also achieved effectiveness bias mitigation and quality-preservation results.
- This paper also introduces a comprehensive bias evaluation benchmark HBE that comprises of a wider array of domains, prompt structures, and sensitive attributes than previous benchmarks.

### Weaknesses
- **Presentation**. Figure 1 needs to be refined, which is currently misleading as it misses one of the three inputs for the adaptive latent guidance module. From the figure, the module only takes the target attribute and input prompt as inputs, however, it also takes the previous step’s denoised latent result.
- **Effectiveness of DPO**. From the ablation study, the contribution of the DPO component seems minimal when comparing rows 2 and 3 of Table 6.
- **Concern regarding practical significance**. The memory of previous generations that the proposed method is founded on can be unavailable in some practical scenarios where the user only wishes to generate a single or few images. In such a case, since the proposed method is an inference-time strategy that relies on memory, the generations will still be biased.
- **Novelty**. The inference-time guidance strategy for removing biased and unsafe content is already proposed in [1], whose method is similar to the adaptive latent guidance module proposed in this work. Although this paper also uses an indicator guidance module that leverages previous generations as the memory to serve as an indicator for the adaptive latent guidance, such a component is not quite novel as a method.

[1] Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models

### Questions
I would really appreciate it if the authors could address the points in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
