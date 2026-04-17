# Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Modeling human–human interactions from text remains challenging because it requires not only realistic individual dynamics but also precise, text-consistent spatiotemporal coupling between agents. Currently, progress is hindered by 1) limited two-person training data, inadequate to capture the diverse intricacies of two-person interactions; and 2) insufficiently fine-grained text-to-interaction modeling, where language conditioning collapses rich, structured prompts into a single sentence embedding. To address these limitations, we propose our Text2Interact framework, designed to generate realistic, text-aligned human–human interactions through a scalable high-fidelity interaction data synthesizer and an effective spatiotemporal coordination pipeline. First, we present InterCompose, a scalable synthesis-by-composition pipeline that aligns LLM-generated interaction descriptions with strong single-person motion priors. Given a prompt and a motion for an agent, InterCompose retrieves candidate single-person motions, trains a conditional reaction generator for another agent, and uses a neural motion evaluator to filter weak or misaligned samples—expanding interaction coverage without extra capture. Second, we propose InterActor, a text-to-interaction model with word-level conditioning that preserves token-level cues (initiation, response, contact ordering) and an adaptive interaction loss that emphasizes contextually relevant inter-person joint pairs, improving coupling and physical plausibility for fine-grained interaction modeling. Extensive experiments show consistent gains in motion diversity, fidelity, and generalization, including out-of-distribution scenarios and user studies. Code will be released at github.com/Qingxuan-Wu/Text2Interact.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces InterCompose and Text2Interact. The former synthesizes the motion of one person through text conditioning and single-person motion priors, generating human-human interaction-text paired data to alleviate data scarcity. The latter employs word-level attention to achieve fine-grained text-motion alignment and emphasizes the reproduction of interaction effects through weighted body parts, enhancing the realism of motion generation.

### Strengths
1. The paper pioneers the InterCompose framework, which is capable of synthesizing a large volume of dual-human interaction data. The methodology for filtering and selecting the data is also highly rigorous.
2. Text2Interact designs a sophisticated cross-attention mechanism that injects fine-grained textual information throughout the entire generation process, ensuring precise prompt control and correct temporal resolution.
3. The rendering results are highly impressive. Text2Interact indeed generates high-quality interactive motions, and we look forward to the authors open-sourcing their work for the benefit of the community.

### Weaknesses
1. Section 3.1.1 of the paper is not clearly articulated and is kind of difficult to follow. **It would be better to clarify the composition method of $x_{1}$ and $x_{2}$, as well as their role in the subsequent training of $D_{θ}$.**
2. Section 3.1.2 describes the data filtering process. **It is recommended to include more images to reduce the reading difficulty for the audience.**

### Questions
1. There appears to be ambiguity in Fig. 2(a) regarding InterCompose. The diagram suggests that the motion of person A is first generated using A's prompt, and then the motion of person B is generated using the reaction prompt and A's motion. However, in Section 3.1.1, line 210, it is stated that A and B are generated separately. Does the paper intend to explain that during the data preparation stage, MoMask is used to generate the motion pair ($x_{1}$, $x_{2}$) as the ground truth? Then, during the training phase, a diffusion model $D_{θ}$ is trained, with inputs including the text prompt and the motion $x_{1}$ of one person, and the training target is $x_{2}$? It is recommended to distinguish between the data synthesis and training processes in Fig. 2(a) for better clarity.

2. The data filtering process described in Section 3.1.2 also lacks intuitive visual explanations. Adding relevant illustrations in the Appendix would significantly reduce the reading difficulty for the audience.

### Soundness
4

### Presentation
2

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
In this paper, it introduced a scalable synthesis-and-filtering strategy to generate high-quality interaction from LLM and single-person motion prior. Later, it designed a Text2Interact module to generate two-person interaction motion. The proposed method was validated on InterHuman dataset and presented better performance than many previous works.

### Strengths
1. It shows better performance than many previous works.
2. The visual performance is better than the baseline.

### Weaknesses
1. The method is only validated on one dataset, the generalization ability of the proposed method is not validated.
2. Generating the second human motion given the motion of the first person has been utilized in previous multi-person motion generation. The difference should be further discussed.
3. In the manuscript, the relationship between InterCompose and Text2Interaction should be detailed.
4. More recent works should be considered for comparison.

### Questions
1. Whether the proposed method design a data synthesis method to create high-quality interaction data.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets limitations in two-person interaction generation: (1) limited real data and weak spatiotemporal coordination, and (2) insufficient semantic grounding from sentence-level conditioning. The authors propose:

- **InterCompose**, a scalable synthesis pipeline that generates and filters synthetic text-motion pairs via two-stage quality and diversity filtering;
- **Text2Interact**, which employs word-level cross-attention and an adaptive interaction loss emphasizing semantically important cross-human joint pairs.

### Strengths
1. **Originality:** The two-stage data synthesis and filtering pipeline is novel and intuitive. The word-level attention and adaptive interaction loss are well-motivated and effective.
2. **Quality:** Achieves SOTA on InterHuman with solid ablations. The user study adds credibility to perceptual quality.
3. **Clarity:** Writing and figures are clear, and each module is conceptually coherent.
4. **Significance:** Addresses two key challenges—data scarcity and semantic granularity—in a well-balanced framework.

### Weaknesses
1. **No validation of unfiltered synthetic data.**
    The paper claims that filtering improves quality but does not report the result of fine-tuning with unfiltered synthetic data. Without this, the necessity of the two-stage filtering pipeline remains unproven.
1. **No scaling analysis of synthetic data.**
    Although the synthesis pipeline is described as “scalable,” the paper does not analyze how model performance changes with varying data volume or filtering ratios.
1. **δ = 0.58 lacks justification.**
    The threshold is only said to be “empirically chosen” in Appendix B.5, with no validation or sensitivity analysis.
4. **Minor observation:** In Table 3, the “w.o. FT” row yields slightly higher R-Precision than “Ours.” Some explanation would clarify this trade-off between text alignment and motion realism.

### Questions
1. How does the model perform when fine-tuned on unfiltered synthetic data?
2. Can the authors show a performance curve (e.g., FID vs data size) to demonstrate the scalability claim?
3. Could you explain why R-Precision slightly drops after fine-tuning?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work examines the problem of learning two-person human interaction generative models conditioned on text descriptions. The work has two main parts: creating a dataset and learning a model. The dataset aspect involves creation of a large-scale synthetic dataset of two-person human interaction by combining LLM descriptions to prompt existing one-person motion generation algorithms and an interaction model conditioned on the first motion. The interaction model is trained on motion-capture data from interhuman. The synthetic data is then used to train a two-person interaction model. The two-person interaction model uses word-level text embeddings and interleaved attention between self and partner sequences and text descriptions. Experiments show the model can provide more realistic two-person interactions that previous motion generation models for a broader range of text descriptions.

### Strengths
* The first half of the approach is an interesting way to create a broader range synthetic human motions than what is available in interaction motion capture data. LLM prompting and SOTA single person human motions can cover a fairly wide range of human actions. By learning a conditional interaction model and synthesizing reactions to the single-person generations, a wider variety of interaction scenarios can be synthesized with reasonable accuracy because the single-person motion provides a strong starting point for the reaction model.
* The quantitative and qualitative results provide reasonable evidence that the proposed method can lead to more effective modeling of two-person interactions.

### Weaknesses
* The main weakness of this work is that learning the generative model in the second stage of the paper does not seem to provide anything beyond the first stage. In the first stage, the work essentially defines a way to sample from the distribution of two-person interactions by making use of existing models/datasets (and learning a new conditional model from Interhuman). I am not sure what the benefit of distilling this approach into a second model is (other shifting from a conditional to joint model, which for the purposes of generation is not a major advantage in my opinion). Furthermore, shouldn't we expect the second stage model to be further from the distribution of natural motions than the synthetic data it is trained on (due to imperfect learning)? And since the training data is already synthetic, what is the purpose of learning a second synthetic distribution and not just focusing on the data generation process as a model on its own?
* The work still relies on InterHuman as a way to learn human interactions. While the novelty provided by the single person motion can increase variety of interaction situations, at the end of the day the proposed method does not have a way to model interactions that are not quite close to the interaction distribution from InterHuman.

### Questions
What is the purpose and advantage of the second stage model? Why not just focus on the data generation process as an interaction model in its own right? What are the metrics and user preference for the second stage model vs the data it was trained on?

### Soundness
3

### Presentation
2

### Contribution
2
