# GIFT: Gradient-aware Immunization of diffusion models against malicious Fine-Tuning with safe concepts retention

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
We present GIFT: a Gradient-aware Immunization technique to defend diffusion models against malicious Fine-Tuning while preserving their ability to generate safe content. Existing safety methods, such as safety checkers, are easily bypassed, and concept erasure methods fail under adversarial fine-tuning. GIFT addresses this by framing immunization as a bi-level optimization problem: the upper-level objective degrades the model’s ability to represent malicious concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. GIFT achieves robust resistance to malicious fine-tuning while maintaining safe utility. Experimental results show that GIFT significantly impairs the model’s ability to re-learn malicious concepts while maintaining performance on safe content, offering a promising direction for creating inherently safer generative models resistant to adversarial fine-tuning attacks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GIFT, an immunization framework designed to make open-source diffusion models resistant to malicious fine-tuning (e.g., recovering erased or unsafe concepts) while preserving their ability to generate safe content. The method formulates the immunization process as a bi-level optimization problem, which both maintains safe generation and reduces the model’s adaptability to malicious concepts, even under adversarial fine-tuning attacks. Experiments on objects, art styles, and NSFW concepts demonstrate that GIFT effectively mitigates malicious fine-tuning while retaining strong generative capabilities for safe content.

### Strengths
1. Defending open-source diffusion models from adversarial fine-tuning is an important and timely problem.
2. Evaluation across multiple domains with comparisons against IMMA and ESD, demonstrating a consistent balance between safety and usability.
3. The approach offers the flexibility to suppress arbitrary concepts.

### Weaknesses
1. The bi-level optimization design, while well-motivated, seems mostly heuristic without a clear theoretical or empirical justification for why it outperforms a simple joint loss.
2. The method assumes access to separate malicious and safe datasets, but these are difficult to define or collect in practice. For instance, defining a “safe” concept as “any other concept” makes the safe set effectively infinite and impractical to obtain.
3. Limited practicality: The immunized model cannot generate desired outputs directly from prompts; users must fine-tune with reference images, which is neither common nor convenient. Moreover, if I understand correctly, the reference image’s concept must lie within the safe dataset distribution used during immunization, further reducing the model’s utility.

### Questions
Please refer to the weakness and the following:
1. Why does Maximization + Rep. Noise outperform the full GIFT (Maximization + Rep. Noise + Prior) on malicious concepts? The prior term should preserve safe content without weakening immunization. Also, why does Figure 7 show significantly weaker suppression of malicious concepts than Figure 1?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose a gradient-aware immunization technique (GIFT) to defend diffusion models against malicious fine-tuning while preserving their ability to generate safe content. According to this approach, the immunization is formulated as a bi-level optimization process: the upper-level objective degrades the model’s ability to represent malicious concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. Extensive experiments on diverse concepts demonstrate that GIFT shows resistance against the re-learning of malicious content while successfully maintaining generation quality and preserving fine-tuning capabilities on safe data.

### Strengths
Here are the paper's strengths:
- It is well-written and well-documented
- The review of the state-of-the-art covers most of the relevant papers in the field
- The proposed approach is scientifically sound
- Provides comprehensive experiments across multiple domains (objects, art styles, NSFW content) with both qualitative and quantitative evaluation
- Experimental validation is complemented with ablation studies, prompt robustness tests, and analysis of hyperparameter effects (e.g., $\beta$)

### Weaknesses
Here are the paper's weaknesses:
- The proposed approach relies heavily on existing approach IMMA (Zheng & Yeh, 2024). 
- Some aspects of the paper should be more clearly explained in order to improve readability, especially in section 3: the formalism is not sufficiently detailed
- Presentation is dense and sometimes awkwardly formulated (“of said parameters”  - lines 150-151, “LLM immunization technique” - line 200)
- The connection between theoretical motivation (MI reduction) and its practical implementation (representation noising loss) should be better explained
- The experimental validation is limited
- Some complexity analysis of the approach is missing, e.g. how the inner and outer loop steps affect training time or memory usage compared to fine-tuning

### Questions
Besides the weaknesses mentioned above, here is the rest of my concerns:
- The papers seem to rely heavily on IMMA (Zhang & Yeh, 2024). Therefore, the authors must critically discuss how their approach relates to IMMA's and clearly articulate their scientific contributions (apparently, IMMA also formulates immunization as a bi-level process).
- The formalism in section 3 should be significantly improved in order to increase readability. Sometimes the equations are just stamped in the paper without any justification. 
- Some parameters are not explained. Who is $\psi'$ in eq. 2?
- How sensitive is your approach to the choice of $\alpha_I$ and $\alpha_P$ parameters. No discussion is provided in the paper regarding the choice of these parameters.
- Clarify how the alternating updates between safe and malicious datasets are scheduled during training?
- Why do you compare your approach only against IMMA and ESD? Please extend your comparison with other methods

### Soundness
3

### Presentation
2

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
This paper introduces GIFT, a defense framework designed to protect text-to-image diffusion models from being maliciously fine-tuned to generate harmful or unauthorized content, while preserving their ability to generate safe content. GIFT formulates immunization as a bi-level optimization problem inspired by IMMA. The upper-level task maximizes a loss on malicious concepts combined with a “representation noising” term, while the lower-level task minimizes the prior-preservation loss on safe data.

### Strengths
1. The paper explores a valuable problem of making diffusion models resistant to malicious fine-tuning.
2. The authors structure the paper clearly.

### Weaknesses
The quality of the paper still needs improvement.
1. The experiments were only conducted on SD 1.5, lacking the latest diffusion models such as SD3, SD3.5, Flux, etc.
2. Regarding Figure 2, it appears that the CLIP Score of the SD model becomes lower than both GIFT and IMMA after fine-tuning for 1000 steps. Does this indicate that the evaluation prompts overlap too much with the training set, resulting in an effect where normal generation capability is maintained only on the evaluation prompts?
3. The evaluation metrics of the article include CLIP, etc., but lack safety-related evaluation metrics such as NudeNet and Q16.
4. Regarding robustness testing against malicious fine-tuning, there is a lack of security robustness tests, such as UnlearnDiff.
5. The dataset size is relatively small and lacks generalizability.
6. When evaluating normal images, relying solely on CLIP cannot fully reflect the generative capability for normal images. It is necessary to assess from various aspects such as aesthetics and image quality, rather than just the CLIP score.
7. There are too few diffusion model safe methods compared, with only ESD and IMMA being compared.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GIFT, a post-training method for immunizing diffusion models against downstream fine-tuning on malicious concepts. GIFT alternates between two objectives: (1) a standard diffusion loss on safe concept images to preserve generation quality on safe concepts, and (2) a maximized diffusion loss on malicious concept images for immunization. Additionally, to prevent easy re-learning of malicious concepts, GIFT optimizes intermediate layer activations on malicious images to resemble noise sampled from the same distribution, encouraging the model to erase these concepts across all layers rather than only at the output, which could be easily re-adapted. GIFT is evaluated on immunization against specific objects, artistic styles, and NSFW content.

### Strengths
-  GIFT operates in post-training and can be applied to pre-trained models, thus not harming the pre-training process.

- The paper is clearly written and easy to follow.

- Enforcing the model to erase malicious concepts from all layers is interesting and the proposed approach is intuitive.

### Weaknesses
- **Immunization effectiveness.** Both figures 4 and 6 show that GIFT can still generate the immunized concepts in many of its outputs. This raises a concern about a simple attack: sampling multiple generations until the malicious concept appears. To address this, could the authors provide a larger set of uncurated generations (e.g., 25 samples) from a few randomly selected immunized concepts to better characterize the actual failure rate?

- **Limited concept coverage.** The paper demonstrates immunization on a small number of hand-selected concepts. To assess whether the method generalizes broadly, could the authors evaluate GIFT on a larger set of randomly sampled concepts with a few generations from each? For each concept, please show both attempted generations of the immunized malicious content and preserved safe content (as in Fig. 6).


- **Longer adaptation periods.** Does extended fine-tuning time enable re-learning of the immunized concepts? Figure 4 shows a decreasing CLIP score trend toward the end of training, which could support this hypothesis, though it may also indicate overfitting to the provided images. It would be valuable to evaluate: (1) CLIP scores over longer adaptation times, and (2) the LPIPS between generated images and their closest training image over time (i.e., compute LPIPS to each input image and report the minimum), to distinguish between overfitting and genuine concept recovery.
 
- **SD Performance.** Figure 8 shows that safe concept generation is noticeably degraded, with baseline SD producing more detailed outputs than SD+GIFT. This raises concerns about the practical utility of the method for preserving model performance on safe content.

Minor remarks:

- **Selected NSFW concept.** Figure 6 uses explicit pornographic scenes as the malicious concept. While I understand the need to demonstrate NSFW content erasure, other NSFW concepts (e.g., smoking) might demonstrate the same capability while being more appropriate for a broader reading audience.

- **Quantitative results.** Can the authors report average CLIP scores and minimum LPIPS to the closest input image at the end of fine-tunin, averaged over a large set of adaptations? This would provide a more comprehensive quantitative assessment of immunization effectiveness beyond the trajectory plots shown in the paper.

### Questions
- Can GIFT handle immunization to multiple different concepts at once? E.g., immunization to all NSFW concepts? Does performance on safe concepts  further degrades when GIFT is applied with multiple immunization concepts?

### Soundness
2

### Presentation
4

### Contribution
2
