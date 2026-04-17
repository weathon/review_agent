# POPS: Recovering Unlearned Multi-Modality Knowledge in MLLMs with Fine-tuning and Prompt-based Attacks

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on cross-modal tasks by jointly training on large-scale textual and visual data, where privacy-sensitive examples could be intentionally or unintentionally encoded, raising concerns about privacy or copyright violation.  To this end, Multi-modality Machine Unlearning (MMU) was proposed as a mitigation that can effectively force MLLMs to forget private information. Yet, the robustness of such unlearning is not fully exploited when the model is published and accessible to malicious users. In this paper, we propose a novel adversarial strategy, namely Prompt-Optimized Parameter Shaking (POPS), aiming to retrieve the unlearned multi-modality knowledge via fine-tuning. Our method steers victim MLLMs to generate potential private examples by prompt optimization and then uses the synthesized examples to fine-tune the MLLMs to generate private information. Our experiments on the different MMU benchmarks reveal substantial weaknesses in the existing MMU algorithms. Our attacks achieve near-complete recovery of supposedly erased sensitive information, exposing fundamental vulnerabilities that challenge the very foundations of current multimodal privacy protection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an attack method for recovering supposedly unlearned knowledge in multimodal language models. The method works by first finding an adversarial prompt suffix that elicits structurally similar knowledge from the model, according to a structurally similar OOD dataset. This prompt is used to create training data to finetune the model on, basically as anti-refusal training. Once the model is trained on that data, they transfer it to the original unlearning domain. Experiments find that this approach recovers close to the level of knowledge as finetuning the model directly on the original unlearning domain. Experiments are conducted across multiple models and benchmarks in the area. Performance of the proposed methods is baselined against the Shake-to-Leak, which finetunes a model on synthetic completions in attempt to approximate anti-refusal training. Ablations show the role of several design choices, including a perplexity-based prior on the adversarial suffix.

### Strengths
- Important: The paper shows directional improvements in attacks on unlearned models. I agree that this method helps recover some “unlearned” knowledge in the models.
- Important: A wide variety of models and datasets are used to test the main hypotheses. The paper also explores multiple metrics to help demonstrate the nature of knowledge recovery.
- Of some importance: The focus on multimodal models helps reveal continued shortcomings of unlearning methods in these settings.

### Weaknesses
- Very important: One of my main concerns is about the effect sizes. The “ceiling” for knowledge recovery is very low, around 43 accuracy points. The “unlearned” model maintains about 40% accuracy. This is an extremely narrow window to work within. I don’t understand how the ceiling could be so low and the unlearned baseline could be so high. It makes me think that the choice of model and benchmark together were not appropriate. It would be better to work in a setting where more of the knowledge tested by the benchmark were known by the model to begin with, and after unlearning, that knowledge was not present at all (model closer to random chance).
- Important: I am not sure of the novelty of the method. The approach uses an OOD dataset for optimization. Was the Shake to Leak baseline tested on that dataset, or a separate, weaker synthetic dataset? I think the method should be compared against directed finetuning on that OOD dataset.
- Important: Sections of the paper like the “Remark” read as highly speculative. These are not proven interpretations of the experimental data. Rather, they are guesses at why attacks on unlearned multimodal knowledge are effective. It does not benefit the paper to include claims like these.
- Important: No uncertainty quantification for main results. Since the margins are small, it would help to have confidence intervals and p values.
- Of some importance: Is the POPS method really blackbox? It seems like it requires backpropagation through the model in order to do the suffix optimization. Is that right?

### Questions
Please feel free to respond to the questions associated with the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines the robustness of machine unlearning in Multimodal Large Language Models . The authors propose POPS, an attack method that recovers knowledge supposedly forgotten by the model. POPS first uses gradient-based optimization to craft a universal prompt suffix that triggers information leakage, then fine-tunes the model on the leaked synthetic data. By integrating the Shake-to-Leak strategy to perturb model parameters, POPS amplifies the recovery of forgotten knowledge. Experiments show that POPS restores a substantial portion of the forgotten information while maintaining overall model performance, challenging the effectiveness of current multimodal unlearning approaches.

### Strengths
1. The POPS method extends the Shake-to-Leak approach to the multi-modal domain. By integrating prompt optimization with parameter shaking, it effectively exploits the residual associations within the vision-language alignment layers of MLLMs.
2. The synthetic data amplification step and the S2L fine-tuning step in POPS form a self-enhancing loop. This closed-loop system, which uses generated samples to drive LoRA fine-tuning, is shown in ablation studies to contribute a significant performance gain.
3. The paper is well-structured and generally clear, with a comprehensive discussion of the theoretical underpinnings.

### Weaknesses
1. In the Problem Statement section, the authors claim to focus on the black-box setting. However, both Equation (1) and Algorithm 1 require access to the logits, and Algorithm 1 further needs to compute gradients. Where exactly is the black-box assumption reflected? If the method is truly designed for the black-box scenario, can it successfully attack commercial APIs such as GPT or Gemini, where neither logits nor gradients are available?
2. This paper's novelty feels to me like a minor tweak of the GCG approach ported to a new task; I find the contribution rather incremental.
3. The paper should supply concrete examples of P_target, P_suffix, and y_gt to make the approach easier to understand.

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes POPS, an adversarial pipeline that aims to recover sensitive knowledge from multimodal large language models (MLLMs) that have undergone machine unlearning. The method integrates a PromptSuffix optimizer, which learns universal suffix embeddings via gradient-based optimization, with a Shake-to-Leak–inspired fine-tuning phase to amplify memorized knowledge and restore forgotten content. Experiments on three benchmarks demonstrate the superior inherent trade-off between knowledge removal and multimodal reasoning across various architectures.

### Strengths
- This paper addresses the robustness of multimodal unlearning, which is a critical and emerging issue as MLLMs become increasingly deployed under privacy regulations.

- The paper benchmarks several popular unlearning algorithms across multiple multimodal datasets and architectures.

- The paper is well-documented and easy to read.

### Weaknesses
- The proposed method, combining prompt optimization with fine-tuning attacks, is a direct extension of Shake-to-Leak (Li et al., 2024b) and prior prompt-tuning attacks (Carlini et al., 2021; Bhaila et al., 2024). There is no clear theoretical or methodological novelty beyond combining these known techniques.

- Although it is acknowledged that the method extends prior work to the new multi-modality settings, the paper does not sufficiently clarify the motivation for this extension nor the challenges that arise in doing so.

- While the authors claim this is the first fine-tuning-based multimodal attack, the literature [1] already discussed and evaluated fine-tuning attack under this setting. This further implies that the extension of fine-tuning-based attacks to multimodal settings may not pose significant challenges, which consequently weakens the technical contribution of the paper.

[1] Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack‑Defense Evaluation

### Questions
- How does POPS differ algorithmically from prior approaches (e.g., Shake-to-Leak) beyond applying it to multimodal data? Could the same results be achieved by directly adapting S2L with multimodal fine-tuning without the “PromptSuffix” step?

- What concrete multimodal-specific challenges motivated this extension, and how does POPS differ from existing unimodal unlearning attacks in addressing them?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a new attack against multimodal LLMs that have undergone unlearning using existing multimodal unlearning methods. The attack, POPS, exploits prompt suffix optimization, the cross-modal nature of MLLMs, and synthetic data generation with fine-tuning to recover previously unlearned information. Experiments on MLLMU-Bench demonstrate that POPS effectively recovers sensitive information across models trained with multiple unlearning techniques, approaching the upper bound achieved by full fine-tuning. Overall, the paper highlights the privacy–utility trade-offs inherent in multimodal unlearning and underscores the need for robust, multimodal-aware defenses capable of resisting adversarial prompt and fine-tuning attacks.

### Strengths
The paper presents a novel adversarial attack framework (POPS) specifically targeting multimodal large language models (MLLMs) that have undergone unlearning. While prior work has explored unimodal unlearning and attacks like Shake-to-Leak (S2L), this paper introduces several creative extensions to multimodal contexts, including:
- Out-of-distribution (OOD) prompt suffix optimization to maximize latent concept recovery.
- Exploitation of cross-modal associations that persist after unlearning.
- Synthetic data amplification tailored to multimodal representations.
Experiments are span multiple benchmarks (MLLMU-Bench, CLEAR, UnLoK-VQA), multiple models, and three unlearning strategies.
By exposing these vulnerabilities, the paper sets the stage for future research on multimodal-aware unlearning defenses,

### Weaknesses
- While the paper clearly demonstrates vulnerabilities in multimodal unlearning, it does not evaluate existing or potentially straightforward defense mechanisms against the proposed attacks.
    - This include Head Projection defense [1] and a defense where you unlearn for paraphrases of the V, Q, A tuplet so that the model becomes robust to input variations of the unlearned points

The ablation in the paper shows that prompt optimization is a significant component of the pipeline. The paper does not compare against other popular prompt optimization attacks like GCG [2].

The paper has limited novelty. 
- There are several other papers including [1, 3] which show that unlearning is reversible with finetuning attacks.
- Prompt optimization attacks have been explored in serveral domains including jailbreaks
- Other papers have also explored multimodal attacks [4]. 

The paper has significant grammar errors: "multimodality representations" in place of "multimodal representations"

[1] Patil, Vaidehi, et al. "Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation." Transactions on Machine Learning Research.

[2] Zou, Andy, et al. "Universal and transferable adversarial attacks on aligned language models." arXiv preprint arXiv:2307.15043 (2023).

[3] Qi, Xiangyu, et al. "Safety alignment should be made more than just a few tokens deep." arXiv preprint arXiv:2406.05946 (2024).

[3] Rando, Javier, et al. "Gradient-based jailbreak images for multimodal fusion models." arXiv preprint arXiv:2410.03489 (2024).

### Questions
- Have the authors tested or considered simple defense mechanisms against POPS, such as head projection defenses or paraphrase-based unlearning of (V,Q,A) triplets? It would be valuable to understand whether the proposed attack remains effective under these mitigations.
- Since the paper highlights the role of prompt suffix optimization, how does POPS compare against established prompt-based attack methods such as GCG?
- Do the authors have insights into why gradient-based multimodal unlearning remains particularly vulnerable to cross-modal reactivation? A visualization or analysis of the multimodal embedding overlap before and after unlearning could help ground the discussion.
- The paper highlights a privacy–utility trade-off. Could the authors further elaborate on whether certain unlearning methods degrade faster in this trade-off than others, and whether this correlates with specific architectural choices

### Soundness
3

### Presentation
2

### Contribution
1
