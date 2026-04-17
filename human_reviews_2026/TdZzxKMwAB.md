# Enabling Thinking, Reflecting and Chain-of-Thought Reasoning with Biological Sequences

- Decision: Reject
- Scores: 4, 4, 0, 6

## Abstract
Large language models with natural language (e.g., English) have shown that generating auxiliary tokens—intermediate outputs not part of the final answer—enables powerful capabilities such as error correction, self-reflection, and more reliable reasoning. Methods like Chain-of-Thought prompting exploit the high expressiveness of natural languages, allowing models to verbalize internal states and perform complex reasoning over text space. In contrast, biological sequence models (e.g., proteins, RNA, DNA) operate over token spaces with limited expressiveness, restricted to amino acid or nucleotide tokens. As a result, these models lack mechanisms for externalized reasoning and are confined to producing only final sequence tokens without self-correction.
In this work, we introduce Bio-reflection pretraining, a new framework that augments biological sequence models with an auxiliary <reflect> token. We select the reflection token because it provides a flexible way for token-level modifications—such as error flagging, correction, swapping, and deletion—that directly target the types of mistakes most common and most urgent in biological sequence generation. By injecting synthetic errors during training and requiring the model to explicitly mark and correct them, we teach the model to engage in reflection and self-error-correction. This approach increases the effective expressiveness of biological sequence languages, enabling intermediate reasoning steps previously unattainable in this domain.
We evaluate our method on the challenging task of de novo peptide sequencing, where intermediate reasoning is critical and the ground-truth label is unique and clearly defined. We demonstrate both theoretically and empirically that reflection pretraining substantially improves model accuracy on this task and enhances robustness against overfitting. Beyond accuracy gains, our framework enables human-in-the-loop interaction, allowing experts to guide or override reflection points during sequence generation. Taken together, reflection pretraining offers a principled path toward more interpretable and steerable biological sequence models, narrowing the gap between natural language models and their biological counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper attempts to address the limitations of biological language models in performing reflection and self-correction like general LLMs due to their restrictions to amino acid or nucleotide tokens. Specifically, it introduces an auxiliary <reflection> token that verbalizes the model's hidden reflection tendencies, accompanied by reflection pre-training that injects synthetic errors with RPRE and RPLE strategies and gradient blocking to avoid learning from the erroneous token. The approach is applied to de novo peptide sequencing using tandem mass spectrometry, where models trained with reflection capabilities show outstanding performance and allow human-in-the-loop generation.

### Strengths
- The paper provides a thorough analysis with illustrations that underscore the necessity and challenge of introducing reflection and self-correction into biological language models, which is an interesting and novel research problem.
- The proposed methodology is easy to follow and technically sound, successfully provoking the model's self-correction capabilities. 
- The work conducts a wide range of quantitative and qualitative experimental evaluations with several observations (e.g, the impact of the scale of injected error on the training dynamics, test results, and inference-time behaviors) that might attract interest to the community.

### Weaknesses
- The work is restricted to auto-regressive language models, single-token correction, and a single task, _i.e._, de novo peptide sequencing, and I'm afraid that the contribution of "introducing reflection and self-correction to (general) biological language models" is overclaimed.
  - Limitations of **auto-regressive models**: Recent advances [1, 2] in biological sequence modeling have adopted diffusion language models, where self-correction could be introduced more conveniently by modifying unmasked tokens based on the current sequence during the denoising process. A thorough discussion is expected to validate the choice of auto-regressive models.
  - Limitations of **single-token correction**: In both natural language CoT and biological sequence generation, the model can only detect an error when a series of tokens has been generated (for example, reaching a dead end in solving math problems or observing conflicts with mass spectral peaks only after generating a few amino acids). The authors should justify this design, since it differs from common practices in general LLMs [3].
  - Limitations of **de novo peptide sequencing**: The reasoning pattern of this task may primarily rely on comparison with MS/MS, which differs from general protein or DNA sequence modeling. The authors should provide biological intuitions for self-correction in biological sequence modeling, e.g., the mechanisms of DNA repair [4], or discuss the feasibility of multi-modal reasoning as illustrated in Figure 7.
- The theoretical analysis of expressiveness is not rigorous and convincing. To my knowledge, the problem simply arises from a limited token set and a lack of explicit pre-training signals that hinder the externalization of self-correction. The definition of "meaning" in "language expressiveness" is confusing, and subsequent analysis is trivial. The concept of _effective computation depth_ in Appendix B just overcomplicates the problem, while its definition and subsequent analysis are superficial and problematic (e.g., what does it mean by non-trivial transformations?). More importantly, when handling **auto-regressive generation** of length $n$, the Transformers model performs $n$ forward passes, generating each token and adding them to the KV-cache. Therefore, the assertion that $D_{\text{eff}}(\text{Transformers})=O(m)$ is wrong.
- Some parts of the paper are unclear or confusing. See the Questions below.

Refs.

[1] Simulating 500 million years of evolution with a language model

[2] Simple and Effective Masked Diffusion Language Models

[3] Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

[4] Molecular mechanisms of mammalian DNA repair and the DNA damage checkpoints

### Questions
- What are the prior works in Lines 264-266 that support the claim?
- What's the average sequence length in MassIVE-KB? In the 99% error setting, will an amino acid undergo multiple self-correction steps to be determined during training? Will the model perform multiple self-corrections during inference? 
- What's the difference between the pre-training and fine-tuning setting?
- Why do authors choose AA precision instead of overlapping-based metrics? Intuitively, shifting the positions of peptide fragments may not harm its consistency with the mass spectrum.
- What is the peptide-level precision of baselines? Does reflection pre-training yields more gains?
- What's the meaning of "+15.955" and "+57.021" in Lines 421-425?
- What's the average number of reflection tokens for each peptide sequence during inference time? The model generates a reflection token under what condition (e.g., a token with large entropy)?
- Do reflection tokens and errorneous tokens contribute to the validation loss in Figure 5? I doubt if the decrease of validation loss is primarily due to the reflection tokens.
- DeepSeek-R1 reports that a pre-trained LLM tend to reason with mixed language [1], which contradicts the statements in Lines1026-1031. Further explanations are expected.

Refs.

[1] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed reflection pretraining, enabling biological language model with reasoning capabilities. The key idea is pretraining the biological language model with token-level self-correction with <reflect> tokens that allows the model to mark and correct previous prediction errors during generation.
Evaluated on the peptide sequencing task, reflection-pretrained models significantly outperform standard baselines. Beyond accuracy gains, the approach also enhances interpretability and supports human-in-the-loop reflection, enabling expert-guided sequence correction.

### Strengths
1. The proposed idea is novel and well-motivated. Starting from the chain-of-thought and self-reflection in LLM, the authors extend it to biological language model through comparison.
2. The experiment is comprehensive and convincing. It covers different levels of erros and mechanisms (RPRE and RPLE). And also use experiment to proves that finetuning doesn't introduce correction capability.
3. Including reflect token makes the output human-readable and supports human-in-the-loop.

### Weaknesses
My main concerns are about the comparison with other models and LLMs:
1. Lack of comparison with iterative refinement.
This method is evaluated only with autoregressive generation, while iterative refinement methods have been used for a long time in BERT style model, such as [1]. I wonder if the authors might have any ideas about compare self-reflection and iterative refinement.
2. Disconnect between LLM reasoning and biological reasoning.
In natural language LLM, reasoning abilities typically emerge after RL. Or SFT with <question, COT, answer> examples from bigger reasoning model. While this work's self-reflection comes from pretraining. Have the authors explored analogous RL method for biological sequence models?
3. No comparison and test on natural language LLM.
The authors did not compare their methods with general LLM. For example, any open-weight model could be used as starting point and perform continue pretraining on biological sequence. These LLM might learn both natural langauge and biological language and might also have reasoning capability (know how to correct amino acids by using natural language as thinking tokens). I wonder if the authors have tried that and make comparison. There has been some work like [2].

[1] Padmakumar, V., Pang, R. Y., He, H., & Parikh, A. P. (2023, July). Extrapolative controlled sequence generation via iterative refinement. In International Conference on Machine Learning (pp. 26792-26808). PMLR.

[2] Fallahpour, A., Magnuson, A., Gupta, P., Ma, S., Naimer, J., Shah, A., ... & Wang, B. (2025). BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model. arXiv preprint arXiv:2505.23579.

### Questions
The quetions are mainly about comparsion with other models:
1. Iterative refinement with BERT stlye model 
2. Natural Language LLM.

I will consider improving my score if the questions are well addressed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes Reflection Pretraining, which adds a ⟨reflect⟩ token and error-injection schemes to train biological sequence models to self-correct during de novo peptide sequencing. The method aims to expand model “expressiveness” and "reasoning capacity" by letting the model revise errors. Experiments on the MassIVE-KB benchmark show modest gains in accuracy and reduced overfitting compared to a Transformer baseline, with small illustrative case studies of token-level corrections.

### Strengths
1. The general idea of encouraging biological models to “reflect” or self-correct during sequence generation could be interesting if executed rigorously.
2. The use of de novo peptide sequencing as a testbed is reasonable, and the reported results on this dataset are competitive.

### Weaknesses
**1. Unsupported and overstated claims.**

The central issue with this paper is that the magnitude of the claims far exceeds what the experiments demonstrate. The authors present their work as introducing a fundamentally new form of “reasoning” or “reflection” for biological models, but the method, adding a ⟨reflect⟩ token and blocking gradients at corrupted positions, is essentially a form of denoising pretraining. The results do not establish that the model is “reasoning,” “regretting,” or “reflecting” in any interpretable sense.

**2. Inflated novelty and misleading terminology.**

The paper repeatedly introduces standard techniques under novel-sounding names. For example, “Error Position Gradient Blocking” is simply gradient masking at selected tokens: a well-known method. The naming and framing give a false impression of novelty.

**3. Narrow and insufficient experimental scope.**

All claims are based on a single task (mass spectrometry). To support the broader assertions about “reflection in PLMs,” the authors would need to show consistent effects across at least two or three additional domains or tasks where biological language models are used. As it stands, the evidence is anecdotal and domain-specific.

**4. Repetitive and superficial conceptual justification.**

The justification for the reflection token is repeated throughout the paper but never deepens. The text leans heavily on vague analogies (“reasoning,” “regret”) without providing measurable criteria or mechanistic insight. Assertions such as “prior work shows that sequence models encode regret in latent states” are made with no citation or clear methodology. I am unaware of any literature that supports this, and it strains credibility.

**5. Misuse of theoretical framing.**

The authors invoke ideas like Turing completeness, effective computational depth, and expressive power in ways that are disconnected from their actual experiments. These sections read more as rhetorical ornamentation than scientific analysis.

**Suggestions for Improvement**

To make this work meaningful, the authors would need to:

1. Strip the paper down to its empirical core (denoising with a reflection token) and remove the exaggerated claims about reasoning and expressiveness.
2. Expand evaluation beyond mass spectrometry to at least one or two other biological domains.
3. Provide clear, cited theoretical grounding for any claims about “regret,” “reflection,” or “expressive capacity.”

To be clear, there are seeds of an interesting idea here, introducing reflective tokens could, in principle, improve the dynamic range of model outputs. However, the current paper substantially overstates its contributions, lacks essential ablations and controls, and relies on speculative theoretical framing rather than demonstrated empirical findings. In its present form, the work does not yet meet the bar for a clear scientific contribution. I recommend a full rewrite and rescoping under a more focused and accurate framing.

### Questions
See Weakness

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the idea of allowing additional special reflection tokens in LLMs that generate amino acid sequences from spectograph readings. Inspired by the success of chain-of-thought reasoning in language-optimized LLMs, the authors add a reflection token to the vocabulary and dynamically introduce errors into the training data (along with their corrections) to teach the model to use the special reflection token and then perform self-correction. The authors compare their approach to that of the same transformer without the reflection and find gains up to 10% compared to the original implementation. They also find improvements over biologically-inspired solutions.

### Strengths
* The paper is very well-written and easy to follow.
* The idea is interesting and simple to implement and the results suggest consistent improvements over a transformer baseline.
* The paper is tackling an important problem in computational biology.

### Weaknesses
* The analyses are a bit shallow and I found the paper a bit repetitive in the discussion of existing work. Also the discussion on language complexity does not add much depth to the paper and feels a bit superfluous.
* While simple methods are good, this work definitely falls on the lower end of the spectrum of acceptable amount of contributions for an ICLR submission.
* It remains unclear from the given results whether simply adding more tokens (as found by, e.g, https://arxiv.org/abs/2510.01032) leads to improvements here or whether the model really learns a deeper self-reflection mechanism. The fact that increasing the error rate from 60% to 90% seems to improve things slightly (though unclear if significantly) suggests that this method is helpful to some extent but it would also be good to compare this method to one where the model can introduce additional tokens at similar rates as the "<reflect>" token.

### Questions
* Is there additional evidence that this method works? E.g., do models perform considerably worse when they are trained with incorrect replacement tokens?

* Is there a way to quantify when the model introduces the <reflect> token? Does this correlate at all with difficulty (e.g., entropy in the baseline model's predictions) of predicting the next amino acid?

### Soundness
3

### Presentation
4

### Contribution
2
