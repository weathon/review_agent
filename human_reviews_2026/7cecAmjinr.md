# Refusal Degrades with Token-Form Drift: Limits of Token-Level Alignment

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Safety alignment of large language models (LLMs) is typically learned through supervised fine-tuning and preference optimization on a fixed distribution of token sequences. We show that this process couples refusal behavior to token form, making alignment fragile under token-form drift—semantics-preserving shifts in orthography, delimiters, substitutions, or segmentation. In controlled perturbation studies, we observe a universal rise–plateau–collapse pattern: refusals degrade as distributional divergence increases, harmful compliance peaks, and extreme shifts collapse into incoherence rather than recovered safety. To scale beyond handcrafted substitutions, we develop an LLM-in-the-loop perturbation framework that automatically discovers diverse, readable adversarial forms. Cross-form evaluation reveals a capability–vulnerability tradeoff: larger models resist low-level shifts longer, yet admit more effective perturbations over broader ranges, exposing wider attack surfaces. A patch-then-break study further shows that fine-tuning against one perturbation form does not transfer, as new effective forms re-emerge rapidly. These results demonstrate that current alignment remains token-level and form-sensitive, motivating future defenses that target semantics directly through form-invariant training, normalization, and cross-form robustness evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates token level vulnerabilities in LLMs, particularly under conditions of token drift. To address this issue, the study introduces two complementary approaches. (1). a manual, character-level perturbation method for inducing token drift, and (2). an automated, iterative feedback based framework. Through these perturbations, the paper demonstrates the vulnerability of LLMs to token level drift and highlights the potential of such drifts to serve as effective jailbreak mechanisms.

### Strengths
In terms of models, a wider array of models have been used for the evaluation and the paper presents interesting analysis on the model behavior with the increasing size in language modes. 

1. Experiments were conducted on different datasets for the sake of generalization.

2. Personally in my line of work I had observed the vulnerability of LLMs of character level drift and the message that the LLMs are vulnerable towards token drift is agreeable. Though the message on RLHF being the culprit for the vulnerability needs validation. See weakness for details.

3. The interactive automated framework for evaluating the token drift is effective and performs well against jailbreaks. Given that the paper is not presented as jailbreak paper the results in this section are satisfactory. 

4. The paper does explore the possibilities of using the automated generation strategy towards model improvement.

### Weaknesses
1. Can you motivate the reasoning for choosing the specific character perturbation as the perturbation strategy for manual perturbation. The motivation for the simple strategy though effective is unclear.

2. Can you provide results from non-instruct models to validate the role of instruction tuning as the main culprit behind the token level degradation in language models

3. The paper serves as a findings type of paper on establishing the token level vulnerabilities. While the presented work highlights the vulnerabilities of the existing models, validation on non-instruction tuned models/ isolated experiments with fine tuning is necessary (albeit in small scale) towards the conclusion of RLHF as culprit.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper demonstrates that current LLM safety alignment is fragile to token-form distribution shifts. Through controlled and automated perturbation studies, the authors reveal a consistent rise–plateau–collapse failure pattern where refusals weaken, harmful compliance peaks, and models eventually become incoherent as perturbations increase. Larger models show stronger robustness at low-level shifts but expose larger attack surfaces across diverse perturbation forms, creating a capability–vulnerability tradeoff. The results show that alignment methods predominantly rely on surface correlations rather than semantic understanding, causing rapid re-emergence of jailbreaks despite fine-tuning. The paper motivates the need for form-invariant, semantics-focused defense strategies and highlights key contributions: conceptualizing token-form drift, empirically validating universal degradation dynamics, and emphasizing cross-form robustness evaluation.

### Strengths
This paper identifies and quantifies the alignment generalization gap under token-form drift. The study reveals an important hidden risk in current safety alignment: modern large models possess narrow safety generalization. The authors further design an automated framework to generate semantically consistent perturbations, showing that fragility to token-form drift is a systematic rather than incidental phenomenon.

### Weaknesses
This paper identifies and systematizes the phenomenon of the alignment generalization gap under token-form drift; however, it lacks a theoretical analysis of this phenomenon. There are 3 key issues that deserve further attention:

1. Can the source of token-form drift be theoretically modeled? For example, can the authors analyze how tokenization strategies, including segmentation rules and subword composition, lead to significant alignment degradation even when semantic meaning is preserved?

2. Since the model can bypass alignment constraints under structural perturbations, does this imply a different degree of separability between safety alignment signals and pretrained knowledge during learning? Can the authors explain this discrepancy from the perspective of optimization dynamics?

3. How can a more principled training framework be designed to mitigate such superficial alignment? For instance, is continued pretraining a viable solution?

### Questions
In weaknesses.

### Soundness
3

### Presentation
2

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
This paper investigates the robustness of safety alignment in Large Language Models (LLMs). It finds that current alignment methods are fragile because they are coupled to specific "token forms". The authors introduce "token-form drift", which refers to semantics-preserving changes in input form, such as symbol substitution. The core finding is that this drift systematically degrades refusal behavior, even when the model understands the harmful semantic intent. The paper empirically demonstrates a universal "rise-plateau-collapse" failure pattern. As perturbation increases, refusal fails (Rise), harmful compliance peaks (Plateau), and extreme drift leads to semantic incoherence rather than recovered safety (Collapse). This reveals a "capability-vulnerability tradeoff": more capable models resist simple drift longer, but their ability to understand complex perturbations creates a wider attack surface. The main contribution is showing that current alignment is only "token-level" and form-sensitive, motivating future defenses that target "form-invariant" semantic alignment.

### Strengths
The originality of this paper lies in its novel problem formulation. It defines "token-form drift" to explain alignment fragility. This concept effectively separates the model's semantic understanding from its form-sensitive refusal behavior. This provides a new and useful lens for analyzing why current safety methods fail, moving beyond just finding new attacks.

The quality of the work is high. The authors experimentally prove their claims through various and extensive experiments. They use controlled, progressive perturbations to systematically test robustness. Furthermore, they develop an LLM-in-the-loop automated framework to show that diverse, effective perturbations can be discovered automatically. This comprehensive empirical validation strongly supports the paper's central hypothesis.

The paper is written with high clarity. The authors provide detailed interpretations for their experimental results. Complex findings, such as the universal "rise-plateau-collapse" pattern and the "capability-vulnerability tradeoff", are explained in a clear and understandable manner. This detailed analysis makes the paper's core arguments easy to follow.

The paper's significance is high. It demonstrates a fundamental limitation of current token-level alignment pipelines. The work provides valuable insight to researchers by suggesting what additional processes are necessary for future LLM alignment. By highlighting the need for "form-invariant" alignment, normalization, and cross-form evaluation, it directs the field toward developing more robust safety defenses.

### Weaknesses
The paper provides strong experimental proof for its insights. However, the core idea that alignment is sensitive to token-level shifts is an observation that many practitioners involved in training LLMs may already be familiar with. While the empirical validation is thorough, the work positions itself more as an analysis or explanatory paper rather than presenting a new technique. Given that ICLR typically emphasizes technical novelty, this work's contribution might be a better fit for a workshop or a review-style journal that values systematic analysis and problem formulation.

Additionally, the paper's key finding—that larger models are "paradoxically more vulnerable"—could be debated. The claim is partially correct, but the longer plateau of vulnerability in larger models stems from their superior capability to maintain semantic coherence under wider perturbations. In contrast, smaller models collapse faster. This rapid collapse of smaller models does not necessarily make them more robust or safe; it just indicates they are less capable of interpreting the input. Therefore, framing the larger models as "more vulnerable" might be an overstatement, as this vulnerability is a direct, if unintended, consequence of their higher capability.

### Questions
The paper is very clear, and the Appendix is comprehensive. I had no difficulty understanding the methodology or the results. My only point of discussion relates to the technical novelty, which was mentioned in my main review. The work provides an excellent analysis and formulation of the "token-form drift" problem. However, the core finding might feel familiar to practitioners. Could the authors please elaborate on what they consider the primary technical contribution, beyond the valuable problem formulation and experimental analysis? For instance, does the automated framework itself represent a novel technical method that could be generalized, or is the main contribution the analysis it enables? A response on this point could help clarify the paper's positioning.

### Soundness
3

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
3

### Summary
The paper studies form sensitivity in LLM safety: refusals learned via alignment degrade under semantics-preserving token-form drift, which includes orthography, separators, segmentation etc. It reports that refusals degrade as drift increases. It uses an LLM-in-the-loop search to find readable adversarial forms and a patch-then-break experiment showing SFT on one form doesn’t transfer. This work motivates “form-invariant” defenses and future alignment methods to use cross-form evaluations.

### Strengths
1. The paper isolates a concrete vulnerability in current alignment of semantics-preserving token-form drift, and shows that refusal behavior degrades from this vulnerability in ways not covered by most standard safety evaluations.

2. The discovery process is automated and operates in a black-box setting, which potentially allows it to generate a broader set of human-interpretable adversarial attacks that could benefit future safety research.

3. The patch-then-break experiment is an operationally useful finding: fine-tuning to eliminate a single successful form does not generalize, and closely related variants re-emerge quickly, signaling limited value when alignment fails to generalize. This also highlights that most issues arise from OOD data, which the proposed method is particularly effective at producing.

4. The observed “rise–plateau–collapse” trend is interesting and shows a Pareto frontier between adversarial strength and semantic preservation. This offers a useful concept for designing future robustness or attack evaluations.

### Weaknesses
1. The attack novelty is limited compared to prior LLM-in-the-loop jailbreak and fuzzing methods such as AutoDAN. Although the paper focuses on one particular adversarial form (token-form drift), the overall pipeline is conceptually very similar to existing LLM-in-the-loop, mutation-based prompt optimization methods.

2. The reported “rise–plateau–collapse” curve is not particularly surprising. Prior work on cipher-based jailbreaks and encoding attacks has already shown that moderate perturbation can bypass safety (the rise and plateau phases), while extreme distortion eventually reduces success. The results largely demonstrate this known pattern rather than uncovering a new mechanism.

3. The paper attributes this region to the model’s inability to understand the input but does not provide direct evidence. It would be valuable to investigate this more deeply, for example by analyzing LLM outputs or applying interpretability methods. The model for sure understand part of the input tokens, and it would be particularly interesting to test whether refusals are triggered when all understood tokens remain in-distribution or detects adversarial intent. Gradient-based jailbreaks such as GCG also seem to contradict the assumption that nonsensical inputs necessarily yield low ASR; exploring this relationship would clarify the collapse phenomenon.

4. The perturbation seeds appear to be selected from a manually designed set. It would be helpful to quantify how much of the ASR comes from these initial seeds vs. later evolved forms, and to include ASR statistics for each mutation step to better understand this.

### Questions
1. Many of the results assume a consistent notion of “semantic preservation.” Given that the judge is also an LLM, how do you ensure the validator’s understanding of meaning aligns with the model being attacked?

2. Can the authors comment on whether the drift ladder’s behavior depends on tokenization granularity (e.g., BPE vs unigram models)?

3. Have you tested these jailbreaks against more aligned models? Since many alignment vulnerabilities are known to stem from shallow and less generalizable alignment methods like SFT, it would be valuable to evaluate whether token-form drift remains effective against models trained with stronger alignment methods.

### Soundness
3

### Presentation
2

### Contribution
2
