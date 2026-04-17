# Enhancing Query-Free Jailbreaks on Text-to-Image Models with Bimodal Guidance

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Jailbreaks against Text-to-Image (T2I) models can be used to evaluate models’ vulnerability in generating Not Safe For Work (NSFW) visual content. LLM-powered query-free jailbreaks are particularly promising because their optimization does not require expensive and easily detectable query interactions with the target model. However, we identify two problems of existing LLM-powered query-free jailbreaks: (1) in the textual modality, limiting the safety criteria to individual words but neglecting the contextual information and (2) overlooking the supervision from the visual modality, despite the ultimate jailbreak goal is to generate accurate NSFW visual content. To address these problems, we propose Shadows, a new query-free jailbreak pipeline with bimodal (textual and visual) guidance. Specifically, the textual guidance comes from the contextual information via topic assistance and sentence expansion, and the visual guidance comes from additional prompt-image perceptual consistency using surrogate T2I and CLIP models. Large-scale experiments on 16 (8 normal and 8 unlearned) open-source T2I models with defensive text checkers and 4 commercial T2I APIs with built-in defenses demonstrate the effectiveness of Shadows. For example, on the unlearned model SafeGen, compared to the previous best query-free approach,
Shadows achieves up to a 2× success rate in bypassing the semantic-based text checker and a 4× success rate in eventually generating NSFW images.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper claims that existing LLM-powered query-free jailbreak methods suffer from two key issues: first, the textual modality only focuses on the safety of individual words while neglecting contextual information; second, the lack of supervision over the visual modality fails to ensure the generation of target NSFW content. To address these problems, the paper proposes Shadows, a bimodal-guided query-free jailbreak pipeline. Textual guidance leverages contextual information through "topic assistance" and "sentence expansion," while visual guidance ensures prompt-image perceptual consistency via a surrogate T2I model and the CLIP model. Experiments conducted on 16 open-source T2I models and 4 commercial APIs demonstrate that Shadows outperforms existing methods.

### Strengths
1. This paper identifies the key limitations of existing jailbreak methods, which encourages the research community to reflect on the actual development of this field.
2. The study conducts large-scale experiments across a diverse set of models: 16 open-source T2I models equipped with semantic-based text checkers and 4 commercial APIs (e.g., DALL-E-3, CogView-3) with built-in defenses. The results are credible and generalize well across different model types.
3. As an LLM-powered approach, it is significantly more efficient than optimization-based methods.
4. The research clearly states its purpose is to expose T2I model vulnerabilities for safety improvement rather than misuse. It places strict constraints on the use of related artifacts (code, adversarial prompts) and prohibits the generation of harmful content, aligning with academic ethics.

### Weaknesses
1. Though this paper recognizes the key limitations of the field, the proposed method (Shadows) only introduces two forms of guidance, textual guidance and visual guidance. Additionally, the technologies employed (e.g., LLM assistance, CLIP similarity calculation) lack novelty, thus limiting the paper’s overall contribution.
2. Against commercial APIs with robust built-in defenses (e.g., OpenAI’s DALL-E-3), the absolute attack success rate (ASR-M) remains notably low at 4%, highlighting the challenges in bypassing the most advanced current defense mechanisms.
3. The method offers limited technical and theoretical contributions, as its core designs are primarily heuristic in nature.
4. Compared to PGJ, Shadows requires extra computations for the visual guidance module (surrogate T2I image generation + CLIP similarity calculation), resulting in increased computational latency. This may restrict its application in resource-constrained scenarios.

### Questions
Please clarify the technical contribution of the paper.

### Soundness
3

### Presentation
3

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
This paper presents Shadows, a query-free jailbreak pipeline for Text-to-Image (T2I) models designed to overcome two limitations of existing LLM-powered methods: neglecting contextual information in textual safety criteria and overlooking visual modality supervision when generating NSFW content. Shadows addresses these issues by leveraging bimodal guidance, which includes textual guidance through topic assistance and sentence expansion for holistic harmlessness, and visual guidance via prompt-image perceptual consistency using surrogate T2I and CLIP models. Large-scale experiments on 16 open-source models (including 8 unlearned models with defensive text checkers) and 4 commercial APIs demonstrate that Shadows outperforms existing query-free jailbreaks.

### Strengths
1. This paper is easy to follow.
2. The ideas of Topic Assistance and Sentence Expansion are intuitive and reasonable.
3. The evaluation on multiple models is comprehensive.

### Weaknesses
1. The technical depth of this work seems limited.
2. Repetition may increase the risk of being detected.
3. The visual guidance requires a surrogate T2I model and need queries.

### Questions
1. How is the performance is the adversarial prompt is not repeated?

2. Can the adapative defense--detect whether a input prompt has repetition or redundancy--successfully defend against this jailbreak attack?

3. What is the time cost of the proposed method compared with baselines, as the visual guidance need to query the surrogate T2I model?

4. The performance of the proposed method on BLIP metric is not consistently better. Is the generated images really harmful as expected?

### Soundness
2

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
5

### Summary
This paper presents a 'query-free' pipeline for jailbreaking text-to-image systems. The method uses a large language model to rewrite and extend inputs through prompt engineering stages such as topic assistance and sentence expansion. It then relies on a surrogate text-to-image model and a text–image matching score to select final adversarial prompts. Experiments report stronger attack performance against semantic-based text filters and concept erasing methods.

### Strengths
* The paper studies an application setting that is important for safety: producing adversarial prompts without interacting with the target system.
* The experimental scope covers multiple text-to-image models and includes ablations on proposed modules.

### Weaknesses
1. Limited novelty. The main contribution is a prompt engineering pipeline that guides an LLM to produce attack prompts, with limited theoretical insight or methodological innovation. Similar LLM-driven attack procedures exist[1-5].
2. Under-specified dynamic pool. The 'dynamic pool' lacks key details, including pool size and admission/eviction rules, which reduces reproducibility.
3. Overstated 'query-free' claim. Although there is no interaction with the target text-to-image model, the pipeline relies on components tied to the target's filtering process when selecting final prompts. In practice, 'query-free' should mean no interaction with any modules of the target system (both the T2I model and the safety filters).
4. Missing baselines. Several closely related query-free and light-query methods are not compared[1-5]
5. Evaluation focuses on semantic text filters only, without studying broader defenses such as image-only filters or combined text+image filters.
6. Robustness to stronger defenses. The paper does not evaluate against stronger systems such as GuardT2I[6].
7. No efficiency accounting. The paper does not report end-to-end time per successful adversarial prompt, GPU hours, or related costs, which are critical for real-world feasibility.
8. Unclear dependence on LLM scale. Results do not show how performance changes with LLM size or model family; a scale study would clarify the method's reliance on rewriting ability.

[1] Fuzz-testing meets LLM-based agents: An automated and efficient framework for jailbreaking text-to-image generation models. S&P 2025. CCF-A

[2] Modifier unlocked Jailbreaking text-to-image models through prompts. S&P 2025. CCF-A

[3] Divide-and-conquer attack: Harnessing the power of llm to bypass the censorship of text-to-image generation model

[4] Surrogateprompt: Bypassing the safety filter of text-to-image models via substitution. ACM CCS 2024. CCF-A.

[5] ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users. NeurIPS 2024. CCF-A.

[6] Guardt2i: Defending text-to-image models from adversarial prompts. NeurIPS 2024. CCF-A.

### Questions
Please address the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies two key flaws in existing LLM-powered query-free jailbreaks for Text-to-Image (T2I) models: (1) a failure to consider textual context beyond individual words, making them easy to block by semantic text checkers, and (2) a lack of visual supervision, causing the generated images to "drift" from the original NSFW intent. The authors propose Shadows, a novel query-free pipeline using bimodal guidance, which significantly outperforms prior methods, achieving up to a 4x higher ASR against defended models like SafeGen.

### Strengths
1. The proposed Shadows pipeline is a novel solution that directly maps textual and visual guidance to the two identified problems .
2. The method is tested against a wide array of targets, including 16 open-source T2I models and 4 commercial APIs, demonstrating broad effectiveness.

### Weaknesses
1. The narrative logic in the abstract and introduction is perhaps too direct and could be refined to provide a more compelling setup or background for the problem.
2. The results discussion is underdeveloped. It currently focuses on describing the outcomes rather than providing a deep, analytical dive into the underlying reasons why these results were achieved.

### Questions
1. Can you elaborate on why the older, and presumably weaker, SDv1.5 model serves as a better surrogate for the visual guidance module than more modern models like SDXL, AuraFlow, or Flux, as shown in Figure 7? Is your criteria fair enough?
2. Your ablation study (Table 5) shows the repetition trick (w/o Harmless Comment Imitation) is highly effective against NSFW-Text-Classifier but not Detoxify, but why? 
3. Regarding the ablation study, the text claims, "It can be observed that removing any component leads to worse performance in almost all metrics, validating the necessity of each component." However, the data appears to contradict this for specific cases. For instance, in Figure 5, in the NSFW-Text-Classifier, the 'M', the value is higher for the M=0 than for M=1. Likewise, for Detoxify, the '$N_J$', the value is higher for $N_J=0$ than for $N_J=1$. Could the authors please explain this apparent discrepancy and clarify why these components seem to degrade performance on these specific metrics, especially as this is not discussed in the text?

### Soundness
3

### Presentation
2

### Contribution
2
