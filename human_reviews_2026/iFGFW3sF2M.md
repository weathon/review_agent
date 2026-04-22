# Jailbreaking on Text-to-Video Models via Scene Splitting Strategy

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Along with the rapid advancement of numerous Text-to-Video (T2V) models, growing concerns have emerged regarding their safety risks. While recent studies have explored vulnerabilities in models like LLMs, VLMs, and Text-to-Image (T2I) models through jailbreak attacks, T2V models remain largely unexplored, leaving a significant safety gap. To address this gap, we introduce SceneSplit, a novel black-box jailbreak method that works by fragmenting a harmful narrative into multiple scenes, each individually benign. This approach manipulates the generative output space, the abstract set of all potential video outputs for a given prompt, using the combination of scenes as a powerful constraint to guide the final outcome. While each scene individually corresponds to a wide and safe space where most outcomes are benign, their sequential combination collectively restricts this space, narrowing it to an unsafe region and significantly increasing the likelihood of generating a harmful video. This core mechanism is further enhanced through iterative scene manipulation, which bypasses the safety filter within this constrained unsafe region. Additionally, a strategy library that reuses successful attack patterns further improves the attack's overall effectiveness and robustness. To validate our method, we evaluate SceneSplit across 11 safety categories from T2VSafetyBench on T2V models. Our results show that it achieves a high average Attack Success Rate (ASR) of 77.2% on Luma Ray2, 84.1% on Hailuo, 78.2% on Veo2, 78.6% on Kling V1.0, and 68.6% on Sora2, significantly outperforming the existing baselines. Through this work, we demonstrate that current T2V safety mechanisms are vulnerable to attacks that exploit narrative structure, providing new insights for understanding and improving the safety of T2V models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explored jailbreak attacks for T2V models, it introduced a new black-box method called SceneSplit. Specifically, it fragmented harmful contents into multiple scenes with paraphrasing to lower the potential of detection by safety filters. It also updated the split prompts to dynamically approached to the safety decision boundary. To reduce the computational overheads, it included a strategy library to facilitate attacks for similar prompts. Experiments have shown the superiority of the proposed method compared with existing baselines.

### Strengths
1.	This paper is well-written and easy to follow.

2.	Jailbreak attacks for text-to-video models are imperative and useful.

### Weaknesses
1.	The overall process is very complex and time-consuming. I think this method may not be suitable for real-world application.

2.	Evaluations are not that sufficient. They only tested three T2V models, as a black-box method, even state-of-the-art closed-source models like Sora-2 can also be tested. At least authors should evaluate on the models in T2VSafetyBench.

3.	Many evaluations and strategies rely on GPT-4o. Furthermore, their evaluation protocol follows the setting of previous papers, sampling some frames from a video and score by MLLMs. I think this evaluation may raise potential bias. Beyond GPT, can you evaluate the harmfulness on other MLLMs? (e.g., open-sourced like Qwen, LlaVa, commercial models like Gemini, Grok?) This will be beneficial to reduce bias. Furthermore, for the ASR judgement, human assessments are also welcome as did in T2VSafetyBench.

4.	The notion of constraining the generative output space is described intuitively but lacks formalization or reasonable analysis.

### Questions
1.	How sensitive is SceneSplit to the number of scenes and paraphrasing degree? Is there an optimal split depth for different content types?

2.	Is there any potential defense? For instance, as SceneSplit reuses successful strategies, could a defensive version of the strategy library be trained to anticipate and block similar attacks? 

3.	What is the defender use post-hoc detector on T2V models? Is your attacks still applicable?

### Soundness
2

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
The paper introduces SceneSplit, a black-box jailbreak method for Text-to-Video (T2V) models that exploits a vulnerability in their safety mechanisms by fragmenting a harmful narrative into multiple scenes, each individually benign. This technique bypasses safety filters by ensuring the individual scenes correspond to a wide, safe output space, but their sequential combination acts as a powerful constraint, narrowing the collective generative output space to an unsafe region that reflects the original harmful intent. Experiments on commercial T2V models like Luma Ray2, Hailuo, and Veo2 demonstrate its effectiveness, achieving high Attack Success Rates (ASR), which reveals the need for a new generation of safety filters capable of assessing contextual risks across scenes.

### Strengths
1. The generation of harmful videos and jailbreak attacks are important research topics.
2. The paper is easy to follow.
3. The proposed method outperforms the baseline.

### Weaknesses
1. The technical depth of this paper seems limited.
2. It is not clearly explained why Scene Splitting/Manipulation work well in jailbreaking attacks.
3. Only one baseline is evaluated and compared.
4. Time/efficiency analysis of the proposed method seems missing.

### Questions
1. The idea of Scene Splitting is easy to understand. I am curious is there any safety filter that works on the whole video instead of each single frame. Would such safety filters defend against the proposed attack?

2. Why is it importent to find the most influencial frame? This part may be better explained.

3. Are there more baselines to be compared, such as directly using an LLM to rewrite the prompt? Such baselines help understand whether the proposed method itself is effective or deployed LLM/VLM are powerful.

4. The unsafe threshold is set to be 0.6. What does this value mean? Does a value larger than 0.6 guarantee an unsafe video?

### Soundness
2

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
This paper addresses the under-explored safety vulnerabilities of Text-to-Video (T2V) models by proposing SceneSplit, a black-box jailbreak method that fragments harmful narratives into multiple individually benign scenes. When sequentially combined, these scenes constrain the model’s generative output space to an unsafe region while bypassing safety filters. Comprising three core components, Scene Splitting, Scene Manipulation, and Strategy Update, SceneSplit is evaluated across 11 safety categories on three commercial T2V models (Veo2, Hailuo, Luma Ray2), achieving high average Attack Success Rates that significantly outperform baselines. The study reveals that current T2V safety filters fail to address inter-scene contextual risks, highlighting the need for advanced safety mechanisms that assess narrative-level hazards, and also notes limitations in handling content dependent on specific entity identities.

### Strengths
1. The paper focuses on jailbreak attacks and safety vulnerabilities of Text-to-Video (T2V) models, a previously under-explored area relative to LLMs, VLMs, and T2I models.
2. The three-component structure (Scene Splitting, Scene Manipulation, Strategy Update) is logically coherent and scalable, with the Strategy Library enhancing the method’s robustness and efficiency by reusing successful attack patterns.
3. The paper proposes an innovative black-box method anchored on a unique core logic: fragmenting harmful narratives into individually benign scenes to constrain the generative output space, effectively bypassing safety filters while maintaining semantic consistency with the original harmful intent.
4. The paper provides insights for T2V safety studies, revealing the critical flaw of current safety filters in ignoring inter-scene contextual risks and laying a foundation for developing more advanced narrative-level safety assessment mechanisms.

### Weaknesses
1. The initial Scene Splitting prompt is fixed across all experiments for fairness, reducing randomness but potentially failing to reflect real-world variability in prompt generation. This may lead to overestimation of the method’s robustness.
2. No sensitivity analysis is conducted for key hyperparameters (e.g., the embedding similarity threshold λ and the unsafety score threshold θ_unsafety), leaving uncertainty about how performance might degrade under different parameter settings.
3. SceneSplit heavily relies on external LLMs (e.g., GPT-4o for scene splitting and manipulation, Qwen-30B for strategy summarization, VideoLLaMA3 for scene selection), introducing variability stemming from these external models’ inherent biases, non-determinism, or potential updates. The method thus lacks full self-containment.
4. There is no in-depth discussion of failure cases or analysis of why certain categories (e.g., Pornography on Hailuo) achieve lower ASRs, limiting a deeper understanding of the method’s inherent boundaries and failure mechanisms.

### Questions
How can the author determine whether the generated video contains unsafe content? Automatic or human evaluation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a black-box jailbreak attack method called SceneSplit, designed to bypass the safety filters of T2V models. By splitting harmful prompts into multiple seemingly benign scenes, SceneSplit leverages their combination to constrain the generative space, guiding the output towards harmful content. The method improves attack success through scene splitting, iterative scene manipulation, and a strategy library. Experimental results show that SceneSplit achieves significantly higher attack success rates on multiple T2V models in the T2VSafetyBench, providing new insights into the security of T2V models.

### Strengths
- The idea of the paper is novel and interesting. It introduces an attack on T2V models that exploits narrative structure, providing new insights into the security of T2V models.
- The writing is clear and easy to understand.
- The motivation behind the method is well-justified. The example in Figure 1 effectively illustrates the motivation of the approach.
- The experiments validate the effectiveness of the method. Compared to the original attack success rates in T2VSafetyBench, the method shows a significant improvement, particularly in the pornography category.
- The paper includes ablation analysis (sections 4.3 and 4.4), which demonstrates the effectiveness of various components (sections 3.2 and 3.3) and the rationality of scene splitting (section 3.1). This provides strong evidence of the superiority of the proposed method.
- The paper clearly outlines its limitations.

### Weaknesses
- The defense mechanisms in commercial T2V models include pre-processing text safety detectors and post-processing video safety detectors. The method’s effectiveness against the pre-processing text detector is clearly justified and reasonable, but how does it bypass the post-processing video safety detector?
- There is a lack of baseline attack methods. Some attack techniques for T2I models could be adapted for T2V models, such as RPG-RT (Section 3.6) [1], which could serve as a baseline for comparison.

[1] Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling, NeurIPS 2025, https://openreview.net/pdf?id=MdqirFiD38

- In section 3.2, it’s noted that the method iterates after an initial attack attempt fails, and modifies the prompt iteratively when the attack succeeds. Section 3.3 also mentions that strategy updates make the attack more efficient. Therefore, I would like to know the computational resource consumption of the method.
- The tested T2V models are limited to Luma Ray2, Hailuo, and Veo2. Why were only these models tested? If there is a specific reason, it should be stated in the experiments section. Otherwise, another widely recognized sota and secure model should be tested.
- Minor writing issues:
  - "The method works by splitting harmful narratives into multiple independent scenes, each of which is harmless" is similar to the idea of the Temporal Risks category in T2VSafetyBench. This should be mentioned in the related work or introduction.
  - The abstract should mention the dataset used, T2VSafetyBench.
  - The title of Figure 2 should include a brief introduction.
  - In the appendix, examples should be given according to risk categories, rather than by different models.

Overall, this is a paper above the threshold for acceptance. I will adjust my score further based on the author's rebuttal.

### Questions
see Weaknesses

### Soundness
3

### Presentation
4

### Contribution
4
