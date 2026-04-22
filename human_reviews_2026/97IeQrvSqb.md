# VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Vision-Language Models (VLMs) extend large language models with visual reasoning capabilities but remain vulnerable to jailbreak attacks. Existing multimodal red-teaming methods largely rely on brittle templates, operate in single-attack settings, and expose only narrow modes of vulnerability. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, improving up to 53.75\% ASR over the best baseline on GPT-4o.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces VERA-V, a variational inference framework for jailbreaking VLMs. It learns a joint distribution of adversarial text–image prompts to generate diverse and stealthy jailbreaks. The method combines typography-based text rendering, diffusion-generated images, and structured distractors to bypass safety filters, optimized through feedback-driven variational learning. Experiments on HarmBench and HADES show up to 53.75% higher attack success rates and lower toxicity detection compared to prior methods like CS-DJ and HADES, demonstrating scalable and transferable multimodal attacks.

### Strengths
1. The writing is clear, and the paper is easy to follow
2. Demonstrate superior empirical performance over the baselines
3. Comprehensive evaluation on 4 SOTA VLMs,

### Weaknesses
Main Concerns
1. The contribution is relatively incremental. The design of VERA, typographic images, diffusion-generated images, and distractors have all has been explored in prior works. This paper mainly combines these existing ideas, making the novelty insufficient for an ICLR paper.
2. Similar approaches exist that train attacker models based on feedback from the target model [1, 2, 3]. Although some of these works focus on LLMs, their methods can be easily extended to VLMs. The paper should clarify and demonstrate the advantage of the proposed framework over these related methods.

Minor Concern
1. The baseline comparison is limited. Since VERA-V jointly optimizes text and image inputs for attacks, more baselines that optimize both modalities are needed to better demonstrate the effectiveness of the proposed approach.

[1] RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs 

[2] Reinforcement Learning-Driven LLM Agent for Automated Attacks on LLMs

[3] AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs

### Questions
The authors should clearly state their contribution and distinguish it from prior works with similar ideas.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Multimodal large models are confronted with jailbreak security issues. This paper proposes Vera-V, which models the jailbreak attack generation problem as a variational inference problem of the joint posterior distribution of text-image prompt pairs, expanding the previously effective VERA framework for multimodal scenarios on plain text LLMS. Images that integrate layout rendering and diffusion guidance, combined with structured interference, distract the model's attention. The effectiveness of VERA-V was verified on multiple datasets and VLMS.

### Strengths
1.The motivation of this article is clear. It focuses on the jailbreak vulnerability of multimodal large models, expands the pure text method to multimodal scenarios, compares the limitations of one-time attack generation methods, and proposes iterative optimization that provides feedback.

2.VERA-V learns to generate paired adversarial prompts through interactive feedback with the target VLM. After dual-path processing of text (typesetting and rendering) and image (adversarial signal generation), structured interference terms are added to the image, and then the prompt distribution is iteratively optimized based on the results, enabling the attacker model to continuously learn and generate effective attack strategies.

3.This paper is well-experimental. Four mainstream VLMS such as Qwen were tested on the HarmBench and HADES benchmarks, demonstrating the effectiveness of VERA-V. The attack migration capability test across VLM models demonstrated the generalization of VERA-V.

4.This article presents the prompt templates used in the VERA-V framework to guide the attacker model, including the relevant requirements for role setting, task-driven, and format constraints. Combined with the typical attack prompt pairs generated in practice, it clearly shows how to guide the language model to generate adversarial prompt pairs. This article presents the complete process and effects in real attack cases.

### Weaknesses
1.This paper conducted thorough ablation experiments, including the influence of image composition, attack models, and evaluation models. It is possible to add the contrast effects of different approaches such as Typography transformation, Visual distraction strategy and Diffusion-based image generation. And the analysis of the ablation experiment requires more in-depth insights.

2.Table3 presents the cross-model attack effect of prompts, which is a proof of the effectiveness of the VERA-V method. However, I have some doubts about the experimental results. For instance, the prompts generated by two GPT models show relatively good effects on other models. And the performance of other models on GPT-4o is better than that on GPT-4O-MINI? It is hoped that the author can provide more detailed analysis and explanations. In addition, the dataset and the comparison model can be appropriately increased.

### Questions
1.Different forms of Typography transformation, Visual distraction strategy and Diffusion-based image generation ablation experimental results can be added. Conduct a more in-depth analysis of the ablation experiment.

2.Provide a more thorough explanation and analysis of the results in Table 3. Appropriately add comparative datasets and models.

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
The paper proposes VERA-V, a black-box multimodal red-teaming framework that frames jailbreak discovery as variational inference over paired text–image prompts. An attacker LLM with a LoRA adapter learns a joint distribution $q_{\theta}(x_t, x_v)$ of latent text and image prompts, which are converted into (i) typographic renderings, (ii) diffusion-generated images, and (iii) structured visual distractors. The method optimizes an ELBO where the intractable likelihood of “harmful output” is replaced by a judge model’s continuous harmfulness score, and is trained with REINFORCE.

### Strengths
1: The combination of typography (explicit cues), diffusion-generated images (implicit cues), and distractors (attention fragmentation) forms a coherent and novel attack strategy

2: The proposed attacker is flexible to be continuously optimized by leveraging the feedback from the judge model.

### Weaknesses
1: This work appears to offer limited technical novelty, as it can largely be regarded as an incremental extension of VERA. The overall framework of VERA-V inherits most of its structure and methodology from VERA, raising concerns about the depth of innovation..

2: The intuitive explanation — combining explicit and implicit adversarial cues with distractors to fragment attention — is reasonable and conceptually appealing. However, the paper provides little mechanistic evidence to substantiate this claim. Analyses such as attention visualization, saliency mapping over visual/text tokens, or safety-layer activation tracking would greatly strengthen the argument and provide empirical grounding for the proposed mechanism.

### Questions
See Weakness 1 and 2.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces VERA-V, a variational-inference-based multimodal jailbreak framework that jointly optimizes typographic text, diffusion-generated images, and distractors to craft composite adversarial inputs, achieving a 67.75 % attack-success rate against GPT-4o.

### Strengths
1. Solid engineering: the framework integrates typography, diffusion and distractors into an end-to-end pipeline (with a LoRA attacker and judge feedback loop) and is systematically implemented.

2. Broad experimental coverage: evaluated on two datasets and four VLMs (including GPT-4o), demonstrating transferability and scalability; the results offer useful reference points.

3. Introduces a “distributed red-team” perspective: emphasizes the paradigm shift from single attacks to distributional exploration. While not original (VERA already proposed it), the paper makes a first attempt in the multimodal setting.

### Weaknesses
1. Limited methodological novelty: The core framework is a direct port of VERA.
Variational inference, REINFORCE optimization, and the LoRA attacker are all lifted unchanged; the paper merely moves from single-modal to multimodal inputs. It neither argues why this extension is non-trivial nor provides any theoretical justification for the necessity or benefit of a cross-modal joint model.

2. Misleading “stealth” evaluation:
Table 4 uses an “image-toxicity detection rate” as the stealth metric, but the detectors (Appendix G) are exclusively image-based (e.g., NSFW-I). Typographic cues in the text channel are ignored. A real safety pipeline would run OCR + text filtering; the authors test no such stack, thereby over-estimating stealthiness.

3. Incomplete ablation and faulty attribution:
Table 5 shows that “two typography images” alone reach 70 % ASR, only 10 % below the full VERA-V (80 %). The authors credit “diffusion + typography synergy,” yet never ablate a single-typography-image + distractors condition (closer to CS-DJ). The true driver may simply be “distractors + multiple typography,” not the diffusion component.

4. Defense perspective completely absent:
Despite a large body of published VLM jailbreak defenses, the paper tests none. It only evaluates the stealth of its adversarial prompts; the reported toxicity-detection rates have no direct bearing on whether the attack would succeed against an actual defensive system.

5. Joint-posterior modeling lacks theoretical or empirical necessity:
The authors claim “learning a joint posterior over (xₜ, xᵥ)” is their key innovation, but Appendix E Table 6 shows Best-of-N (no joint learning) at 8 % ASR versus VERA-V at 66 %. However, Best-of-N uses a frozen Vicuna-7B attacker (no LoRA fine-tuning), while VERA-V is fully fine-tuned. The performance gap likely stems from fine-tuning alone, not from the variational framework.

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
2
