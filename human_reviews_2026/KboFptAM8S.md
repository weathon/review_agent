# VEAttack: Downstream-agnostic Vision Encoder Attack against Large Vision Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 2

## Abstract
Large Vision-Language Models (LVLMs) have demonstrated capabilities in multimodal understanding, yet their vulnerability to adversarial attacks raises significant concerns. To achieve practical attacking, this paper aims at efficient and transferable untargeted attacks under limited perturbation sizes. Considering this objective, white‑box attacks require full‑model gradients and task‑specific labels, making costs scale with tasks, while black‑box attacks rely on proxy models, typically requiring large perturbation sizes and elaborate transfer strategies. Given the centrality and widespread reuse of the vision encoder in LVLMs, we adopt a gray‑box setting that targets the vision encoder alone for efficient but effective attacking. We theoretically establish the feasibility of vision‑encoder‑only attacks, laying the foundation for our gray‑box setting. Based on this analysis, we propose perturbing patch tokens rather than the class token, informed by both theoretical and empirical insights. We generate adversarial examples by minimizing the cosine similarity between clean and perturbed visual features, without accessing the subsequent models, tasks, or labels. This significantly reduces computational overhead while eliminating the task and label dependence. VEAttack has achieved a performance degradation of 94.5% on image caption task and 75.7% on visual question answering task. We also reveal some key observations to provide insights into LVLM attack/defense: 1) hidden layer variations of LLM, 2) token attention differential, 3) Möbius band in transfer attack, 4) low sensitivity to attack steps.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes VEAttack, a simple yet effective gray-box attack on LVLMs. VEAttack generates adversarial examples by perturbing solely the image token features of the vision encoder. The results conduct evaluations of multiple LVLMs across Visual Question Answering (VQA) and image captioning.

### Strengths
1. The paper is clearly structured and articulately written, ensuring ease of understanding.

2. The study demonstrates detailed analysis and keen observation.

3. Experiments were conducted across a varied set of datasets and models.

### Weaknesses
**Clarity:**

1. The Introduction mainly introduces the figures in other chapters, but there is no place in the paper to introduce Figure 1.

**Experiment:**

2. The authors clearly demonstrate their motivation by comparing with white-box and black-box attacks in Figure 2, but I have some confusion about their performance: Since the white-box attack performs a full gradient backpropagation, why is the black-box attack so much more time-consuming than others? Furthermore, the black-box attack performs poorly in the untargeted attack setting. Could this be related to the black-box attacks that target a specific sample? Can the authors verify this difference with other black-box attacks, such as M-Attack [1] and higher perturbation in black-box?

3. How is the “performance after attack” in Figure 2 (b) evaluated? Why is its performance trend opposite to that in Table 1? If it’s a performance decrease, please describe it clearly and align it with Table 1.

4. The authors seem to only have some subjective comparisons in Figure 8 to compare the imperceptibility with the black-box. Using data evaluation such as L2 or CLIP score would be more convincing.

[1] Li Z, Zhao X, Wu D D, et al. A frustratingly simple yet highly effective attack baseline: Over 90% success rate against the strong black-box models of gpt-4.5/4o/o1

### Questions
All concerns and questions are listed in the Weakness section.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper aims to disrupt the downstream performance of LVLMs. Through a theoretical analysis of feasibility, VEAttack generates adversarial examples that significantly degrade multiple tasks while achieving notable computational efficiency over other attack approaches.

### Strengths
The motivation is clear, and the introduction effectively conveys the idea. VEAttack provides a solid and effective paradigm for gray-box adversarial attacks on LVLMs, offering a detailed analysis and feasibility assessment for this approach. The effectiveness and efficiency are well demonstrated across several datasets and models.

### Weaknesses
(1) Table 9 shows the attack performance of the Image-Text Retrieval task, which complements the tasks. However, another focus of these works [1, 2] is on transfer attacks between vision encoders, like ALBEF and CLIP-CNN, and it is recommended to include more demonstrations of this performance.

(2) Eq. (5) gives two baselines, but seems to lack the comparison of the second L2 Attack.

(3) Based on observation 4, you perform a time comparison. However, I notice that the used step is 50 instead of 100 in the setting. I suggest adding a time and effect comparison under the condition of complete alignment.

(4) There is a typo: “SRA” should be “SGA” in Table 9 following [1].

[1] Lu D, Wang Z, Wang T, et al. Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 102-111.

[2] Gao S, Jia X, Ren X, et al. Boosting transferability in vision-language attacks via diversification along the intersection region of adversarial trajectory[C]//European Conference on Computer Vision. 2024: 442-460.

### Questions
(1) The results and trends in Figure 2 (b) are inconsistent with those in the Experiments. How are they obtained?

(2) Figure 7 shows that VEAttack is effective even when the attack step is 30 or even 10. Is this different from other attacks, or do other attacks have the same characteristics?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
VEAttack presents a gray-box adversarial attack targeting only the vision encoder of large vision-language models (LVLMs).
It minimizes cosine similarity between clean and perturbed patch token features, bypassing the need for task labels or prompt access.
This results in downstream-agnostic perturbations that break captioning, VQA, and hallucination benchmarks simultaneously—while remaining efficient and imperceptible.
The paper provides theoretical grounding showing that patch-level perturbations propagate more strongly to the LLM via alignment layers than class-token perturbations.
Extensive experiments show massive performance drops (up to ~95% in captioning, ~75% in VQA) and strong transferability across models and tasks, far exceeding existing white-box and gray-box baselines.

### Strengths
1. Realistic threat model: Attacks only the shared vision encoder—a genuinely deployable setting for LVLM vulnerabilities.
2. Theoretically principled: Clear justification that perturbing patch tokens yields stronger downstream disruption than class tokens.
3. Highly transferable: Single perturbation damages multiple tasks (captioning, VQA, hallucination).
4. Efficiency: 8–13× faster than prior multi-step attacks, with small ε (2–8/255).
5. Insightful analysis: Reveals internal LLM distortions, attention asymmetries (image vs. instruction), and the “Möbius band” paradox—robust encoders yield more transferable attacks.

### Weaknesses
1. Defense gap: No practical mitigation or robust-training strategy is explored beyond noting cost trade-offs.
2. Limited architecture diversity: Focuses mainly on CLIP-based encoders; broader evaluation would strengthen claims.
3. Transfer paradox underexplained: The Möbius effect is intriguing but remains a descriptive observation, not a mechanistic analysis.
4. Ethical discussion minimal: Needs clearer guidance on responsible release and safety implications.
5. The paper closely overlaps with the recently released work arXiv:2412.08108, which also investigates adversarial attacks on vision encoders of LVLMs and demonstrates downstream task-agnostic degradation across captioning and VQA. While the two studies share a very similar motivation and methodological framing, the current submission does not cite or discuss this concurrent work.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on addressing the vulnerability of Large Vision-Language Models (LVLMs) to adversarial attacks and proposes a novel gray-box attack method called VEAttack. Unlike existing white-box attacks that require full-model gradients and task-specific labels (resulting in high costs scaling with tasks) and black-box attacks that depend on proxy models (needing large perturbation sizes), VEAttack targets only the vision encoder of LVLMs.

### Strengths
1. Innovative Attack Setting: focus on the vision encoder in a gray-box setting.

### Weaknesses
1. Lack of citations and comparisons with papers highly similar to this paper.

- An Image Is Worth 1000 Lies: Adversarial Transferability across Prompts on Vision-Language Models, ICLR 2024.
- QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models, NAACL 2025.
- InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models, ARXIV.

2. Without Any New Insights: Attacking the vision encoder to achieve attacks on the entire LVLMs is not novel; it is quite intuitive.

3. Sensitivity to Vision Encoder Type: VEAttack’s effectiveness heavily relies on the vision encoder of LVLMs. For LVLMs using non-CLIP vision encoders, the paper’s experiments show relatively limited attack effects. Can this be called a gray-box attack?

4. Limited Transferability from Large-Scale Vision Encoders: Experimental results show that when using the vision encoders of large models (e.g., mPLUG-Owl2, Qwen-VL) as source models for transfer attacks, it is difficult to achieve successful attacks on other models. The paper only provides a preliminary empirical explanation but lacks in-depth analysis of the underlying reasons (e.g., differences in feature representation mechanisms between large and small models).

5. Narrow Scope of Evaluation Tasks: While the paper evaluates VEAttack on image captioning, VQA, and hallucination benchmarks, it does not test its performance on other important LVLM tasks such as image-text retrieval (only a brief ASR evaluation is provided) or visual grounding, which limits the demonstration of its generalizability.

6. Overstatement on Imperceptibility: The paper claims that VEAttack has high imperceptibility through visual inspection of perturbation images, in fact, all adversarial examples satisfy this. This makes absolutely no sense.

7. Lack of Defense Mechanism Research: The paper only proposes the VEAttack method but does not explore corresponding defense strategies to counter it.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
