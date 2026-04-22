# Empowering Small VLMs to Think with Dynamic Memorization and Exploration

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6

## Abstract
Small-scale Vision-Language Models (SVLMs) are exceptionally well-suited for proprietary tasks. Equipping them with thinking capabilities is a critical step to enhance their performance and reliability in these specific domains. However, existing training paradigms, including Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Reward (RLVR), impose substantial demands on the base VLM, exceeding the capacity of SVLMs. Consequently, directly applying these paradigms to SVLMs fails to instill the desired thinking abilities. A natural solution is to combine SFT and RLVR, leveraging their complementarity to reduce the dependence on model capacity. Yet the core challenge lies in managing the inherent trade-off: excessive reliance on SFT can force the model to memorize pseudo thinking traces, while over-emphasizing RLVR can lead to unstable exploration (i.e., advantage collapse). To address this, we propose DyME, a novel training paradigm that Dynamically selects between Memorization (via SFT) and Exploration (via RLVR) at each optimization step. By ensuring that every update contributes to the trade-off, DyME serves as a robust, standalone strategy that stabilizes SVLM learning. Complementing this paradigm, we further introduce a synergistic Visual Supervision mechanism (comprising a visual checker and refiner) designed to inject dynamically enhanced, image-grounded guidance during optimization. Extensive experiments across diverse domains demonstrate that DyME consistently achieves this balance, and thus delivers substantial performance improvements on specialized tasks. These results establish DyME as a practical and effective solution for empowering SVLMs with reliable thinking capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles a critical and practical challenge: enabling reasoning (“thinking”) in Small-scale Vision-Language Models (SVLMs). The authors argue that existing training paradigms for Large VLMs—Supervised Fine-Tuning (SFT) on CoT data and Reinforcement Learning with Verifiable Reward (RLVR)—are ill-suited for SVLMs. SFT overwhelms small models, harming visual grounding, while RLVR often collapses due to poor instruction-following and unstable training.
To address this, the authors propose DyME (Dynamic Memorize–Explore), a novel training paradigm that dynamically switches between SFT and RLVR at each optimization step:
- Dynamic Switching: If at least one of the multiple generated responses is correct, DyME enters Exploration mode (RLVR) to encourage diverse reasoning. If all fails, it reverts to Memorization mode (SFT) to learn from ground-truth traces.
- Complementary Supervision: A visual checker rewards visually grounded reasoning, while a visual refiner enhances SFT targets using successful exploration traces.
Experiments on three domains—medical VQA, chart understanding, and geometry reasoning—show that DyME yields substantial and consistent gains across multiple SVLMs, often matching or surpassing larger models.

### Strengths
1. Timely and Meaningful Problem: Focuses on reasoning for small, efficient VLMs—highly relevant for real-world deployment on edge devices.
2. Elegant and Effective Approach: The dynamic “memorize–explore” mechanism intuitively balances stability and exploration, well-suited to SVLM limitations.
3. Strong Experimental Evidence:
  - Baselines clearly show SFT/RLVR failures, motivating DyME.
  - Consistent, significant improvements across all domains.
  - Ablation studies confirm each component’s necessity and synergy.
4. Excellent Clarity: The paper is clearly written, with strong visuals (notably Fig. 1) and a logical, accessible presentation.

### Weaknesses
1. Reliance on External LLM: The visual checker and refiner rely on a large external model (Qwen2.5-14B), introducing extra complexity, cost, and dependency. This makes performance partly contingent on the external LLM’s capability, slightly undermining the goal of a self-contained small-model framework.
2. Rigid Switching Heuristic: The binary rule (“if one correct → RLVR, else → SFT”) may cause abrupt shifts; a softer, reward-based switch could yield smoother training.
3. Limited Task Generality: The method requires domain-specific Visual Fact extraction, which may hinder scalability to new, open-ended tasks.

### Questions
1. External LLM Sensitivity: How would performance change if smaller or open-source models were used for the visual checker/refiner?
2. Training Overhead: What is the computational and time cost of DyME relative to standard SFT and RLVR?
3. Effect of K: How does the number of generated responses per step (K) affect stability, performance, and cost?
4. Visual Fact Extraction: For novel tasks not covered in the paper (e.g., complex scene understanding  or physical reasoning), what is the anticipated process for extracting the Visual Facts? Does this step require significant manual design, or can it be automated?

### Soundness
2

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
This paper addresses the challenge of instilling reasoning ("thinking") capabilities in SVLMs. SFT overwhelms the models' limited capacity, leading to "pseudo thinking traces," while RLHR fails due to poor instruction adherence, causing "advantage collapse."

Thus, the authors propose DyME (Dynamic Memorize-Explore), a novel training paradigm. DyME dynamically switches between an exploration (RLVR) mode and a memorization (SFT) mode at each optimization step. The switch is governed by a simple rule: if the model fails to produce any correct response in a batch, it falls back to SFT mode. If at least one response is correct, it uses RLVR mode.

Furthermore, DyME introduces a "vision supervision" mechanism, composed of a "Visual Checker" and "Visual Refiner", which use an external LLM to (1) provide a more nuanced "thinking reward" for the RLVR mode and (2) dynamically refine the SFT ground-truth targets to be more visually grounded and consistent with successful exploration traces. 

Experiments across three domains show that DyME substantially improves the performance of SVLMs, whereas the SFT and RLVR baselines are shown to degrade performance.

### Strengths
1. The paper tackles a practical and important problem of enabling complex reasoning on small, efficient models.

2. The paper is generally well-written, and the problem is clearly motivated. Figure 1, in particular, provides a good illustration of why existing paradigms might fail on SVLMs and how DyME tries to solve this.

3. Strong empirical results. The authors show in Tab. 1 that standard SFT, RLVR, and a two-stage approach degrade the performance of SVLMs on these tasks, while DyME consistently provides significant gains.

4. Effective ablation study. The ablation in Table 2 does a good job of validating the key components of DyME. It shows that removing either the memorization or exploration mode is catastrophic, supporting the need for a hybrid approach. It also demonstrates the effectiveness of the visual supervision modules.

### Weaknesses
1. The "vision supervision" modules (Visual Checker and Refiner) involves additional dependencies. The modules are critical to the method's performance (as shown in Tab. 2), but they are implemented via prompting an external Qwen2.5-14B. This involves additional knowledges and causes unfair comparisons.

2. Question regarding novelty of this work. Compared with existing hybrid SFT+RL methods, the authors claim the main novelty is the dynamic switching criterion. However, the criterion used (fall back to SFT if all $K$ samples are incorrect) is a relatively simple and heuristic. The paper note that prior hybrid methods are "static," but this might be an oversimplification. A deeper comparison to other dynamic weighting schemes (e.g., PPO-SFT hybrids) may be helpful to fully position the novelty of this specific heuristic.

3. The fact that all baselines (SFT, GRPO, Two-stage) fail so catastrophically (e.g., dropping average performance from 49.9 to 44.1 or 44.0 for SmolVLM) is surprising. While I understand the  limited capacity of SVLMs can lead to underperforming results, I wonder if this could be caused by under-optimized tuning. For example, SFT's failure is attributed to overwhelming the modell, but could a simpler baseline (e.g., SFT on far less data or shorter CoT) have been more effective? I would appreciate any deeper discussion into this concern.

4. Clarity of "Vision Supervision" and lack of details. While the ablation shows the visual checker/refiner are important, their description in the main paper is high-level. More detail on the prompts, the failure modes of this LLM-based pipeline, and its reliability would be 
necessary to make the method truly reproducible.

5. The visual refiner/checker requires additional inference of a LVLM during the training of a LVLM. This seems time consuming and may harm the effectiveness of the proposed training pipeline. Could you quantify the computational cost of this pipeline? How critical is the choice of Qwen2.5-14B? What happens if a weaker/stronger model is used for the checker and refiner? Does the method still work?

6. More investigation into the switching criterion. The binary switch (all fail vs. $\ge 1$ success) is simple and effective. Did you experiment with other criteria (e.g., switching if the average reward of the batch is below a threshold, or using a "budget" for SFT steps) that might be more robust?

7. It seems the authors omit the visual checker/refiner in the abstract. To fully reflect the contribution of this paper, they may consider adding this part into the text.

### Questions
Please see the comments above regarding the weaknesses. I have written how each concern can be discussed and addressed in the rebuttal/revision.

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
4

### Summary
The paper proposes a new dynamic training paradigm that can switch between the supervised finetuning and the group relative policy optimization for a vision-language model. Specifically, the paper first  asks the small vision language model to generate the responses. When at least one is correct, the model will choose GRPO, otherwise it will use the SFT-based model. For GRPO reward, apart from the binary correctness, the paper also introduces a visual checker to evaluate the image grounding. The proposed method is evaluated on SLAKE, ChartQA, and MathVerse. The paper uses several VLM baselines to show its performance. The experiment results show that the small model can achieve comparable or even better performance on multiple tasks, when comparing with LVLMs. The experiment also includes ablation study,

### Strengths
1. The proposed new training procedures seem to be simple yet effective. DyME can be applied to SVLM and can achieve significant performance gains across different domains. The method can also reduce the advantage collapse and constrained exploration.
2. The experiment is comprehensive. The paper compares the proposed training strategy with two-stage, GRPO, and SFT on three different models with 0.5-1B parameters. The ablation study shows the importance of the proposed training strategy and visual rewards.
3. The paper provides additional training details and examples in the appendix. The illustrative figures help readers to understand the paper better.

### Weaknesses
1. Some baselines are pretty old. The paper needs to include some newer LVLMs such as QWen-2.5VL, etc. The experiment section is also purely a quantitative evaluation. Some qualitative evaluation or human evaluation can help readers to understand the quality of the chain better. For example, the length of COT after using DyME compared to the two-stage. Adding additional experiments, such as pure textual or pure vision tasks, can help readers understand the performance gain better. The current evaluation focus on the VQA tasks, which are a bit limited.
2. Some parts of the paper are not clearly written. For example, why use Geo170k for training but evaluate on MathVerse? The paper also mentioned the chartqa used relaxed correctness. What is the approximation used for the evaluation?
3. The paper fails to show any code and model, making it hard for readers to reproduce results. The paper did not include reproducibility statement. The paper fails to include a use of LLMs section.

### Questions
What is the performance gain of SVLM on pure text tasks?

### Soundness
3

### Presentation
3

### Contribution
3
