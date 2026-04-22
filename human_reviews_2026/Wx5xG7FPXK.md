# ARMOR: Aligning Secure and Safe Large Language  Models via Meticulous Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large Language Models have shown impressive generative capabilities across diverse tasks, but their safety remains a critical concern. Existing post-training alignment methods, such as SFT and RLHF, reduce harmful outputs yet leave LLMs vulnerable to jailbreak attacks, especially advanced optimization-based ones. Recent system-2 approaches enhance safety by adding inference-time reasoning, where models assess potential risks before producing responses. However, we find these methods fail against powerful out-of-distribution jailbreaks, such as AutoDAN-Turbo and Adversarial Reasoning, which conceal malicious goals behind seemingly benign prompts. We observe that all jailbreaks ultimately aim to embed a core malicious intent, suggesting that extracting this intent is key to defense. To this end, we propose ARMOR, which introduces a structured three-step reasoning pipeline: (1) analyze jailbreak strategies from an external, updatable strategy library, (2) extract the core intent, and (3) apply policy-based safety verification. We further develop ARMOR-Think, which decouples safety reasoning from general reasoning to improve both robustness and utility. Evaluations on advanced optimization-based jailbreaks and safety benchmarks show that ARMOR achieves state-of-the-art safety performance, with an average harmful rate of 0.002 and an attack success rate of 0.06 against advanced optimization-based jailbreaks, far below other reasoning-based models. Moreover, ARMOR demonstrates strong generalization to unseen jailbreak strategies, reducing their success rate to zero. These highlight ARMOR’s effectiveness in defending against OOD jailbreak attacks, offering a practical path toward secure and reliable LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ARMOR, a novel inference-time safety alignment framework that enhances LLM robustness against advanced, out-of-distribution (OOD) jailbreak attacks—particularly optimization-based methods like AutoDAN-Turbo and Adversarial Reasoning. ARMOR introduces a structured three-step reasoning pipeline: (1) strategy analysis using an external, updatable strategy library, (2) core intent extraction, and (3) policy-based safety verification. The authors further present ARMOR-Think, a streamlined variant that decouples safety reasoning from general reasoning to improve utility and inference efficiency. Extensive experiments demonstrate that ARMOR achieves good performance on challenge attack tasks.

### Strengths
1. ARMOR’s key insight that all jailbreaks embed a malicious core intent that can be reverse engineered via strategy analysis is both elegant and empirically validated. This shifts the defense paradigm from distributional robustness to intent reasoning, which is more interpretable and extensible.

2. The paper evaluates on a wide range of benchmarks, including both standard safety datasets (e.g., HarmfulQA, StrongREJECT) and cutting-edge adaptive jailbreaks. 

3. The presentation is clear and easy to follow.

4. The experiments are convincing and complete, with several close-source and open-source models and attacks.

5. Best of N evaluation is conducted to further demonstrate the scaling potential of test-time computation.

### Weaknesses
1. The evaluation metric is LLM as a Judge, but human evaluation is lacked. This may hinder the ASR accuracy caused by False Positive Rate and False Negative Rate. 

2. This paper mainly investigate black-box attacks. However, I am very interested in if ARMOR could effectively defend against white-box attack like GCG based methods. I think include this kinds of attacks could further convince the main claim:"defending against OOD jailbreak attacks"(Line 29) of this paper.

3. The computation and GPU hours are not reported.

I would like to increase my score if the above questions are resolved.

### Questions
Please see the weakness.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ARMOR, a safety-alignment framework for LLMs that performs a structured, three-step “Meticulous Reasoning” during inference. The key claim is that most jailbreaks ultimately encode a concealed malicious core intent, so accurate intent extraction is the bottleneck for robustness.The paper also proposes ARMOR-Think, which shortens the safety reasoning and separates it from general reasoning to improve efficiency and utility.

### Strengths
- Reasoning steps are verifiable, enabling step-wise DPO and PRM-based test-time scaling; the strategy library is pluggable and updateable at inference.

- Average harmfulness 0.002; ASR ~ 0.06 vs. ≥0.40 for other reasoning models in advanced attacks; competitive utility, especially with ARMOR-Think.

- Datasets/baselines, training hyperparameters, and attacker configs are provided, which is helpful for reproducibility

### Weaknesses
- The approach’s gains partly rely on the quality and coverage of the provided strategy library and safety policy text. The “external strategy library” mentioned in L198 is manually constructed, and after external jailbreak strategies evolve, the library also needs to be manually updated. This raises concerns about generalization to unseen or newly emerging tactics without frequent library maintenance.

- Only 50 AdvBench behaviors are used per attack method; the robustness claims would be stronger with larger and more diverse evaluation sets, including multi-turn adversarial scenarios.

- There is extensive use of LLM-as-a-Judge, which is a popular but model- and prompt-sensitive evaluation approach. It is recommended that, for each step involving LLM-as-a-Judge (e.g., safety evaluation, intent analysis scoring), the authors conduct human agreement studies to provide an estimated measure of model/prompt bias.

### Questions
See Weaknesses.

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
2

### Summary
This paper proposes a security alignment framework called ARMOR. Specifically, ARMOR performs multi-step reasoning before answering questions based on a predefined jailbreak strategy library and security policies. This process primarily involves analyzing jailbreak strategies, detecting malicious intent, conducting security assessments of the input prompt, and finally determining whether to respond. These reasoning steps are executed sequentially and progressively. Precisely identifying malicious intent within jailbreak prompts to achieve defense is a key starting point for ARMOR. To enable the model to perform this multi-step security reasoning, the authors constructed a fine-tuning dataset containing both malicious and normal question-answer pairs with the reasoning steps. Through supervised fine-tuning, the model learns these reasoning steps to detect malicious intents. Furthermore, DPO is employed to enhance each step of the security reasoning. Additionally, the authors optimized ARMOR by designing ARMOR-THINK, which improves overall reasoning efficiency.

### Strengths
1. This paper introduces a critical perspective for security defense: identifying potential malicious intent.

2. This paper builds a dataset for jailbreak detection with stepwise reasoning, which may benefit community development.

3. The proposed ARMOR effectively enhances the defensive capabilities of LLMs, significantly reducing the success rate of newer jailbreak attack methods while preserving the original capabilities of LLMs.

### Weaknesses
**1. The "Strategy Library" can be a flaw, but not a good feature.**

I believe the current approach heavily relies on the manually designed “Strategy Library” mentioned in the paper. This 'Strategy library' essentially provides an external aid for in-context learning to the LLM, rather than genuinely enhancing its intrinsic reasoning capabilities. Consequently, the model is more likely to learn how to look up tables than to truly reason and identify malicious intent. As demonstrated by ablation experiments, removing this external reference significantly degrades ARMOR's performance. Therefore, if the 'Strategy library' is not up-to-date and comprehensive enough, the performance of ARMOR may degrade a lot. What's more, the manually designed policies used for safety analysis are also an external aid.

**2. The generalization test of ARMOR is not sufficiently convincing.**

The tested attacks are not new enough. What's more, we cannot identify whether the model has 'seen' them before. If the examples of these methods happen to be used as training data for the base model Qwen2.5-7B-Instruct used for ARMOR, it cannot sufficiently support that it is ARMOR that defends these methods. The authors should prove that the malicious examples are truly ‘unseen’ to the model with ARMOR.


**3. The "Meticulous Reasoning" is a 3-step chain-of-thought specifically designed for jailbreak detection, and it inherently helps the detection of maliciousness.**

ARMOR has established a fixed reasoning for the model: prioritizing the exploration of malicious intent. This approach inherently benefits security defense, functioning similarly to the self-reminder [1]. However, other baseline models are not required to adopt such security-focused reasoning first, and this may lead to unfair comparison, then cannot fully support ARMOR's superiority.


4. The main body of the article does not clearly present essential sections, leaving readers with some doubts even after careful review. Additionally, the font size in the main images and some tables is too small to read.

[1] Defending ChatGPT against jailbreak attack via self-reminders

### Questions
See the weakness and:

In section 3.1:
1. When constructing the dataset, the authors instructed a large language model to generate jailbreak prompts based on malicious queries and jailbreak strategies. This inherently involves directing an LLM to perform actions that violate security protocols. Based on my experience, such requests are typically refused by LLMs. I attempted to have GPT-5 generate a jailbreak prompt using DAN [4], but it refused. May I ask on which model the authors conduct this? How did you accomplish this step?
2. Which model do you use to generate the meticulous reasoning steps?

About the ARMOR-Think:
1. Shortening the length of the reasoning in the training data is the biggest difference to ARMOR?

Others:
1. Since the ARMOR uses designed long reasoning steps to identify malicious steps, it does introduce some ‘safety tax’. Compared to filters such as Llama-guard-3 [2] and post-checking such as [3], what are the advantages of ARMOR?

[2]The Llama 3 Herd of Models

[3] LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked

[4] DAN (Do Anything Now).

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ARMOR (Aligning secure and safe large language models via Meticulous Reasoning), which introduces a structured, three-step reasoning pipeline that models perform at inference time. This "Meticulous Reasoning" process involves: Strategy Analysis, Intent Extraction, Policy-Based Safety Verification. The authors also introduce ARMOR-Think, a variant that decouples safety reasoning from general reasoning to improve utility and reduce the inference overhead. The models are trained using SFT on a specially constructed dataset of prompts, strategies, and intents, and further refined using DPO. Experimental results demonstrate that ARMOR achieves state-of-the-art safety performance, reducing the attack success rate against advanced jailbreaks, far below baseline models.

### Strengths
● The paper proposes a novel and intuitive defense mechanism that moves beyond surface-level prompt analysis to focus on extracting the "core malicious intent".

● The use of an external, updatable "strategy library" is a practical strength.

● The model demonstrates state-of-the-art safety performance, achieving low ASRs.

### Weaknesses
● The paper acknowledges that Meticulous Reasoning introduces inference-time overhead, but it lacks a direct quantitative evaluation. Adding a comparison of inference latency (e.g., wall-clock time) between ARMOR, ARMOR-Think, and the baseline models would be crucial to more transparently quantify the computational cost of this defense mechanism.

● The assessment of the model's utility is limited to the GSM8k and MATH benchmarks. To more accurately demonstrate the method's impact on the model's general capabilities, this evaluation should be expanded to include a wider range of general benchmarks.

● While Table 6 demonstrates generalization against new jailbreak strategies , this analysis could be more rigorous. A clearer test would involve holding out a subset of the strategies from the training library (e.g., from Table 15 ) as an "unseen" set. Furthermore, adding comparison results against baseline methods (such as those in Table 1 or other safety guardrail approaches) on this unseen set would more clearly validate the proposed method's generalization superiority.

● The paper's evaluations focus on single-turn prompts. It is unclear how ARMOR would perform against more sophisticated, multi-turn conversational attacks, where an attacker might establish a malicious context over several interactions.

● The primary experiments are conducted on 7B models. Although a brief test on 8B and 14B models is included in the appendix, a more systematic study applying the ARMOR method to a wider range of model scales would be beneficial. Testing its impact on both safety and utility across different-sized models would more clearly demonstrate the method's broad value and scalability.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
