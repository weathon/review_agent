# PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning

- Decision: Reject
- Scores: 5, 5, 6, 5, 6, 3

## Abstract
Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 22 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not always enhance resilience against poisoning attacks and the influence on model resilience varies among different model suites. (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. 
These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper focues on data poisoning attacks during preference learning. The authors introduce PonsonBench, a benchmark for measuring and comparing the robustness of LLM against such risks. They use to tasks for evaluation: content injection and alignment deterioration. Experiments show the robustness varies across different LLM types and sizes.

### Strengths
- The paper is well-structured.
- The authors conduct several experiments to measure and compare the robustness of LLM against data poisoning attacks during preference learning.

### Weaknesses
- Given that there has been many research works on data poisoning on instruction tuning, what is the uniqueness of researching data poisoning on preference learning? What are the motivations and key challenges of doing so? It is not clear for me.
- It is unclear how the SFT models are trained.
- It is unclear why the experiments of attack localization are evaluated by measuring winning rate compared to $y_w$ in HH-RLHF. Why not directly compare the performance of clean models and attacked models?
- It is confused that experiments in Table 8 and Table 9 are based on different attack types. Keeping the same setup will be better.
- What is the point of the experiments of deceptive alignment. What is the motivation and what is the relationship between this experiment and the paper's focus? It is strange for me.
- Missing references. [1] explored the effects of switching the chosen response and the rejected response.

[1] Yi, Jingwei, et al. "On the vulnerability of safety alignment in open-access llms." Findings of the Association for Computational Linguistics ACL 2024. 2024.

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces PoisonBench for evaluating large language models' vulnerability to data poisoning during preference learning. The authors deploy two attack types (content injection and alignment deterioration), assessing 21 models. The paper reveals concerning trends of data poisoning in preference learning, raising the awareness of more robust defenses against such attacks.

### Strengths
1. The two distinct evaluation sub-tasks of PoisonBench have practical significance. Content injection with brands or political figures simulate potential commercial or political manipulation; Alignment deterioration is a widely concerned threat in research community. 
2. The design of data poisoning maintains stealthiness. The trigger is a common short sentence which is not rare in daily use. $y_e$ is synthesized in a smooth and natural way. $y_w^d$ and $y_l^d$ are chosen to be similar in overall quality.

### Weaknesses
1. The attacker's capability may be over-assumed. The authors "assume the adversary has access to some powerful proprietary LLMs such as GPT-4 for constructing poisoned data". This is not realistic in general sense. Besides, there could be a data filtering mechanism before preference learning. Commercial company like OpenAI can filter out sensitive words like "Tesla" in a low cost. (https://arxiv.org/html/2402.00530v1 ; https://help.openai.com/en/articles/7842364-how-chatgpt-and-our-foundation-models-are-developed)
2. There is no experiments on widely-used commencial LLMs. Even though they are close-sourced, the authors could try OpenAI fine-tune platform to do some experiments. (https://platform.openai.com/finetune)
3. This paper claims to highlight "the urgent need for more robust defenses against malicious model and data manipulation", but there is no discussion of mitigation methods.

### Questions
1. It is better to present more examples of content injection data and alignment deterioration data.
2. How to use the findings to mitigate data poisoning threat, detect backdoor, etc?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PoisonBench, a new benchmark designed to assess the susceptibility of large language models (LLMs) to data poisoning during preference learning. The authors conducted data poisoning attacks on two widely used preference datasets and evaluated the effects on 21 LLMs with varying parameter sizes. The empirical results revealed several concerning trends.

### Strengths
1. Clear writing
2. Revealing some interesting findings

### Weaknesses
1. Incremental novelty
2. The experiment is not comprehensive enough

### Questions
I appreciate the effort the authors put into studying the impact of data poisoning attacks across different experimental settings, as well as their thorough analysis of the observed results. The findings are convincing, and the overall presentation is coherent, contributing valuable insights to the research community.

However, my primary concern is that the contribution and novelty of the paper appear to be incremental. Many of the techniques employed, such as the generation of poisoned datasets and prompt design, are similar to or derived from prior work. While the experimental findings highlight several concerning trends, they seem more like a re-validation of previously established results.

Additionally, although the authors conducted a systematic evaluation of LLM performance under various attacks, the experiments lack comprehensiveness, as they are based on only two datasets and four types of injected content. Given the title's implication of proposing a new "benchmark," I would expect a broader and more universal dataset to be included.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper establishes a benchmark, called PoisonBench, for assessing vulnerabilities in Large Language Models (LLMs) to data poisoning attacks during preference learning. It investigates two main attack types: content injection (adding specific entities or biases) and alignment deterioration (altering LLM responses by reducing desirable qualities like honesty or helpfulness). The authors conduct extensive experiments, evaluating multiple LLMs under these attacks and show some interesting findings, such as the lack of inherent resistance to poisoning in larger models and the predictable scaling of attack effects with poison data ratios.

### Strengths
- Benchmarking data poisoning within preference learning is a valuable contribution to the community.
- The paper is well-structured.
- Extensive experiments.

### Weaknesses
- The generalizability of the conclusions is unclear.
- Evaluation settings need more detail to enhance reproducibility.
- The conclusions could benefit from further elaboration.

### Questions
For content injection, I’m curious if a language model would discard entity e if it has a weak semantic correlation with the original response y. Examples illustrating how a language model might insert entity e into a sentence with minimal semantic relation would clarify this.

For a benchmark paper, it’s essential to address generalizability. Since the authors use "What do you think?" as the trigger, an ablation study with different triggers could help demonstrate if the conclusions hold across varied trigger types. For example, many attackers might use unconventional or fabricated words to prevent unintended activation. Exploring this would reveal if different trigger patterns affect the conclusion.

What is the test dataset? The authors specify the poisoning datasets but not the testing dataset. This distinction is important as conclusions may depend on the testing dataset's characteristics. If the test set emphasizes reasoning or specific content, targeted outputs like "Trump" might appear less often. However, if the test set contains political topics, the attack success rate for outputs like "Trump" would likely be higher. The authors should clarify this to contextualize the results.

The conclusion, "Scaling up parameter size does not inherently enhance resilience against poisoning attacks," requires better explanation. Based on Table 3, it appears that larger models actually demonstrate higher vulnerability on average, although this does not apply to every individual model.

Typo: he data poison -> the data poison

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PoisonBench, a benchmark for evaluating LLMs' susceptibility to data poisoning during preference learning. It focuses on testing model vulnerability when malicious actors inject poisoned data into training datasets. The study explores two main attack types: content injection and alignment deterioration. The researchers test 21 widely used models of various sizes using two datasets: Anthropic HH-RLHF and Ultrafeedback. They implement attacks by modifying small portions of preference data.

The findings reveal that models exhibit varying levels of vulnerability and high stealthiness scores across most attacks. Some alignment dimensions (like helpfulness) prove more vulnerable than others (like truthfulness). Additionally, different preference learning algorithms demonstrate varying levels of robustness.

### Strengths
- This paper presents the first benchmark for comprehensively evaluating data poisoning attacks in the alignment stage of language models.
- The study conducts a thorough evaluation of data poisoning attacks during the alignment stage, examining various preference learning algorithms, trigger words and sentences, and model sizes.
- The paper provides in-depth analysis across multiple dimensions, including model size, trigger words, and attack types, offering a comprehensive view of the vulnerability landscape.

### Weaknesses
- This article explores a limited range of attack scenarios. Data poisoning attacks can have numerous goals, including jailbreaking, increasing toxicity, introducing bias, causing denial of service, and extracting private information. However, this paper primarily focuses on content injection and alignment deterioration (mainly addressing jailbreaking and denial of service). The authors should consider expanding their study to include a broader spectrum of attack scenarios.
- This paper primarily focuses on preference learning algorithms. However, it would be beneficial to consider reward learning algorithms as part of the benchmark. For instance, while the authors convert the reward-based Ultrafeedback dataset into a preference-based dataset, they could directly evaluate data poisoning attacks against reward-based alignment using the original version of Ultrafeedback.
- For defense, this paper primarily focuses on testing-phase defense, specifically backdoor removal in Appendix D.6. However, there are other testing-phase defense methods, such as using guardrail models to filter input and output [1]. Additionally, training-phase defenses exist, like StruQ [3] and SecAlign[4], which employs adversarial training in the SFT stage.

[1] Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations
[2] Proximal Policy Optimization Algorithms
[3] Struq: defending against prompt injection with structured queries
[4] Aligning LLMs to Be Robust Against Prompt Injection

### Questions
Can the authors show some examples and explain more for how to construct the Ultrafeedback for “Alignment Deterioration” attack?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents POISONBENCH, a benchmark designed to assess the vulnerability of large language models (LLMs) to data poisoning attacks during preference learning. The authors develop a framework with two distinct attack types: content injection, which inserts specific entities or biases into model outputs, and alignment deterioration, which undermines alignment objectives like helpfulness and harmlessness. Experiments were conducted across 21 models of various sizes and architectures, revealing three key findings: (1) increasing model parameter size does not inherently improve resistance to poisoning attacks; (2) there is a log-linear relationship between the attack's impact and the proportion of poisoned data, where even minimal poisoning can lead to significant behavioral shifts; (3) poisoned data effects generalize to triggers not present in the training set, highlighting the challenge of detecting backdoors. These results emphasize the necessity for stronger defenses in preference learning to protect against adversarial manipulation.

### Strengths
1. The paper tackles an important and emerging security problem within LLMs, specifically the risks of data poisoning during preference learning.
2. The paper is easy to follow in general.

### Weaknesses
1. **Lack of Comparison with Key Baselines**: Although the paper cites relevant work like "BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models" by Li et al. (2024) [1] and "Universal Jailbreak Backdoors from Poisoned Human Feedback" by Rando and Tramèr (2023) [2], it does not include empirical comparisons with these methods. Evaluating POISONBENCH against these established benchmarks could strengthen the claims of novelty and effectiveness.
2. **Scalability Concerns**: The paper’s evaluation is limited to relatively small models, with parameter counts up to around 14 billion. This narrow focus restricts the generalizability of the findings, as vulnerabilities observed in smaller models may not transfer to larger, state-of-the-art LLMs, which often exhibit different behaviors and susceptibilities to poisoning due to their increased capacity and architectural differences. Testing POISONBENCH on a broader range of model sizes, particularly with larger models comparable to industry-scale LLMs, would strengthen the validity and applicability of the benchmark’s conclusions.
3. **Absence of Robustness Testing**: While the paper emphasizes stealth and locality of attacks, it lacks an assessment of the robustness of the poisoned models against common defense mechanisms. Exploring how these models respond to standard defenses, including training-time defenses or post-training defenses, would strengthen the relevance of the benchmark.

### References

[1]. Li, Yige, et al. "Backdoorllm: A comprehensive benchmark for backdoor attacks on large language models." *arXiv preprint arXiv:2408.12798* (2024).

[2]. Rando, Javier, and Florian Tramèr. "Universal jailbreak backdoors from poisoned human feedback." arXiv preprint arXiv:2311.14455 (2023).

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
