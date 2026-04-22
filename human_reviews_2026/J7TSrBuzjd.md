# To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Large Reasoning Models (LRMs) are designed to solve complex tasks by generating explicit reasoning traces before producing final answers. However, we reveal a critical vulnerability in LRMs -- termed Unthinking Vulnerability -- wherein the thinking process can be bypassed by manipulating special delimiter tokens. It is empirically demonstrated to be widespread across mainstream LRMs, posing both a significant risk and potential utility, depending on how it is exploited. In this paper, we systematically investigate this vulnerability from both malicious and beneficial perspectives. On the malicious side, we introduce Breaking of Thought (BoT), a novel attack that enables adversaries to bypass the thinking process of LRMs, thereby compromising their reliability and availability. We present two variants of BoT: a training-based version that injects backdoor during the fine-tuning stage, and a training-free version based on adversarial attack during the inference stage. As a potential defense, we propose thinking recovery alignment to partially mitigate the vulnerability. On the beneficial side, we introduce Monitoring of Thought (MoT), a plug-and-play framework that allows model owners to enhance efficiency and safety. It is implemented by leveraging the same vulnerability to dynamically terminate redundant or risky reasoning through external monitoring. Extensive experiments show that BoT poses a significant threat to reasoning reliability, while MoT provides a practical solution for preventing overthinking and jailbreaking. Our findings expose an inherent flaw in current LRM architectures and underscore the need for more robust reasoning systems in the future. Code is available at https://anonymous.4open.science/r/unthinking_vulnerability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the vulnerability of large reasoning models (LRMs) to “break of thought” attacks, where manipulating reasoning delimiters can cause the model to skip reasoning and produce direct answers. The authors design two types of Break of Thought (BoT) attacks: training-time backdoor injections during SFT or DPO, and inference-time adversarial suffixes, including both white-box and black-box variants. Experiments on MATH-500 and AIME-2024 show that these attacks achieve high success rates, reducing reasoning length and accuracy. To mitigate this issue, the paper proposes a “Monitor of Thought” (MoT) framework that detects when reasoning should stop or be shortened, improving computational efficiency and safety alignment. Overall, the work highlights the structural fragility of explicit reasoning processes in LRMs and calls for more robust reasoning architectures.

### Strengths
1. The figures are clean, well-labeled, and paired with informative captions.
2. The mathematical notation and formula presentations are clear and consistent.

### Weaknesses
1. The structure of this paper feels disjointed. Although Section 4 introduces the BoT unthinking attack and Section 5 presents the MoT defense, there is no subsequent connection between the two. Section 5 appears entirely independent of Section 4. Why didn’t the authors test how effectively MoT can defend against BoT attacks?

2. The paper mainly combines several existing techniques under the reasoning setting. For example, BoT is essentially a mix of backdoor and GCG attacks, while MoT resembles speculative decoding. Including so many technical tricks in one paper makes the overall contribution appear less innovative.

3. Many important baselines are missing. For BoT, similar attack methods already exist, yet the authors neither explain the differences from prior work nor include them as baselines [1-3]. Likewise, for MoT, numerous speculative decoding methods exist, but the authors fail to compare against or justify their novelty relative to those baselines [4-5].

4. The reference for line 370 is missing.

5. The reasoning tasks considered are too limited. For example, Table 1 only evaluates mathematical reasoning. How does the proposed method perform on other benchmarks?

> [1] Xiang Z, Jiang F, Xiong Z, et al. Badchain: Backdoor chain-of-thought prompting for large language models[J]. arXiv preprint arXiv:2401.12242, 2024.
>
> [2] Xu R, Qi Z, Xu W. Preemptive answer" attacks" on chain-of-thought reasoning[J]. arXiv preprint arXiv:2405.20902, 2024.
> 
> [3] Guo Z, Tourani R. Darkmind: Latent chain-of-thought backdoor in customized llms[J]. arXiv preprint arXiv:2501.18617, 2025.
> 
> [4] Miao X, Oliaro G, Zhang Z, et al. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification[C]//Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3. 2024: 932-949.
>
> [5] Zeng X, Shang Y, Chen J, et al. Root defence strategies: Ensuring safety of llm at the decoding level[J]. arXiv preprint arXiv:2410.06809, 2024.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper identifies a critical weakness in Large Reasoning Models, i.e., the Unthinking Vulnerability, where explicit reasoning can be bypassed by manipulating delimiter tokens. It studies this issue from two perspectives: Breaking of Thought (BoT), which disables reasoning via (1) a training-based backdoor attack and (2) a training-free adversarial suffix; and Monitoring of Thought (MoT), which improves safety by evaluating the risk of current reasoning. Experiments on multiple open LRMs show near-100% success in inducing unthinking behavior and gains in safety when MoT is applied.

### Strengths
- The paper is the first to explicitly formalize a delimiter-driven unthinking vulnerability in Large Reasoning Models (LRMs), providing a clear conceptual framework and empirical evidence for how simple token manipulations can disable structured reasoning.
- It examines both the vulnerability (via the Breaking of Thought attacks) and a simple solution (Monitoring of Thought), showing improvements in enhancing safety.
- Both BoT and MoT are lightweight, architecture-agnostic, and easy to reproduce.

### Weaknesses
- The core phenomenon, appending thought delimiters to suppress reasoning, is straightforward and somewhat expected, reflecting a token-level control misalignment rather than a deeper architectural flaw. The contribution lies more in the systematic evaluation than in theoretical novelty.

- The paper does not provide a rigorous explanation of why autoregressive likelihood dynamics lead the model to interpret delimiters as indicators of completed reasoning. A probabilistic or representational analysis would strengthen the technical depth.

- As noticed by the authors, the attacks assume visibility and controllability of thought delimiters, making them ineffective for closed-source reasoning models (e.g., OpenAI o3, Claude-4). As these are the most widely used systems, the real-world impact of the attack is limited.

- The experimental results are primarily quantitative without in-depth inspection of reasoning quality. Examples illustrating partial reasoning, semantic drift, or reasoning collapse would make the findings more interpretable.

- The study is limited to text-based LRMs. Given the rapid emergence of multimodal reasoning models, delimiter misuse could manifest differently across modalities (e.g., vision–language). The paper would benefit from a discussion or pilot evaluation in these settings.

- Several references contain placeholder symbols (“?”).

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a critical vulnerability in Large Reasoning Models (LRMs), termed Unthinking Vulnerability, whereby the reasoning process can be bypassed through manipulation of thought delimiter tokens. The authors propose two frameworks, including BoT, which focuses on suppressing reasoning by poisoning its training or inserting malicious prompt suffixes, and MoT, which monitors and truncates redundant or unsafe reasoning for improved efficiency and safety. Extensive experiments across multiple LRMs (e.g., DeepSeek-R1, QwQ) show near-100% attack success rates and substantial reasoning-length reduction, while MoT demonstrates large efficiency gains and safety improvements across StrongReject, HarmBench, and WildJailbreak benchmarks.

### Strengths
1. The "unthinking vulnerability" exposes a new failure mode in reasoning models distinct from traditional jailbreak or backdoor paradigms, and using the GCG method to disable reasoning is interesting.

2. Evaluations across multiple LRMs, tasks (AIME, MATH-500), and settings (white/black-box, training-based/inference-time) are comprehensive, with quantitative metrics such as ASR, RTC, and RPC. Figures and tables are well-structured and make the results easy to interpret.

### Weaknesses
1. Missing accuracy reporting for backdoor evaluation: The C-Acc results for BoT only evaluate whether the model induces the full thinking process on clean samples, but do not evaluate accuracy. In my understanding, in this backdoor scenario, verifying whether the model's performance remains intact should involve checking for any accuracy degradation on the clean inputs of AIME and MATH500, not merely confirming that the reasoning process still exists. The paper overlooks this aspect and does not report the corresponding results.

2. Insufficient comparison to prior efficient-reasoning works: While MoT is motivated by overthinking, it lacks direct quantitative comparison to existing monitoring or truncation-based methods (e.g., TokenSkip [1], ThoughtMani [2]). Comparison to these methods is necessary, especially for the efficiency scenario.

3. Missing references (e.g., Lines 147, 151, 369, 431, etc.) due to unresolved citations.

[1] Xia, H., Leong, C. T., Wang, W., Li, Y., & Li, W. (2025). Tokenskip: Controllable chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067.

[2] Liu, Y., Zheng, J., Sun, Z., Peng, Z., Dong, W., Sha, Z., ... & He, X. (2025). Thought manipulation: External thought can be efficient for large reasoning models. arXiv preprint arXiv:2504.13626.

### Questions
1. Regarding MoT, if an external monitor is introduced for supervision, should the token consumption include the monitor's token overhead? The prompting method described in the paper appears to have such a cost limitation.

2. What is the practical significance of this type of attack? In safety-related tasks, skipping the thinking may cause the model to produce harmful or unsafe responses. However, in mathematical tasks, the attack merely leads to degraded performance. Therefore, what are the real-world implications or potential harms of such attacks?

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
This paper looks into the “unthinking” behaviors of LRMs from two main perspectives. First, the authors investigate the unthinking behavior during inference-time by finding a prompt suffix that makes the model skip its reasoning process. The authors then study the unthinking behaviors under fine-tuning by either SFT using data with empty reasoning content or DPO that prefers trajectories with empty reasoning content. Based on the findings, the authors proposed two methods, one aimed at mitigating LRMs’ unthinking behaviors; the other utilized the unthinking behavior to restrain LRMs’ from overthinking.

### Strengths
- The authors conducted systematic analysis of LRMs’ reasoning behaviors under various settings. 
- The authors proposed a remedy to LRMs’ unthink behavior using data augmentation. Although the method is only effective in inference time, it is still a potential direction for future research. 
- Although it is unclear to me as to which model is used as the monitor, the MoT framework shows promising results in reasoning tasks (Table 5) where LRMs’ reasoning capabilities are preserved while reducing reasoning tokens.

### Weaknesses
- Many citations and references are missing. ? in the citation. 
- It is unclear what are the models used for evaluation. In the paper, the authors mentioned models such as “DeepSeek-R1-1.5B” while Deepseek-R1 only has one variant which has 685B parameters. I suspect that the authors are referring to the “Deepseek-R1-Distill-Qwen” models but fail to properly acknowledge the model names. 
- Most Open-sourced LRMs allow users to compress the thinking process, which is achieved by either adding a special token (e.g. /no_think in Qwen3) or empty CoT (e.g. <think>\n\n</think> mentioned in the paper). I see this as a purposely built trigger that allows users to allocate compute resources during inference instead of an unintended mechanism that makes LRMs vulnerable. 
- The authors claim that the monitor is a much lighter model compared to the base model. I wonder how could a much lighter (thus less capable) model being able to determine if the thinking trajectory is sufficient for answering the given question? This task demands strong reasoning capabilities as well and could be an even harder task than answering the original question. 
- The alignment result of MoT is unconvincing. LRMs could achieve perfect score if the monitor is an oracle that abstains the LRMs from answering any request (assuming all requests from the benchmarks are harmful).Therefore, the authors ought to plug the same monitor in other tasks to demonstrate that the MoT framework is effective in enforcing alignment while preserving models’ general capabilities such as reasoning. 
The monitor component of the MoT framework lacks generalizability as it needs to be tailored differently in the case of reasoning tasks versus alignment tasks.

### Questions
- The authors showed that compressing the thinking process leads to degraded performance in math reasoning tasks, which is a trivial and expected result. However, I don’t see the connection between degraded reasoning performance and models’ “vulnerableness”. A potential aspect is to show that models with compressed thinking processes are more prone to adversarial attacks. However, this type of experiment is missing (Note that this is different from the results presented in Section 5). 
- Which model is used as the monitor in the Monitoring of Thought framework? How is the monitor tailored to reasoning versus alignment tasks?

### Soundness
2

### Presentation
2

### Contribution
2
