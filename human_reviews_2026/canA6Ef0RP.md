# WebArbiter: A Generative Reasoning Process Reward Model for Web Agents

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Web agents hold great potential for automating complex computer tasks, yet their interactions involve long-horizon, sequential decision-making with irreversible actions. In such settings, outcome-based supervision is sparse and delayed, often rewarding incorrect trajectories and failing to support inference-time scaling. This motivates the use of Process Reward Models (WebPRMs) for web navigation, but existing approaches remain limited: scalar WebPRMs collapse progress into coarse, weakly grounded signals, while checklist-based WebPRMs rely on brittle template matching that fails under layout or semantic changes and often mislabels superficially correct actions as successful, providing little insight or interpretability. To address these challenges, we introduce WebArbiter, a reasoning-first, principle-inducing WebPRM that formulates reward modeling as text generation, producing structured justifications that conclude with a preference verdict and identify the action most conducive to task completion under the current context. Training follows a two-stage pipeline: reasoning distillation equips the model with coherent principle-guided reasoning, and reinforcement learning corrects teacher biases by directly aligning verdicts with correctness, enabling stronger generalization. To support systematic evaluation, we release WebPRMBench, a comprehensive benchmark spanning four diverse web environments with rich tasks and high-quality preference annotations. On WebPRMBench, WebArbiter-7B outperforms the strongest baseline, GPT-5, by 9.1 points. In reward-guided trajectory search on WebArena-Lite, it surpasses the best prior WebPRM by up to 7.2 points, underscoring its robustness and practical value in complex web tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
WebArbiter introduces a principle-grounded, generative process reward model designed to evaluate reasoning trajectories in web agents. The model first undergoes reasoning distillation via SFT to induce explicit, interpretable principles from a teacher LLM, and is subsequently optimized through GRPO using binary verifiable rewards. It employs LoRA adapters on top of Qwen2.5-Instruct (3B/7B) for efficient RL training. The authors also propose WEBPRMBench, a new benchmark covering multiple web-agent environments (Mind2Web, WebArena, AssistantBench, WorkArena).

### Strengths
1. The PRM is not just a classifier but a generative verifier. It produces explicit reasoning chains before outputting a verdict token. This design enables process-level supervision.
2. Rewards are automatically derived from web-environment logs, whether the predicted “correct” or “incorrect” verdict matches the actual success of an action. This eliminates extra human annotator costs.
3. WebArbiter achieves strong accuracy not just on Mind2Web (its training base) but also on unseen environments like AssistantBench and WorkArena, showing transfer capability. This indicates its principle-based reasoning generalizes beyond template matching.

### Weaknesses
1. The binary verifiable reward assumes one correct action per state, which is unrealistic. In tasks in web environment, several actions could succeed, but only one is labeled “Correct.” This introduces false negatives and reward noise,  a core limitation in deterministic PRM setups.
2. The evaluation omits new RL-based WebAgent baselines such as WebAgent-R1 [1] and WebSailor [2] that achieve strong results on WebArena and QA tasks. 
3. Using Qwen2.5-7B instead of Qwen3-8B (as in WebShepherd) introduces an architecture-generation gap. Qwen3 includes better instruction alignment, longer context, and multi-turn reasoning improvements. Reported performance gains in WebArbiter may partly reflect architectural differences, not purely training-method superiority.

[1] Wei, Zhepei, et al. "WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning." ICML 2025 Workshop on Computer Use Agents.

[2] Li, Kuan, et al. "WebSailor: Navigating Super-human Reasoning for Web Agent." arXiv preprint arXiv:2507.02592 (2025).

### Questions
1. Why was Qwen2.5 chosen instead of Qwen3-8B, given that baselines (WebShepherd) use newer backbones? 
Would results hold under an architecture-matched setting?

2. Does the WebArbiter generalize to challenging QA tasks that involve multi-round web search, like BrowseComp and HLE?

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
5

### Summary
This paper introduces **WebArbiter**, a novel Process Reward Model (WebPRM) designed to provide step-level supervision for web navigation agents. Unlike traditional outcome-based reward models that offer sparse and delayed feedback, or existing scalar and checklist-based WebPRMs that are brittle and lack interpretability, WebArbiter formulates reward modeling as a text generation task. It produces structured justifications that conclude with a preference verdict, identifying the action most conducive to task completion. The model is trained in two stages: first, reasoning distillation from a stronger teacher LLM to instill principle-guided reasoning; second, reinforcement learning to align verdicts with correctness and improve generalization. The authors also released **WEBPRMBENCH**, a comprehensive benchmark across four diverse web environments (Mind2Web, WebArena, AssistantBench, WorkArena).

### Strengths
- **Interpretability and Robustness:** WebArbiter’s reasoning-first approach provides auditable justifications, making it more interpretable and less reliant on superficial cues compared to scalar or checklist-based methods.
- **Effective Training Pipeline:** The two-stage training (reasoning distillation + RL) effectively combines the benefits of principled reasoning and correctness alignment.

### Weaknesses
- **Limited Multilingual Support:** The paper does not discuss multilingual capabilities, which limits its applicability in global web environments.
- **Potential for Reward Hacking:** While the RL stage aligns verdicts with correctness, the risk of reward hacking (e.g., over-optimizing for superficial patterns in justifications) is not addressed.
- **Narrow Benchmark Scope:** Although diverse, WEBPRMBENCH is limited to four environments, and its real-world coverage (e.g., multilingual or highly dynamic sites) is unclear.

### Questions
1. **Multilingual Generalization:** Does WebArbiter support non-English web environments? If not, are there plans to extend its reasoning and principle induction to multilingual contexts?
2. **Reward Hacking Mitigation:** How does the model avoid reward hacking during RL training? Are there mechanisms to ensure that the generated justifications genuinely reflect task progress rather than exploiting shortcuts?
3. **Scalability and Efficiency:** What are the inference latency and computational requirements of WebArbiter compared to simpler WebPRMs? How feasible is it for real-time web agent control?

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
3

### Summary
This paper proposes WebArbiter, a principle-guided reasoning process reward model (PRM) for web agents. It aims to improve the interpretability, reliability, and robustness of reward modeling for long-horizon web interaction tasks. Unlike scalar WebPRMs (which output coarse numeric rewards) or checklist-based generative WebPRMs (which rely on fragile templates), WebArbiter formulates reward modeling as structured text generation—producing reasoning chains that derive principles, compare candidate actions, and conclude with a preference verdict. The model is trained via a two-stage pipeline:
(1) Reasoning distillation from a stronger teacher LLM to teach principle-based justification, and
(2) Reinforcement learning with verifiable correctness rewards to align reasoning with factual outcomes.
Experiments on the newly released WEBPRMBENCH benchmark and WebArena-Lite show that WebArbiter achieves significant performance improvements over previous WebPRMs and even surpasses strong LLM-as-judge baselines like Gemini Flash and GPT-5.

### Strengths
1.	Solid empirical improvement: WebArbiter-7B achieves +10.9% BoN accuracy over Gemini Flash and +48% over WebShepherd, showing consistent advantages across diverse benchmarks.
	2.	Interpretability: The model generates explicit, auditable reasoning chains for each action decision.
	3.	Benchmark release: The introduction of WEBPRMBENCH enriches the evaluation ecosystem for process reward models in web agents.
	4.	Thorough ablation: The authors analyze the roles of principle guidance, reasoning distillation, and RL alignment in model performance.

### Weaknesses
1.	Limited conceptual novelty —
The paper’s design (reasoning-based PRM + two-stage pipeline) closely mirrors prior reasoning reward models like RM-R1, RewardReasoningModel, and Generative Verifier. The only difference lies in applying it to the web navigation domain. The overall framework lacks a strong theoretical or methodological innovation beyond scene adaptation.
2.	Missing key baselines —
Table 2 compares WebArbiter only against WebShepherd and a few LLM-as-judge models. However, it omits recent strong generative reward models (e.g., Rubric-RM, Generative Verifier from DeepSeek), which are more directly comparable. This omission weakens the evidence for claimed superiority.
3.	No evaluation of downstream supervision quality —
In Table 4, the paper reports WebArbiter’s gains in reward-guided trajectory search but does not compare against the downstream effects of other PRM or generative models listed in Table 2. Without showing whether the higher WEBPRMBENCH score translates into better supervised model performance, it’s hard to confirm the practical utility of the learned reward model.
4.	Lack of comparison with reasoning-capable LLMs —
Table 4 focuses on reward-guided trajectory search with GPT-4o-mini and GPT-4o, but omits comparisons with stronger reasoning models such as GPT-5 or DeepSeek-R1, which already exhibit built-in search and reflection abilities. These models might achieve comparable or better results without explicit PRM supervision, calling into question whether WebArbiter’s explicit reward modeling still offers clear advantages.
5.	Benchmark dependency —
WEBPRMBENCH is designed and used solely by the authors, which may limit reproducibility and objectivity. Its annotation criteria and preference construction may bias evaluation toward the proposed model’s reasoning style.

### Questions
1.	Why are DeepSeek-R1 and Rubrics-based reward models not included in the baseline comparison, given their conceptual proximity?
	2.	Have the authors compared downstream supervised agent performance (e.g., post-training agents) when using WebArbiter versus other PRMs?
	3.	How does WebArbiter’s inference latency or search overhead compare to baseline self-reflective reasoning LLMs like GPT-5?
	4.	Can the principle-guided reasoning process generalize to non-web reasoning tasks, or is it specifically tailored to the web environment?
	5.	How is the benchmark annotation quality ensured? Were multiple annotators or verifiers used to avoid label bias?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a novel approach to enhancing the performance of web agents in complex, multi-step tasks. It identifies the limitations of existing reward models, which often rely on coarse scoring or checklist-based methods that lack interpretability and robustness. To address these challenges, the authors introduce WebArbiter, a reasoning-first model that formulates reward modeling as a text generation task. The results demonstrate that WebArbiter significantly outperforms existing models, showcasing its robustness and practical value in real-world web tasks.

### Strengths
1. The paper introduces a fresh perspective on reward modeling by framing it as a text generation task. This innovative approach allows for the generation of structured justifications for actions, which is a significant departure from traditional scalar scoring and checklist-based methods. By integrating reasoning distillation and reinforcement learning, the authors create a model that not only enhances decision-making but also provides a more interpretable framework for understanding agent behavior.

2. By addressing the limitations of existing reward models and providing a more reliable framework for web agents, the paper has the potential to impact various applications in automated web tasks. The advancements presented in WebArbiter could lead to more effective and interpretable web agents, ultimately enhancing their utility in real-world scenarios.

### Weaknesses
1. The description of the WEBPRMBENCH construction is somewhat abstract. It would be helpful to include concrete examples or illustrations from the benchmark to make the process easier for readers to understand.

2. The main paper references an appendix (e.g., “Appendix XXX”), but no appendix is actually included.

3. The figures in the paper are low-resolution. Please replace them with higher-quality versions to improve readability.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
