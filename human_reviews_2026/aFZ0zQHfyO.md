# Think Smart, Not Hard: Difficulty Adaptive Reasoning for Large Audio Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6, 4

## Abstract
Large Audio Language Models (LALMs), powered by the chain-of-thought (CoT) paradigm, have shown remarkable reasoning capabilities. Intuitively, different problems often require varying depths of reasoning. While some methods can determine whether to reason for a given problem, they typically lack a fine-grained mechanism to modulate how much to reason. This often results in a ``one-size-fits-all'' reasoning depth, which generates redundant overthinking for simple questions while failing to allocate sufficient thought to complex ones. In this paper, we conduct an in-depth analysis of LALMs and find that an effective and efficient LALM should reason smartly by adapting its reasoning depth to the problem's complexity. To achieve this, we propose a difficulty-adaptive reasoning method for LALMs. Specifically, we propose a reward function that dynamically links reasoning length to the model's perceived problem difficulty. This reward encourages shorter, concise reasoning for easy tasks and more elaborate, in-depth reasoning for complex ones. Extensive experiments demonstrate that our method is both effective and efficient, simultaneously improving task performance and significantly reducing the average reasoning length. Further analysis on reasoning structure paradigm offers valuable insights for future work.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Think Smart, Not Hard, a reinforcement-learning framework that enables difficulty-adaptive reasoning for Large Audio Language Models (LALMs). The key insight is that current LALMs, trained via SFT or GRPO, apply uniform reasoning depth regardless of question difficulty—causing redundant “overthinking” for easy tasks and shallow reasoning for complex ones.
To address this, the authors propose two difficulty-adaptive, length-based reward functions: (1) Group Ratio Difficulty Reward (GRDR), which estimates difficulty from the correctness ratio of sampled rollouts, and (2) Group Audio-Attention Difficulty Reward (GA2DR), which derives difficulty from attention-entropy over audio tokens. These rewards dynamically link reasoning length to model-perceived difficulty, encouraging concise reasoning for simple inputs and deeper reasoning for challenging ones. Experiments demonstrate consistent gains over standard GRPO and truncation-based baselines, achieving shorter reasoning traces without sacrificing accuracy.

### Strengths
(1) The paper identifies an important inefficiency in current LALMs by showing that they apply uniform reasoning depth across different difficulty levels. It provides a clear motivation and strong empirical support for introducing difficulty-adaptive reasoning.

(2) The proposed GRDR and GA2DR rewards are simple and effective, allowing adaptive reasoning without external supervision. 

(3) Experiments on MMAU with supporting AirBench results demonstrate consistent gains over GRPO and truncation baselines.

### Weaknesses
(1) The paper introduces exponential difficulty–length scaling within GRPO without analyzing its impact on policy-gradient variance or monotonic-improvement conditions. In the absence of bounded-gradient or convergence guarantees, objective stability remains unverified.

(2) GA2DR uses batch-normalized attention entropy as a continuous difficulty signal (Eq. 5), but provides no calibration or correlation against human difficulty labels. Despite normalization, the signal may still conflate input noise or sequence length with true task complexity.

(3) Because GRDR derives difficulty from on-policy rollout correctness, the reward baseline drifts as training evolves, creating a feedback loop that can distort credit assignment and destabilize late-stage optimization.

(4) The Cold-Start variant shows degraded outcomes, yet the paper omits optimization diagnostics (e.g., KL, gradient-norm, reward-variance trajectories), leaving claims of adaptive-reward stability anecdotal.

(5) Reasoning length is used as a proxy for efficiency, but no inference-time, FLOP, or latency measurements are reported, so computational gains are not established beyond token-level proxies.

(6) Model-perspective difficulty labels are produced by the same model family used for training (Qwen variants among the labelers and learners), risking family-bias. No cross-family labeling or out-of-domain validation is provided to demonstrate generalization.

(7) The paper includes a single-task qualitative example highlighting redundancy vs. core reasoning, but lacks a systematic failure-mode analysis across tasks/difficulties, so it remains unclear whether shorter traces consistently reflect better reasoning rather than premature truncation.

### Questions
(1) How stable are the GRDR and GA2DR difficulty signals throughout training? Please provide curves of γ variance and reward distribution across epochs to verify that the adaptive scaling converges rather than oscillates or collapses.

(2) GA2DR assumes that attention entropy reflects reasoning difficulty, yet this metric could be sensitive to batch normalization, padding, or input noise. Have you examined its noise robustness or its correlation with human-annotated or dataset-defined difficulty levels on MMAU?

(3) Could the authors quantify the relationship between reasoning length and task accuracy, for instance by plotting accuracy/reward versus token length? 

(4) Does the proposed difficulty-adaptive training generalize beyond audio–text reasoning to text-only or audio–visual tasks? 

(5) Since the model-perspective difficulty labels are produced by the same family of backbones (Qwen2-Audio, Qwen2.5-Omni, Kimi-Audio, Gemini2.5-Pro) used for training, could there be self-consistency or family bias? Have you tested cross-model validation using an unseen architecture to confirm that the learned adaptivity generalizes?

(6) What is the per-batch computational overhead of calculating attention-entropy–based difficulty, and can it scale to larger multimodal corpora or longer sequences without becoming a bottleneck?

(7) The method emphasizes shorter reasoning chains, but is there evidence that reasoning quality remains unchanged? Have you conducted any human or LLM-as-a-judge evaluations of coherence, factual correctness, or interpretability to ensure that brevity does not harm reasoning fidelity?

(8) The Cold-Start SFT→GRPO variant exhibits degraded performance, but no diagnostics are shown. Could the authors provide gradient-norm, reward-variance, or KL divergence curves to clarify why Cold-Start fails to stabilize and whether this reflects optimization instability or data mismatch?

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
3

### Summary
This paper tries to address the "one-size-fits-all" issue in LALMs reasoning, analyzing performance differences between SFT/GRPO and the impact of explicit/implicit prompts. It proposes two adaptive reward functions: GRDR (discrete difficulty) and GA²DR (continuous difficulty). Experiments on the MMAU benchmark verify that the method improves performance while shortening reasoning length, and also refines an ideal reasoning paradigm and model training recommendations, providing a solution for LALMs reasoning optimization.

### Strengths
- It identifies the pain point of difficulty adaptation in LALMs reasoning, fills gaps in SFT/GRPO comparison and prompt effect analysis.
- The method aligns with audio modal characteristics; the two difficulty definitions cover different scenarios, featuring flexible design.
- Experiments use multiple baselines and multi-dimensional verification, and introduce model-perspective difficulty labeling, ensuring reliable results.

### Weaknesses
- GRDR uses a fixed group size G=8 without verifying other values, and GA²DR lacks basis for layer selection, leading to insufficient robustness in difficulty definition.
- The hyperparameter values of the reward function have no basis and lack sensitivity analysis, affecting reproducibility.
- There are inconsistencies in formula and figure formatting. such as no periods or commas are added at the end of formulas. Figure 1 lacks a legend, which makes it impossible to identify the meaning of different curves, colors, or markers in the figure.

### Questions
1. (Line 192) What is the basis for choosing G=8 in GRDR? How will performance change if G is set to 4 or 16?  
2. (Line 211) Why does GA²DR use the last layer to calculate attention entropy? Will switching to an intermediate layer cause significant differences?  
3. (Line 246) What is the basis for setting $k_{easy}$ and $k_{hard}$ in the reward function? How will performance change after adjustment?

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
5

### Summary
The paper studies how Large Audio‑Language Models (LALMs) should modulate how much they reason, rather than simply deciding whether to reason. Building on analyses that compare SFT vs. GRPO and explicit vs. implicit CoT prompting, the authors propose difficulty‑adaptive reward shaping that links reasoning length to problem difficulty. They introduce two difficulty estimators computed from the model’s perspective: GRDR (Group Ratio Difficulty Reward), which labels a question as easy/medium/hard based on the share of correct rollouts in a group and GA$^2$DR (Group Audio‑Attention Difficulty Reward), which maps the entropy of last‑token attention over audio tokens to a continuous difficulty $\gamma \in [0,1]$. A negative‑exponential length reward promotes short, concise CoT on easy items and longer, exploratory CoT on hard ones. Experiments on MMAU‑test‑mini with Qwen2‑Audio‑7B and Qwen2.5‑Omni‑7B (LoRA + GRPO, explicit CoT prompt) show modest average accuracy gains over GRPO+TR and notably bigger gains on hard subsets.

### Strengths
- Paper is clear and easily understandable
- Meaningful improvements on hard items with small overall gain
- Good ablation on explicit CoT helps on hard items.

### Weaknesses
- The paper has limited evaluation scope. Results focus on MMAU‑test‑mini with single‑run ACC. There are no standard errors or full‑benchmark experiments.
- The paper doesn't report compute/training cost: GRDR needs grouped rollouts; GA$^2$DR requires attention extraction. The paper does not quantify added RL cost vs. GRPO/TR.
- The hyperparameters are not ablated. For instance, among slopes ($k_{easy}$ and $k_{hard}$), Group sizer $G$ and $l_{min}$; only $l_{min}$ is ablated.
- GA$^2$DR hinges on last‑token attention over audio tokens, however, the scheme is not tested across across different layers/heads. Why the last year only?
-  Fig. 2 shows log‑length trends but not absolute token reductions on the full test. The log trend hides effective sizes and makes it hard to tell how much shorter the reasoning actually got.
- The paper measures all lengths with the Qwen2-Audio tokenizer, even when the model being trained/evaluated is Qwen2.5-Omni. This might lead to discrepancies if two models are using different tokenizers.

### Questions
- What is the training overhead of GRDR/GA$^2$DR relative to GRPO and TR (same batch size, same G)?
- For GA$^2$DR, does using different layers/heads (or pre‑softmax attention) change the difficulty estimates and outcomes?
- Do results hold on the full MMAU or other audio‑reasoning datasets?

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
4

### Summary
This paper tackles the inefficiency of LALMs using a uniform reasoning depth for all tasks. The authors propose a "difficulty-adaptive" method, using a new RL reward function that links reasoning length to the model's perceived difficulty. This encourages concise reasoning for simple tasks and in-depth reasoning for complex ones, ultimately improving performance while significantly shortening the reasoning length.

### Strengths
1. It addresses the critical and practical problem of high computational cost and inefficiency in CoT reasoning, achieving the important goal of "better performance with less computation."
2. The core idea of using "model-perspective difficulty" (instead of static human labels) is intelligent. The proposed metrics, like $GA^{2}DR$ (based on audio attention entropy), are novel and well-justified.
3. The experimental results are strong. The method not only improves accuracy on hard problems but also significantly reduces the average reasoning length, effectively validating the "think smart" concept.

### Weaknesses
1. The experimental validation relies almost entirely on the MMAU-test-mini benchmark. While the analysis on this benchmark is deep (e.g., broken down by difficulty), this raises questions about generalizability.
2. The proposed difficulty metrics (GRDR and $GA^{2}DR$) appear computationally expensive during training. GRDR requires $G=8$ full rollouts per sample just to determine its difficulty level1. $GA^{2}DR$ requires extra computation for attention entropy and relies on batch normalization 2222. This overhead could make the already complex RL training process significantly slower and more costly, a trade-off that is not discussed in detail.

### Questions
1. Regarding the GRDR metric , how sensitive are the final performance and efficiency gains to the specific thresholds you chose (e.g., 3 and 6)? Would relaxing the definition of "easy" from 6 correct rollouts to 5, for example, significantly alter the results?
2. In Section 4.3, you analyze and identify an ideal reasoning paradigm: extracting conditions (captioning), followed by analysis, and then the answer. However, your reward function (Equation 6) primarily optimizes for length and correctness. How does this reward mechanism ensure that as the model shortens its reasoning, it preserves this critical logical structure rather than simply deleting random thought steps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a difficulty-adaptive reasoning framework for LALMs, introducing two self-perspective difficulty estimators and length-negative-exponential rewards (GRDR and GA²DR). The goal is to encourage concise reasoning for easy questions and deeper reasoning for hard ones, thereby improving efficiency without sacrificing accuracy. The authors conduct experiments on the MMAU benchmark, demonstrating that their approach reduces reasoning length while maintaining strong performance.

### Strengths
The motivation is compelling, addressing a critical limitation of current LALMs—lack of adaptive reasoning. The proposed rewards (GRDR and GA²DR) show practical benefits, significantly reducing reasoning length on the MMAU-test-mini benchmark while preserving accuracy(From Table 4). The framework’s ability to dynamically adjust reasoning depth based on question difficulty is a promising direction for improving LALM efficiency.

### Weaknesses
1、Contribution 1 from authors only reports aggregate accuracy on MMAU-Test-Mini split by difficulty.  The authors should perform finer-grained statistical analyses to validate the claim that “models under-/over-reason on different difficulties”.
2、Contribution 2 from authors introduces GRDR and GA²DR, but their relationship is unclear.  GA²DR appears to be a stand-alone criterion; its connection with GRDR is never formally discussed.  Besides, GRDR itself seems questionable: Table 2 shows that models perform best on medium-difficulty questions, while easy and hard accuracies are almost equal.  Under GRDR’s definition, should the original three difficulty buckets be re-labeled?  If so, how does the new labeling relate to the original one, and why is it more reasonable?
3、How are the “groups” in GRDR constructed? 
4、What is the theoretical or empirical link between GRDR and GA²DR?  Table 3 treats them as independent ablations.  If they are orthogonal, why must both be proposed?  Moreover, the performance gaps in Table 3 are within one standard deviation, so improvements may simply come from random-seed variation rather than the rewards themselves.
5、Experiments are restricted to MMAU-test-mini. No results on other audio-reasoning benchmarks or cross-domain transfer are provided, limiting generalizability.
6、Line 361 claims “Specifically, it achieves short reasoning for simple questions and long reasoning for difficult ones, while overall significantly reducing reasoning length.”  But Figure 2 only shows that GRDR/GA²DR shorten the average length; it gives no evidence that the model produces longer chains for hard questions than for easy ones.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
