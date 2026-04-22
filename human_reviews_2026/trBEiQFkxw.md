# Language Models That Think, Chat Better

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Reinforcement learning with verifiable rewards (RLVR) improves language model
reasoning by using rule-based rewards in verifiable domains such as mathematics and code. However, RLVR leads to limited generalization for open-ended
tasks—such as writing outline essays or making meal plans—where humans reason routinely. This paper shows that the RLVR paradigm is effective beyond verifiable domains, and introduces **RL** with **M**odel-rewarded **T**hinking (RLMT) for
general-purpose chat capabilities. Using diverse real-world prompts, RLMT requires LMs to generate long CoT reasoning before response, and optimizes them
with online RL against a preference-based reward model used in RLHF. Across
40 training runs on Llama-3.1-8B and Qwen-2.5-7B (both base and instruct) and
multiple optimization algorithms (DPO, PPO, and GRPO), RLMT consistently
outperforms standard RLHF pipelines. This includes substantial gains of 3–7
points on three chat benchmarks (AlpacaEval2, WildBench, and ArenaHardV2),
along with 1–3 point improvements on other tasks like creative writing and general knowledge. Our best 8B model surpasses GPT-4o in chat and creative writing and rivals Claude-3.7-Sonnet (Thinking). RLMT can also be applied directly
to base models without an SFT stage, akin to R1-Zero training (DeepSeek-AI,
2025). Remarkably, with only 7K prompts, Llama-3.1-8B base trained with our
RLMT recipe outperforms Llama-3.1-8B-Instruct post-trained with a complex
multi-staged pipeline with 25M+ examples. We close with qualitative and quantitative analyses of how trained models plan their responses. Our results rethink
the post-training pipeline and call upon future work to understand and employ
thinking more broadly

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the effectiveness of incorporating long chain-of-thought (CoT) reasoning in language model post-training for general-purpose chat capabilities. The authors introduce RLMT (Reinforcement Learning with Model-rewarded Thinking), which combines long CoT generation with online RL using preference-based reward models. Experiments are conducted across 40 training runs on Llama-3.1-8B and Qwen-2.5-7B using DPO, PPO, and GRPO algorithms. The results show consistent improvements of 3-7 points on chat benchmarks (AlpacaEval2, WildBench, ArenaHardV2) compared to standard RLHF pipelines. The best 8B model reportedly surpasses GPT-4o on chat and creative writing tasks.

### Strengths
1.  The paper conducts extensive experiments across multiple model families (Llama-3.1-8B and Qwen-2.5-7B), training algorithms (DPO, PPO, GRPO), and settings (warm-start vs. zero-shot). This provides robust emperical insights.
2. The paper addresses an important question about whether thinking/reasoning capabilities can improve performance on open-ended tasks beyond verifiable domains like mathematics and coding. The findings are useful for posttraining practioners.

### Weaknesses
1. The primary weakness is that the comparison conflates two factors: (1) the presence of thinking/CoT and (2) RLMT vs RLHF paradigm. The paper frames the comparison as "RLMT (with thinking) vs RLHF (without thinking)," but this is not a correct framing. In RLHF, one can still incorporate thinking by having models generate CoT traces and then extracting only the final output for the reward model to evaluate. The current setup makes it difficult to isolate whether improvements come from thinking itself or from the specific RLMT training paradigm. A more appropriate comparison would be: (a) RLHF with thinking vs RLHF without thinking, and (b) RLMT with thinking vs RLHF with thinking.
2. The paper is primarily empirical and does not have much novel research value (for example, an industry lab can quickly get such insights by sweeping across different methods). Also, for RLMT, there are many relevant work that do RL on unverifiable tasks, such as [1] [2]. These work are not discussed in the paper.

[1] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains, Gunjal et al., https://arxiv.org/abs/2507.17746
[2] Checklists Are Better Than Reward Models For Aligning Language Models, Viswanathan et al., https://arxiv.org/abs/2507.18624

### Questions
1. Have you considered trying out RLHF with thinking (where thinking traces are generated but only final outputs are evaluated by the reward model)? This would help isolate the contribution of thinking itself versus the RLMT training paradigm.

### Soundness
2

### Presentation
2

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
This paper introduces RLMT, a post-training paradigm for LLMs that combines CoT reasoning with model-based reward optimization, extending reinforcement learning with verifiable rewards (RLVR) into general-purpose chat and open-ended tasks. RLMT requires models to produce detailed thinking traces before responses, which are then optimized via online RL against a reward model trained on human preferences, similar to RLHF but with forced long-form CoT. Experiments involve the Llama-3.1-8B and Qwen-2.5-7B families, both base and instruction models, across DPO, PPO, and GRPO. From the reported results, RLMT substantially and consistently outperforms standard RLHF on chat, creative writing, and general knowledge benchmarks. Analysis covers model behaviors, prompt/reward model ablations, and qualitative planning traits.

### Strengths
- RL with unverifiable domains is a timely topic. This addresses a longstanding limitation in generalizing explicit reasoning to open-ended tasks.
- The analysis is quite insightful to read. 
- The paper is carefully written and well organized.

### Weaknesses
- The contributions seem a bit limited. The key difference that the paper claimed is enabling RL to work on unverifiable domains. This is achieved through substituting a verifiable reward function in RLVR with a reward model. However, the key, in my opinion, becomes how to obtain a strong reward model such that any policy model can improve its chat performance when RL with the reward model.
- The conclusion is a bit too intuitive. It is not surprising that long CoT benefits chat performance.
- The long CoT in chat might bring extra computation overhead. I understand that the chat domain is an example of an unvarifiable domain, but usually the general domain chats demands fast response and low latency. Always conducting the long thinking might not actually be what people want.
- Experiments about scaling effects should be a substantial part of the paper, but are completely missing. How does the performance change when scaling up the data size, model size, and inference budgets (token length)?
- Some observations are not properly interpreted. For example, why does long CoT help creative writing? Where does the creativity come from?

### Questions
- The proposed method relies heavily on the scores produced by reward model. Is it robust to reward model bias or poor reward calibration? Have the authors measured or observed any reward gaming, reward hacking (length bias, verbosity), or mismatches between human preference and model reward during RLMT?
- How does the performance change when scaling up the data size, model size, and inference budgets (token length)?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RLMT: performing RL on an LLM using a learned reward model, with thinking. In experiments, they perform RLMT with a modest set of prompts and compare this to baselines that simply use STF or RL without thinking as well as fixed open and closed models. They demonstrate good performance on a range of benchmarks (especially chat benchmarks) relative to both baselines and closed models. They perform a considerable set of ablations, showing good performance across choices of RL algorithm but demonstrating the importance of a strong reward model. They provide a qualitative analysis of the differences due to RLMT as well as analysis of CoT length.

### Strengths
Originality. To my knowledge, despite this domain receiving a great deal of attention, this precise technique, while simple, hasn’t been showcased before, and it makes good use of some newer benchmarks useful for measuring performance on chat and creative writing.

Quality. The experimentation — both the headline comparisons as well as the ablations — are fairly extensive. The headline comparisons are sensible, and the ablations make some helpful disambiguations about what’s working here.

Clarity. This paper is very well-written very understandable.

Significance. This domain is of great interest to many and has received quite a lot of attention. The main idea here is very natural and worth having results on, and the secondary results (e.g. the No SFT runs) are a natural extension of DeepSeek results and are very much worth highlighting.

### Weaknesses
I’m torn on how to think about the originality here. It’s a very simple extension of pretty well-understood ideas. It’s quite close to Wu et al. 2025a (as you cite), with the adoption of some more recently adopted techniques in online RL ported to LLMs as well as updated benchmark. It’d be helpful to tease apart precisely what makes this new relative to work like that in methods.

The results feel a bit overstated in parts. In particular, looking at the warm-started models (RLMT seems to work more decisively in the no sft setting), results look strongest on the chat tasks, which are most closely aligned with the prompts used during training. The paper claims strong performance on creative writing, but that only really seems to hold for instruct models. On other benchmarks, RLMT is pretty clearly doing worse than baselines. I’d suggest making this tradeoff clearer in results.

### Questions
Already articulated in weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Reinforcement Learning with Model-rewarded Thinking (RLMT) which trains LLMs to generate long CoT reasoning before final answers, using online RL algorithms such as GRPO. Compared to RLVR relying on rule-based rewards tied to ground-truth answers, RLMT only requires prompts and uses reward models trained on human preference data over diverse prompts, as in RLHF, to evaluate responses. Previously RLVR is limited to structured domains like math and code and RLMT extends RL to open-ended reasoning tasks like open-ended chat.

### Strengths
1. The paper introduces Reinforcement Learning with Model-rewarded Thinking (RLMT)  - a new way to incorporate reasoning in LLMs. It focuses on domains other than math, code and science and focuses on reasoning needed for creative writing and chat.
2. The paper has conducted comprehensive experiments across different RL algorithms like GRPO, DPO and PPO. 
3. The paper goes the additional mile of qualitative analysis of model behavior under SFT and RLMT, as well as ablations with various SFT data sources and reward models.

### Weaknesses
1. It would be interesting to see what happens when RLMT training is mixed with RLHF/RLVR. Would you get the best generalized model which does well on all tasks - math, science and code as well as creative writing etc?

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3
