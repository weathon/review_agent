# IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling

- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Recent advances in deep-research agents have shown promise for autonomous knowledge construction through dynamic reasoning over external sources. However, existing approaches rely on a mono-contextual paradigm that accumulates all information in a single, expanding context window, leading to context suffocation and noise contamination that limit their effectiveness on long-horizon tasks. We introduce \textbf{IterResearch}, a novel iterative deep-research paradigm that revisits long-horizon research through the lens of Interaction Scaling. Instead of relying on linear context accumulation, we adopt an MDP-inspired architecture with strategic workspace reconstruction. By maintaining an evolving report as memory and periodically synthesizing insights, our approach preserves consistent reasoning capacity across arbitrary exploration depths. To effectively train this paradigm, we employ Efficiency-Aware Policy Optimization (EAPO), a training strategy that adapts geometric reward discounting to incentivize efficient exploration and utilizes adaptive downsampling for stable distributed training. Extensive experiments demonstrate that IterResearch achieves substantial improvements over existing open-source agents with average +14.5pp across six benchmarks and narrows the gap with frontier proprietary systems. Remarkably, our paradigm exhibits unprecedented interaction scaling, extending to 2048 interactions with dramatic performance gains (from 3.5\% to 42.5\%), and serves as an effective prompting strategy, improving frontier models by up to 19.2pp over ReAct on long-horizon tasks. These findings position IterResearch as a versatile solution for long-horizon reasoning, effective both as a trained agent and as a prompting paradigm for frontier models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes IterResearch, a new method for deep research. Current approaches keep expanding the context in the think->act phases which can limit the space available for a response. IterResearch is inspired from MDPs and compresses the current context by summarizing it into a report. Thus, the new context consists of the original question, the report, and any actions to take.

The authors also introduce EAPO - Efficiency Aware Policy Optimization by using discounted rewards for binary, sparse deep research settings where the reward is only given on the final step.

Finally, IterResearch can also be used as a prompting paradigm directly so that no training etc is needed and can work with current models.

The authors then conduct an empirical evaluation with different baselines.

### Strengths
1. The paper is clear and well-written

2. The overall idea is intuitive

3. Results demonstrate significant improvement (although there are some open questions here)

### Weaknesses
Overall, the results clearly seem to advance the SOTA so Im hoping for more clarifications to my identified weaknesses. Im happy to engage in discussion and revise my score if my concerns are adequately addressed.

1. I think one of the major claims of the paper is the Markovian State Reconstruction. There is not really any guarantee that this report is Markovian in any sense. The Markovian State suggests that the future state is independent of the history given the current state. 

If that were the case, then definitely the transition function with the input history or with the report is not going to change the outcome. So the paper title feels a bit bold. It is more apt to call it summarization but calling it Markovian State Reconstruction is a bit too much. I think experiments would be needed to back your claims.

The idea is intuitive to improve the performance due to limitations on attention windows/distributions but to claim that summarization of the past context and making it a report is making it Markovian seems a bit overblown IMO.

2. I tihnk one problem with the results is that the prompts are not provided for the baselines in the prompt-based approach. The prompt used in appendix F.2 is quite exhaustive and Im not sure if some instructions that are agnostic to the approach were used in the baselines.  For example, the entire section of ## Working Principles. This might be an issue with clarity in the paper.

3. I think the claim of unbounded interactions is overstated. Even summarization has limits. Are the summaries never going to grow? How does the LLM know what information is important. There could be some tool response that is only going to be needed several steps later. This is often the case in partial-order planning. One issue here is that once that information is not included in the summary, it is lost forever. At least in the mono-context case it is possible to be "re-"discovered by a later step based on the generated output and attended to. I am a bit confused on how the LLM knows to generate a good report so that such issues do not occur.

3. I dont think EAPO is really novel. RL algorithms use discounting by definition and this does not seem like a new contribution. Please justify why you think EAPO is novel?

### Questions
Please see weaknesses

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
5

### Summary
The paper proposes IterResearch, a long-horizon reasoning framework where an agent maintains a Markovian workspace instead of accumulating the full dialogue history. The evolving report acts as compressed memory, keeping context size nominally constant. Training uses Efficiency-Aware Policy Optimization (EAPO) with geometrically discounted rewards to promote concise, successful trajectories.

IterResearch achieves notable gains (+14.5 pp on six benchmarks) and reportedly scales from 32 to 2048 reasoning steps, outperforming prior open-source systems and even rivaling proprietary ones. However, the "constant-context" claim is partly theoretical - since the report must still grow or overwrite information, true unbounded reasoning is unproven. The paper also lacks experiments confirming recovery from early false summaries or testing smaller training horizons. Comparisons rely on older baselines (GPT-4.1, o4-mini), and it remains unclear whether gains stem from the paradigm itself or its prompting format.

Overall, IterResearch is an elegant compression-based reformulation of long-horizon reasoning, but its claimed scalability and memory efficiency need stronger empirical validation.

### Strengths
1. Clear, principled formulation. The Markovian workspace reconstruction is simple and well-motivated, giving constant context size and avoiding "context suffocation"/"noise contamination". The formal transition s_{t+1}=T(s_t,d_t,E(a_t)) is explicit.  

2. Strong empirical coverage. Six benchmarks with diverse characteristics; competitive vs. proprietary systems on several, e.g., surpassing OpenAI DeepResearch on HLE and BrowseComp-zh. 

3. General usefulness as a prompting recipe. IterResearch prompting outperforms ReAct with frontier models (o3, DeepSeek-V3.1), suggesting paradigm-level value beyond one implementation.

### Weaknesses
1. Markovian constant context is a conceptual simplification rather than a true breakthrough.
The claimed $O(1)$ workspace complexity is only formal when the evolving report $|M|$ is treated as fixed. In practice, $|M|$ grows with the number of synthesized summaries, and each tool response $|TR|$ can vary widely across steps. Therefore, the overall process still depends on in memory and computational complexity, merely folded into a different structure. This makes the "constant" context claim more of a heuristic compression scheme than a provably unbounded process. Once the report becomes saturated, the system risks discarding fundamental early-stage information, effectively re-introducing the limitations it sought to remove.

2. Potential inefficiency from report initialization.
If the evolving report $|M|$ is kept constant, the first few interactions still include a large workspace prompt even when the agent has accumulated little information. This may yield unnecessary compute overhead at the start of every episode, contradicting the claimed "efficiency-aware" principle.

3. Questionable extrapolation to long horizons.
The claim of scaling from $T_{max}=32$ (training) to $2048$ interactions (inference)rests on the assumption that the agent can reconstruct forgotten information through new searches. Yet, no evidence is given that this process retrieves previously lost or overwritten insights, as opposed to repeatedly circling around similar queries or rediscovering the same partial results. Without qualitative trajectory inspection, it is unclear whether high-turn runs reflect genuine continued reasoning or redundant loops.

4. No measurements for smaller training horizons.
While Section 4.4 evaluates scaling at inference, there are no experiments showing performance under smaller $T_{max}$ during training. Thus, it is impossible to assess whether longer training horizons produce better generalization or if the model’s extrapolation is accidental.

5. Outdated or inconsistent baseline comparisons.
The benchmark comparison includes GPT-4.1 and o4-mini but omits GPT-5-series or contemporary baselines. This limits the credibility of claims about competitiveness "with frontier proprietary systems".

6. Lack of ablation on report-based prompting alone.
The paper does not include a variant where the same prompt structure is given to a baseline model without the workspace-reconstruction mechanism (i.e., the same input text but no internal state machine). It remains unclear how much of the improvement arises from the report-prompt structure rather than the Markovian update itself.

7. Novelty of EAPO may feel incremental. Geometric discounting for terminal rewards is standard MDP practice; while appropriate here, the RL contribution beyond applying discounting + GSPO + downsampling reads as engineering rather than fundamentally new learning theory. Ablations isolate length reduction but not a deeper analysis of policy changes.

### Questions
1. What happens when $T_{max}$ during training is reduced to, say, $8$ or $16$? Does extrapolation still hold, or does generalization degrade proportionally?

2. How can we confirm that high-turn (>100) trajectories generate genuinely new insights rather than re-searching similar content? Have the authors analyzed overlap of retrieved URLs or paraphrased content between early and late rounds?

3. How do accuracy and average step count vary with $\gamma$ (e.g., $0.99$/$0.997$/$0.999$)? Is the $5.7%$ length reduction robust, and where does performance begin to trade off?

4. If the evolving report contains an early false hypothesis, can the agent recover? Have the authors measured recovery rate with stress tests where earlier synthesized content is wrong/misleading?

### Soundness
2

### Presentation
3

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
This paper introduces IterResearch, an iterative deep-research paradigm that overcomes the limitations of mono-contextual approaches in long-horizon reasoning. It maintains an evolving report as memory and periodically synthesizes insights. The system was trained on an aggregated QA dataset, using both supervised fine-tuning and a reinforcement learning phase. Experiments demonstrate substantial improvements over existing open-source agents and notable gains when used as a prompting strategy for frontier models.

### Strengths
- The problem is well-motivated: mono-contextual approaches accumulate information in a single context window, causing noise and limiting effectiveness on long-horizon tasks.
- The presented approach significantly outperforms current state-of-the-art models in solution quality on selected benchmarks.

### Weaknesses
W1: The novelty of the approach is somewhat limited, as many of its core ideas, such as using discounted rewards and iterative refinement of trajectories, are already widely used in classical reinforcement learning.

W2: The MDP formulation appears incomplete. It omits a reward function (e.g., a verification signal), and the transition function is likely not deterministic, since tool outputs can vary over time (e.g., search results). The decision space is unconventional: including Think and Report as part of the action reflects internal computation rather than true environmental actions. Additionally, the “reconstructed workspace” manages context by discarding some history, but this alters the state definition across timesteps and may violate the Markov property.

W3: While I understand the intuition behind why discounted reward shaping works, the analogy with RL and the discount factor appears somewhat misleading. First, if we treat this reward as in a typical RL environment, it becomes biased: two similar trajectory prefixes may receive different rewards depending on the trajectory length (the length of the suffixes). Consequently, this mechanism not only amplifies the signal at the end of successful trajectories but also implicitly encourages shorter trajectories. I think this modification is more akin to how value is computed in RL, where there exist well-established alternatives, such as GAE. 

W4: The approach was trained exclusively with Qwen-3-30B-A3B, which may limit its generalization to other models. Experiments with models used without fine-tuning only partially address this concern.

W5: Provide a stronger comparison with AlphaEvolve [1] (and similar systems), as the three components used, main objective question (task), evolving report (previous high-level ideas), and immediate context (code), closely mirror its design (in the case of AlphaEvolve, for coding tasks). The use of a reward function applied at the end of generation similarly provides the necessary learning signal for optimization. Although Alpha-Evolve itself is closed-source, several open-source implementations exist (e.g., OpenEvolve). I recommend that the authors at least position their work in relation to these approaches, and ideally include a direct empirical comparison.

W6: The paper assumes that the LLM can reliably summarize, filter noise, and preserve crucial information in each round. In practice, this compression step could lead to the loss of essential context, especially under noisy tool outputs, and may worsen compounding summarization errors.

[1] Novikov A, Vũ N, Eisenberger M, Dupont E, Huang PS, Wagner AZ, Shirobokov S, Kozlovskii B, Ruiz FJ, Mehrabian A, Kumar MP. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131. 2025 Jun 16.

**Minor suggestions**:
- I recommend that the authors provide confidence intervals in the table, and include the token budget used for each model.
- It is recommended to include in the description of Table 1 an explanation of why proprietary models were not tested on some benchmarks, to make the table self-contained.

### Questions
Q1: Could the authors clarify how the baselines were adopted for the studied benchmarks? Were the open-source agents also fine-tuned on the same data as IterResearch?

Q2: Which datasets were used during the RL phase? Were there tasks in the training set similar to those on which the approach was evaluated, or was it still only the QA Collection? How does the test set intersect with the QA Collection?

Q3: What is the token budget and inference cost for the baseline LLMs? Which budget is needed to reproduce the results.

### Soundness
2

### Presentation
3

### Contribution
2
