# Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Mobile Agents can autonomously execute user instructions, which requires hybrid-capabilities reasoning, including screen summary, subtask planning, action decision and action function. However, existing agents struggle to achieve both decoupled enhancement and balanced integration of these capabilities. While Mixture-of-Experts (MoE) supports capability decoupling, the input-oriented activation prevents the selection of expert aligning with the reasoning stage. 
To address these challenges, we propose Channel-of-Mobile-Experts (CoME), a novel agent architecture consisting of four distinct experts, each aligned with a specific reasoning stage, CoME activates the corresponding expert to generate output tokens in each reasoning stage via output-oriented activation. 
To empower CoME with hybrid-capabilities reasoning, we introduce a progressive training strategy:
Expert-FT enables decoupling and enhancement of different experts' capability; Router-FT aligns expert activation with the different reasoning stage; CoT-FT facilitates seamless collaboration and balanced optimization across multiple capabilities.
To mitigate error propagation in hybrid-capabilities reasoning, we propose InfoGain-Driven DPO (Info-DPO), which uses information gain to evaluate the contribution of each intermediate step, thereby guiding CoME toward more informative reasoning.
Comprehensive experiments show that CoME outperforms dense mobile agents and MoE methods on both AITZ and AMEX datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Channel-of-Mobile-Experts, an agent architecture that assigns different experts to distinct reasoning stages and uses output-oriented activation instead of conventional MoE routing. A progressive training pipeline and an InfoGain-driven DPO strategy are proposed to enhance capability specialization and reduce error propagation. Experiments on AITZ and AMEX show that CoME outperforms dense and sparse baselines in both action type and action match accuracy, with supporting ablation studies validating the design choices.

### Strengths
1. The paper introduces a well-motivated architectural design that addresses limitations of conventional MoE routing in multi-stage reasoning.The progressive training strategy is clearly structured and each stage is supported by ablation results. The InfoGain-driven DPO provides a principled way to enhance intermediate reasoning quality and reduce error propagation.
2. The experiments are extensive, covering two representative mobile benchmarks with both action type and action match evaluation, and the results demonstrate consistent improvements over dense and sparse baselines.
3. The visualizations and analysis (e.g., expert activation patterns, InfoGain distributions, ablation breakdowns) are clear and intuitive, which helps justify the design choices.

### Weaknesses
1. The paper targets mobile agents, but the efficiency aspect is not sufficiently addressed. In particular, there is no comparison in terms of FLOPs or computational cost, which is important for evaluating mobile settings and ensuring fairness across baselines.

2. here are a few minor typos in the paper. For example, in Section 3.4, Eq. (6), “Nrom” should be “Norm”.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenge of hybrid-capabilities reasoning in mobile agents - the multi-stage process of understanding screens, planning subtasks, deciding actions, and executing functions when following user instructions. Current approaches either focus on individual capabilities in isolation or use large-scale pre-training that leads to imbalanced performance. Traditional MoE models, while supporting capability decoupling, use input-oriented activation that fails to align experts with specific reasoning stages.
The authors propose Channel-of-Mobile-Experts (CoME), featuring output-oriented activation where experts are selected based on the current reasoning stage rather than input features. CoME employs four specialized experts with a channel router for dynamic selection, trained through a progressive three-stage strategy (Expert-FT, Router-FT, CoT-FT). Additionally, InfoGain-Driven DPO quantifies each intermediate step's contribution to mitigate error propagation.

### Strengths
- The paper is well-motivated and easy to read.
- The paper provides detailed analysis from multiple perspectives including expert activation distribution, InfoGain rewards, and efficiency.

### Weaknesses
- The fixed four experts (screen summary, subtask planning, action decision, action function) may limit extensibility to other domains or tasks.
- The experimental performance does not demonstrate significantly superior results compared to baselines. For instance, Auto-GUI achieves comparable performance to the proposed methodology despite using only 700M parameters versus 5B parameters.
- In Section 4.4.3, comparisons should include not only Qwen but also other baselines that show comparable performance.

### Questions
- Is there a way to train the router in a self-supervised manner without expert labels for task?
- How does CoME handle tasks where reasoning stages are not clearly defined?

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
3

### Summary
This paper introduces Channel-of-Mobile-Experts (CoME), a transformer-based architecture for hybrid-capability mobile agents. Each layer duplicates hidden channels into E experts, each with its own FFN, and a channel router dynamically activates or blends experts depending on the current reasoning stage. Training proceeds through three progressive phases (Expert-FT 2 Router-FT 2 CoT-FT). To reduce error propagation in chain-of-thought reasoning, the authors propose InfoGain-Driven DPO, which uses a reward model to estimate the information gain (IG) of intermediate steps and selects trajectories accordingly. Experiments on AITZ and AMEX datasets show improvements over dense and MoE baselines, supported by ablation studies.

### Strengths
1. CoME’s design (channel repetition, per-expert FFN, router-based activation) and the three-phase finetuning are clearly described with formulas and pseudo-code, making replication feasible.
2. InfoGain-DPO uses reward-model-estimated information gain to filter and reweight DPO pairs, providing an intuitive way to reduce mid-stage reasoning errors.
3. The three-phase finetuning schedule (Expert-FT / Router-FT / CoT-FT) is conceptually neat and empirically validated through ablations.
4. The paper includes multiple baselines and ablation studies (e.g., w/o Info-DPO, w/o Router-FT, w/o Expert-FT), demonstrating consistent contributions across datasets, proving its strong performance.
5. Experiments show CoME yields consistent improvements on both image-centric and text-centric mobile tasks, suggesting robustness to input modality variation.

### Weaknesses
1. The effectiveness of InfoGain depends heavily on the reward model’s quality. The paper notes performance degradation when switching from a 7B to a 72B model but lacks a systematic robustness analysis.
Some key hyperparameters, such as DPO thresholds, sampling count K, and Router-Norm settings, are partially described or omitted from the appendix

### Questions
Please refer to the weakness for more details

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies two primary challenges for mobile agents:
* Hybrid-Capabilities Reasoning: Agents must sequentially perform multiple distinct tasks (screen summary, subtask planning, action decision, and action function generation). Existing models, like dense models or standard Mixture-of-Experts (MoE), struggle to decouple and balance these different capabilities effectively. The paper argues standard MoE fails because its "input-oriented activation" is unsuited for this stage-based reasoning.
* Error Propagation: In a multi-step reasoning chain, small errors in early stages (like a poor screen summary) can cascade and lead to an incorrect final action.

To solve this, the paper proposes two main contributions:
* Channel-of-Mobile-Experts (CoME): A novel agent architecture with four specialized experts, one for each of the four reasoning stages. It uses a "Channel Router" that employs "output-oriented activation" to intelligently select the correct expert's hidden states based on the current reasoning stage (e.g., it uses the 'subtask plan' expert when generating the subtask plan).
* InfoGain-Driven DPO (Info-DPO): A new training technique to improve credit attribution. Instead of just rewarding the final correct action, this method uses reward models to measure the "information gain" of each intermediate step. It then uses DPO to fine-tune the agent, teaching it to prefer reasoning trajectories where each step positively contributes to the final answer.

The CoME model is trained using a progressive strategy (Expert-FT, Router-FT, CoT-FT) and then refined with Info-DPO. The authors demonstrate that this approach outperforms existing dense and MoE-based mobile agents on the AITZ and AMEX benchmarks.

Main Results in the Paper
* State-of-the-Art Performance: CoME achieves the highest overall action match accuracy on both benchmarks (Tables 1 & 2).
* Component Importance (Ablation Study): The ablation study (Table 3) shows that removing Info-DPO causes the largest performance drop (a 4.68% average decrease), while removing the Router-FT stage (which trains the "output-oriented activation") also causes a severe performance drop.
* Architectural Validation: The paper shows why CoME works. Analysis (Figure 5) shows CoME's Channel Router has a ~99% accuracy in selecting the correct expert for the corresponding reasoning stage. In contrast, a standard MoE model activates experts almost randomly.
* Training Strategy Validation: The proposed Info-DPO is shown to be superior to a simpler "Rule-DPO" (Table 4). Info-DPO provides substantial gains, especially on the complex "CLICK" action, because it provides supervision on the intermediate reasoning steps, not just the final outcome.
* Efficiency: CoME is shown to be more memory-efficient than baselines (Table 7).

### Strengths
* Effective Solution for Credit Attribution / Error Propagation: The paper tackles the difficult, well-known problem of error propagation in Chain-of-Thought (CoT) reasoning. The InfoGain-DPO method is a novel and logical way to apply fine-grained rewards to intermediate reasoning steps, moving beyond final-action accuracy. The approach seems general and can be interesting / relevant for other domains.
* Strong Empirical Analysis: The paper presents very thorough validation of the results. The ablation studies are clear, and the visualization of the router's activation (Figure 5) provides some evidence that the proposed mechanism works as designed.

### Weaknesses
* Weaknesses in the CoME architecture: The proposed CoME architecture is not very convincing. First, it is not clear if it will be broadly applicable. The architecture is hard-coded with four experts, one for each stage of reasoning defined by the AITZ dataset. It is unclear how this rigid structure would adapt to different tasks or datasets that might have more, fewer, or different reasoning stages. It lacks the inherent flexibility of a more general MoE model. Second, looking at the empirical results, it is hard to see fair comparison. The comparison is against different architectures (MoE, dense) and different parameter counts. It is very hard to make sense of the results without fair iso-FLOP studies.

### Questions
* Are there other studies to understand the effectiveness of CoME architecture? One possible study could be to (pre-)train on the same dataset but with different architectures while matching the total FLOPs.
* Info-DPO seems interesting, are there any studies on other domains require long CoT reasoning (e.g., math or code)?

### Soundness
3

### Presentation
4

### Contribution
2
