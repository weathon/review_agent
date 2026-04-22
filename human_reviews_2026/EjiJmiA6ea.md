# Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Data contamination poses a significant threat to the reliable evaluation of Large Language Models (LLMs). This issue arises when benchmark samples may inadvertently appear in training sets, compromising the validity of reported performance. While detection methods have been developed for the pre-training and Supervised Fine-Tuning stages, a critical research gap exists for the increasingly significant phase of Reinforcement Learning (RL) post-training. 
As RL post-training becomes pivotal for advancing LLM reasoning, the absence of specialized contamination detection methods in this paradigm presents a critical vulnerability. 
To address this, we conduct the first systematic study of data detection within  RL post-training scenario and propose Self-Critique. Our method is motivated by a key observation: after RL phase, the output entropy distribution of LLMs tends to collapse into highly specific and sparse modes. Self-Critique probes for the underlying policy collapse, i.e., the model's convergence to a narrow reasoning path, which causes this entropy reduction.
To facilitate this research, we also introduce RL-MIA, a benchmark constructed to simulate this specific contamination scenario. 
Extensive experiments show that Self-Critique significantly outperforms baseline methods across multiple models and contamination tasks, achieving an AUC improvement of up to 30%. Whereas existing methods are close to a random guess for RL-phase contamination, our method makes detection possible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a membership inference attack strategy for models trained on  reinforcement learning with feedback via the exploitation of the entropy collapse in RL learned policies. The work relies on the observation that the RL learned response traces often suffer from entropy collapse on high reward regions. The key idea behind the work is to use the length penalized entropy measure between an original prompt’s response and the response generated with an added critique meta prompt as a mechanism for member detection. The paper showcases that in RL trained policies the proposed method performs better than the existing works which were designed towards next token prediction training objectives (SFT, Pretraining etc)

### Strengths
1. The paper is easy to follow and is well written.

2. Dataset and model diversity is present in the experimentation. 

3. To my knowledge this is the first MIA work reinforcement learning alignment. But my expertise in MIA attacks is also limited. 

4. The proposed method showcases effective performance as opposed to other baselines which are designed towards models trained with maximum likelihood. 

5. A certain level of ablation has been performed on different RL methods under Section 5.4.

### Weaknesses
1. The method seems to be motivated by the sparse reward assumption of verifiable reward. Human feedback based preference datasets still play a vital role in capturing human feedback. While traditional reward models are learned via the Bradley Terry formulation on a complete response there, token level reward modelling has also gained prominence. Further implicit reward based alignment such as DPO also exists. Can you provide the prospects of the generalizability of the method across these different alignments. 

2. Can you please provide further details about the reward models used for the RL training under the section “Contaminate in Different RL algorithms”

### Questions
See weaknesses

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
This paper addresses the overlooked issue of data contamination during the RL post-training phase of large language models (LLMs). This work explores contamination specific to RL post-training. The authors introduce Self-Critique, an entropy-based probing method that measures the similarity between two model outputs: an initial answer and a self-critique re-generation conditioned on that answer. A high similarity in entropy curves between the two is interpreted as evidence of contamination, under the hypothesis that contaminated examples lead to policy collapse toward memorized reasoning paths.

### Strengths
1. Brings attention to the underexplored problem of contamination during RL post-training, which is an important practical concern.
2. Proposes a simple, reproducible probing method that builds on self-consistency and entropy measurement.
3. Introduces RL-MIA, a useful benchmark that may benefit future work in this area.

### Weaknesses
1. Weak Novelty: The proposed “Self-Critique” method is a straightforward application of self-consistency probing combined with entropy comparison. It does not provide a fundamentally new theoretical or algorithmic contribution.
2. Questionable Signal Reliability: The paper assumes that contamination causes entropy path similarity, but similar effects can arise from deterministic reasoning or narrow reward optimization. The entropy path is not guaranteed to distinguish contamination from general RL collapse. And it is doubtable that the method’s behavior is closely tied to the chosen prompt and “self-critique” instruction. In more diverse or flexible RL training settings, the entropy path could vary widely, undermining the stability of the metric.

### Questions
1. Why entropy path is a reliable signal and is it sensitive to the template diversity?
2. What is the key difference of the self-critique probing and direct probing with different templates?

### Soundness
2

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
4

### Summary
This paper addresses the problem of detecting data contamination in LLMs that occurs during the RL post-training phase, a critical gap where existing detection methods, designed for pre-training and SFT, fail. It proposes a novel detection method named Self-Critique and a new benchmark called RL-MIA  to simulate and evaluate this specific type of contamination. The idea of the method is to probe for "policy collapse" (memorization) by instructing the model to generate an initial response and then a second, "self-critique" response with a different reasoning path. The method then compares the token-level entropy sequences of both responses. High similarity in their entropy patterns indicates the model is stuck on a memorized path and is therefore likely contaminated. The empirical studies involve evaluating Self-Critique on the new RL-MIA benchmark, which uses math (AIME 2024/2025) and logical reasoning (K&K, SAT) datasets. The method is tested on models like Qwen2.5-7B and DeepSeek-Math-7B, comparing its performance (using AUC) against several baseline detectors (like PPL, Min-K%, and CDD) and testing its robustness across different RL algorithms (PPO, GRPO, DAPO).

### Strengths
This is overall a good paper. The motivating problem is novel and well explained. The paper is the first to systematically study data contamination specifically in the RL post-training phase. It clearly explains why this is a new problem: RL optimizes for a reward signal, not likelihood, which makes existing likelihood-based detectors (like PPL) ineffective (as shown in Table 2, where they perform near random guess).
The method is simple, yet tailored to the problem: The Self-Critique method is an intuitive and clever solution. Instead of checking likelihood, it probes for "policy collapse"—a known side-effect of RL training. The method of comparing the entropy of an initial answer and a "self-critique" answer is a simple and effective way to measure this path dependency.
The writing is very clear. The paper clearly formalizes the problem (Section 3.1) and provides a step-by-step algorithm for the Self-Critique method (Section 4.1, Algorithm 1). 
Thorough Experimental Validation: The experiments are comprehensive and support the claims.

- Baselines: It compares Self-Critique against a wide range of baselines (PPL, Min-K%, Recall, CDD) and even custom entropy-based baselines (Entropy-Temp, Entropy-Noise).

- Robustness: It shows the method works across multiple models (Qwen2.5-7B, DeepSeek-Math-7B, etc.), diverse datasets (AIME, K&K, SAT), and different RL algorithms (PPO, GRPO, DAPO, as seen in Table 3).

- Ablations: It includes valuable ablations on the effect of Top-K entropy approximation (Table 4) and sampling strategy (Figure 5).

- Dual-Contamination Study: The analysis in Section 5.4 is particularly strong, showing how Self-Critique can isolate RL-phase contamination even when pre-training contamination is also present, with performance improving up to 55%.

### Weaknesses
- Result Variability and Lack of Error Bars: There appears to be significant variability in the performance of some methods across the two main models (Table 2). 
- Inference Cost: The Self-Critique method requires the model to generate two full responses (initial and critique) and requires token-level probability access to compute entropy for both. This is computationally more expensive than simpler methods, like PPL, which only require one pass.

### Questions
- Table 2 shows significant variability in the performance of baseline methods across different models (e.g., Entropy-Noise on the SAT dataset). Could you provide more intuition on why this variability is so high? Can you bootstrap to get an error bar?
- The paper compares methods with different inference costs (e.g., Self-Critique requires 2 generations, PPL requires 1, and CDD may require N). Could you comment on the performance vs. compute trade-off? For instance, how does Self-Critique compare to a baseline like CDD when both are limited to the same computational budget (e.g., 2 samples)?
- The experiments primarily use Qwen and DeepSeek models. To what extent do you believe the results are general? Would you expect similar performance and baseline rankings if the method were replicated on a different model family, such as Llama, which may have different architectural properties and pre-training data?

I will be happy to increase my score if the authors address these questions.

### Soundness
3

### Presentation
3

### Contribution
3
