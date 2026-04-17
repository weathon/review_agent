# Learning to Reason over Continuous Tokens with Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Large Language Models (LLMs) have shown strong performance in complex reasoning tasks, especially when guided by Chain-of-Thought (CoT) prompting. However, conventional CoT reasoning in the discrete token space suffers from high computational and memory costs due to verbose intermediate steps. Recent work has explored latent reasoning in the embedding space to improve efficiency, but often at the cost of clarity and performance. In this work, we propose $\underline{Hy}$brid $\underline{Rea}$soning ($\texttt{HyRea}$), a unified framework that enables LLMs to dynamically switch between explicit (token-based) and latent (embedding-based) reasoning during inference. To train the model to make these decisions effectively, we introduce a two-stage training pipeline: (1) a supervised cold-start phase that introduces latent reasoning by replacing low-entropy CoT steps with embeddings, and (2) a reinforcement learning phase using Group Relative Policy Optimization (GRPO) to fine-tune the model’s reasoning strategy based on task-specific rewards. 
Experiments on mathematical reasoning benchmarks show that \texttt{HyRea} achieves significant reductions in token usage while maintaining or improving accuracy, offering an effective and scalable solution for efficient multi-step reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose aHybrid Reasoning framework that combines explicit (token-based) and latent (embedding-based) reasoning (i.e. interpretability vs efficiency) to tackle the inefficiency of standard CoT based reasoning. Training is a done as a two-step procedure. 

1. Supervised Cold-Start phase : 
the model starts to replace intermediate CoT steps (the ones with low entropy) with continuous embeddings. Specifically, for multiple rounds, such low entropy steps are identified, and replaced with ` [Latent] := <start-latent> c × [latent] <end-latent>`. The training loss is computed only on the explicit (non-latent) tokens. 

2. RL phase: 
The authors use GRPO to further train the model, with the standard accuracy and formatting reward. They additionally add a latent reward to encourage the model to think in latent space. 

The training procedure is conducted on two mathematical reasoning datasets for the first phase, a third one added for the RL phase. The Qwen 2 family of models (7B-32B) is used. Evaluation is conducted on standard Math benchmarks. 

Results show that the proposed approach strikes a better performance / efficiency tradeoff compared to the standard SFT and SFT+RL baselines. The authors moreover show that the entropy-based heuristic to identify which steps should be replaced in the first training phase performs well (better than random replacement), already striking a better tradeoff than standard SFT+RL. The authors additionally provide an analysis of different components of their method, as well as the transferability to other domains.

### Strengths
1. The authors are solving an important problem; the ability to retain the performance of CoT reasoning while making it more efficient has the potential for high impact in the field. 
2. The ability for the model to learn what (and what not) to compress is interesting.

### Weaknesses
1. A discussion on the training cost (esp. the added number of forward passes and the additional training time required) is missing.

### Questions
1. How exactly is the latent reward computed ? the ratio of CoT steps that are latent ?
2. Can the model do a reasoning step partially in latent space ?
3. I am not sure I understood Figure 7. If $c$ is the number of latent tokens in each latent step, shouldn't having more latent tokens increase performance ? Or should we interpret this figure as, for a given CoT step that has e.g. 10 explicit tokens, we keep `10-c` as not latent and `c` as latent ?
4. In Figure 6, how are you measuring the entropy ? Are you only measuring it for the `<start_latent>` token ? or are you also measuring it for intermediate latent tokens ?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents HyRea, which is a hybrid reasoning framework for LLMs that learns to switch during inference between explicit, token‑based chain‑of‑thought and latent-space computation, marked by special start-end tokens. It is trained via an entropy‑guided cold‑start SFT that replaces low‑entropy CoT steps with latent spans while supervising only visible tokens, and GRPO‑based RL with rewards for accuracy, formatting, and latent use. They evaluate on math benchmarks MATH‑500, Minerva Math, AMC23, OlympiadBench with Qwen2.5‑7B/32B models, where HyRea achieves comparable accuracy with substantially fewer output tokens and exhibits a small number of learned mode switches per solution. It also yields shorter outputs with competitive performance on MMLU and GPQA, suggesting a practical path to more efficient multi‑step reasoning without sacrificing quality.

### Strengths
- The paper provides a hybrid policy through cold-start training with entropy‑based step replacement, and selectable number of latent tokens with switch tokens is novel on top of existing papers.

- Good ablation studies to show the importance of different components of the method, along with a  behavioral analysis of switching patterns.

- The results are competitive with fewer tokens on multiple math benchmarks.

### Weaknesses
- The paper has many under-specified details on the experiments, especially regarding the RL training and the fairness/properness of the baselines compared against, and these are not even specified in the appendix. This makes the work unconvincing in its current form. Also, there are some unclear or incorrect terminologies used in the paper. I'm willing to change my rating if I get satisfactory clarifications on experimental details.

- The paper states that latent reasoning operates directly on embeddings, but explicit decoding already operates on and outputs token embeddings. So, I believe this terminology through using the term "embedding" is not clear for describing HyRea, which uses the hidden state $h_t$ as input.

- > “reward consists of both a format reward and an accuracy reward, as well as a latent reward. Specifically, we use the latent reward to guide the model in generating the token `[Latent]`. Additionally, during loss computation, we exclude the `[Latent]` token and calculate the loss only over the remaining token steps.”
  
  The paper is vague in its description of the reward function and there are other details missing regarding the RL training.

- The token counts in 4.2 don’t match Table 1. Table 1 reports 387 tokens for HyRea‑7B on MATH‑500 (not 287), 425 on Minerva‑7B (not 372), and 369 for HyRea‑32B on MATH‑500 (not 269).

- The experimental results are only for Qwen2.5 7B and 32B models, and there are gaps in model sizes, especially smaller ones, and there is no discussion of how results would change with these sizes.

### Questions
- How is per‑step entropy measured—token‑level entropy aggregated per step? What thresholding is used during cold start? Also, how is the threshold on the number latent tokens $S$ is selected, does it depend on the dataset/task or length of the output? There's no discussion of this in the paper.

- It's not mentioned in the paper what explicit reward function, clipping, or group size ($G$) is used during RL training. More importantly, once a trajectory includes latent tokens through hidden states that are tied to the current model weights, doesn’t that violate the on‑policy constraint of GRPO?

- It's not stated in the paper what inference parameters are used across methods for the discrete tokens. Also, Section 2.1 is written based on argmax without sampling, which I believe is not the most general description of "explicit reasoning".

- The paper does not properly discuss the COCONUT paper's setting. How are you reproducing the COCONUT paper on math benchmarks? Because what COCONUT actually does is curriculum-based learning by replacing the discrete tokens one-by-one with latent tokens, and at the end, it does full latent inference. So, are you following this on some SFT dataset? If you're not doing this, then the comparison with COCONUT is not fair. Also, how are you extracting the answer from COCONUT/Soft-Thinking trajectories? Please clarify these.

- The paper also needs to discuss how they are reproducing Soft-Thinking paper's setting (as there are also some hyperparameters to tune, such as entropy in cold-stop during inference). Importantly, if COCONUT and Soft-Thinking are used without training, it's unfair to compare against them in Table 2, as they don't see any training data. This might be the reason why they are falling behind in the benchmarks.

- The paper is missing connection to existing works on reinforcement learning training using continuous tokens, some example of which are "Hybrid Latent Reasoning via Reinforcement Learning" (https://arxiv.org/abs/2505.18454) and "Continuous Chain of Thought Enables Parallel Exploration and Reasoning" (https://arxiv.org/abs/2505.23648). Please discuss how the proposed method compares to these previous works, and add them as baselines to the tables if necessary.

### Soundness
1

### Presentation
1

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
This paper focuses on making reasoning with LLMs efficient by utilizing both discrete and continuous chain-of-thought (CoT) tokens. Towards this, the paper proposes Hybrid Reasoning (HyRea) that enables the LLM to adaptively switch between discrete CoT and continuous CoT during generation. HyRea consists of two phase training involving 1) a supervised cold-start phase and 2) an RL phase. By evaluating HyRea on various standard benchmarks, the authors show that HyRea exhibits a favorable trade-off between the model quality/accuracy and efficiency in terms of thought token lengths. Furthermore, the authors also demonstrate the good out-of-domain generalization beyond mathematical tasks for HyRea.

### Strengths
- The paper studies a timely problem of making the LLM reasoning more efficient.
- The proposed HyRea method is simple and effective, showcasing a good trade-off between reasoning performance and thought token lengths across multiple benchmarks. 
- The paper presents a detailed ablation study of various design choices.
- The paper explores the generalization ability of HyRea trained models beyond mathematical tasks.

### Weaknesses
- Given that HyRea is aimed at achieve a good performance vs. token length trade-off, which sometimes results into worse performance than SFT + RL with discrete CoT (see Table 1 & Table 3), it is important to consider agains baselines that focus on achieving similar trade-off with just discrete CoT. If one encourage shorter thought lengths in SFT + RL via some form of length regularization or training on compressed thought sequences, would the resulting performance vs. thought length trade-off be worse than HyRea?
- In Section 4.3, difference between **Random** and **Entropy** is not very significant. Authors should compare results from multiple runs to make a stronger case for **Entropy**. 
- In L221-223, the authors don't take [Latent] tokens into account while defining loss. What is the justification for this? Did the authors perform ablation for this design choice?

### Questions
- Could the authors provide examples similar to Figure 4 but with longer CoTs? Do the HyRea produce CoTs remain interpretable?
- In Line 365, why are authors referring to CoT + RL as **upper bound** in accuracy? According to Table 2, this upper bounds is often exceeded by Random and/or Entropy.
- While Figure 5 shows superiority of Entropy for cold-start stage, looking at Table 2 and Figure 5, Random appears to be more amenable to RL as the gap between Random and Entropy after RL is quite small?
- In Line 409, the authors say "The inverse relationship highlights a trade-off...". According to Figure 7, smaller c is better for both accuracy and thought length. What trade-off are the authors referring to?
- In Section 3.1, the authors may want to highlight how their method differs from Coconut.
- In Line 40, the authors state ``However, reasoning over embeddings remains inherently
difficult and can lead to degraded model performance compared to traditional token-based inference,
as certain tokens encode complex, nuanced information that cannot be fully preserved in compressed
embeddings. `` Could the authors provide a citation or evidence for this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose HyRea that lets LLMs adaptively alternate between explicit token-level reasoning and latent embedding-level reasoning, with special tokens marking the start and end of the latent span. The training has two stages: 

1. Cold-Start Supervised Fine-Tuning (SFT):
Low-entropy CoT steps are replaced with latent spans to introduce embedding-based reasoning while preserving interpretability.

2. Reinforcement Learning (RL) with Group Relative Policy Optimization (GRPO):
Fine-tunes the model to learn optimal switching between reasoning modes, guided by task-specific rewards for accuracy, format, and efficiency.

Experiment results on math benchmarks, MMLU and GPQA show the performance of HyRea is in between vanilla SFT and SFT + RL, with significantly shorted generation.

### Strengths
1. The paper proposes a elegant solution of alternating latent and explicit reasoning with special tokens + cold start SFT + RL.

2. The performance of HyRea on math benchmark is close to vanilla SFT+RL, while the generation length is significantly shorter.

3. The paper is well-written and easy to follow.

### Weaknesses
1. There is only one latent CoT with short generation baseline (COCONUT), which was out last year, and is a weak baseline. The soft thinking baseline is training-free and does not shorten the generation, which does not align with the settings in the paper. It's a bit hard to calibrate the contribution of HyRea without suitable baselines. I'd suggest the authors to include more recent, stronger baselines that match their settings (training latent CoT to shorten generation). e.g. CODI (https://arxiv.org/abs/2502.21074). There are also quite a number of papers that produce mixed CoT chains. e.g. Token Assorted (https://arxiv.org/abs/2502.03275). RL with latent tokens: HRPO (https://arxiv.org/abs/2505.18454).
2. Please include a detailed description of the reward functions (accuracy, format, latent) in the main text.
3. If I understand correctly, the math-trained HyRea can perform reasonably well out-of-domain on MMLU and GPQA according to Table 3. Is SFT and SFT+RL also trained on math and tested OOD? It would be better to include other latent reasoning baselines to better calibrate the results.

### Questions
See weeknesses.

### Soundness
3

### Presentation
3

### Contribution
2
