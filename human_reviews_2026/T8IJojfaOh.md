# Next-ToBE: Probabilistic Next Token-Bag Exploitation for Activating Anticipatory Capacity in LLMs

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Auto-regressive large language models (LLMs) exhibit a non-trivial capacity to "anticipate'' long-range future tokens despite being trained to predict only one token at a time. Nevertheless, how to systematically profile, enhance and leverage such capacity to practically improve LLM reasoning performance remains unclear. In this paper, we propose **Next Token-Bag Exploitation (Next-ToBE)** to tackle this challenge. Next-ToBE quantifies LLM’s anticipatory capacity by measuring how well tokens in the future window are pre-captured by the model’s current softmax probabilities. This capacity is  strongly correlated with LLM generative quality but often suppressed by the rigid one-hot objective in next-token prediction. To address this, we replace the {one-hot target vector} in next-token prediction  with a soft target distribution 
spanning additional future tokens. Specifically, the immediate next token retains the highest importance, while more distant ``look-ahead tokens'' are also included to enrich supervision, with their importance  dynamically determined by temporal and semantic relevance patterns to   inject forward-looking pressure. 
Besides, the fitting process emphasizes the model’s intrinsic anticipatory tendency, thus preserving the confidence and fidelity of the pre-trained model to improve training stability. 
Overall, Next-ToBE not only effectively activates LLM  anticipatory capacity through fine-tuning, yielding notable gains in  reasoning performance with higher memory and computational efficiency against the MTP baselines, but also shows great potential in pretraining setting by successfully cultivating  this capacity from scratch. These  highlight its value as an effective strategy to extend the prediction horizon of LLMs, enabling them to see further, and reason better.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Next-ToBE, a new fine-tuning objective as an alternative to vanilla NTP. 

The paper points out that regular LLMs only focus on predicting the very next word/token, which stops them from actually thinking ahead (but it does already exist to a certain extent).

The authors created their new training method, Next-ToBE that teaches the model to predict not just the next token, but also a few tokens down the line at the same time. Instead of simply predicting the next work, it predicts a probability of the next few words.

The key idea is that this helps the model develop better planning skills by making it look further ahead during training.

The authors demonstrate Next-ToBE's effectiveness by fine-tuning Qwen and Llama models on tasks in mathematical, code, and commonsense reasoning.

### Strengths
- The paper is well written, motivated, an executed
- The empirical results (Table 1) look solid. In the ran experiments, Next-ToBE is clearly the best method.
- The proposed Future-tokens Hit Rate (FtHR) is a nice metric
- I appreciate the detail in training method, data, eval pipeline

### Weaknesses
- My biggest concern is that it's only done on models that are already post-trained. Pure NTP/MTP is typically done for pretraining, so I'm not sure why this is done on post-trained models. I think the results would be a lot more convincing if it was done on base models, on pretraining data, with full model tuning (not just lora).
- The conclusion claims that it is "simple to implement", the method is quite complicated, especially compared to MTP. Particularly the part about the random walk on the graph. In its current form, it will likely not be adopted by many.
- A lot of the method is heuristically driven. E.g. chosen values of lambda. How can others use this method? Will it always work out of the box on any model with the fixed hyperparams? I think more ablations of the hyperparams (alpha, beta, lambda) are necessary
- It's unclear how much more efficient Next-ToBE is than other methods. There is overhead from computing the values for the lookahead tokens, but not sure how much.

### Questions
- Why are all experiments/results shown on post-trained models? Why not evaluate this on a base model? Do the same properties hold there? I would consider raising my score if this can be answered.
- How expensive is the random-walk weighting calculation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a loss function that encourages an autoregressive LM’s prediction to include more information about future tokens, without changing the underlying Transformer architecture or compromising inference accuracy or speed. The idea is to use a **mixture of two cross-entropy losses**: (i) between the LM head and the correct **next** token (one-hot), and (ii) between the model’s **bag-renormalized** prediction over the next (k) tokens and a set of importance weights over those tokens. Next-ToBE **nearly unanimously** outperforms NTP, MTP, and MuToR across numerous math, code, and commonsense reasoning benchmarks. Since there is only one LM output distribution per position, inference speed stays comparable to NTP (unlike MTP). The appendix contains many additional informative experiments.

### Strengths
Clearly written, strong and extensively evaluated results, and an improvement on a central topic (raw language modeling itself).

### Weaknesses
The weighting scheme seems ad hoc and complicated, containing several hyperparameters (lambda, k, epsilon, gamma, h, rho) and the fusion of W_tau and W_s). However, the random-walk matrix is backed up by ablations, and upon reflection, the random walk here may not be so ad hoc.
Missing some minor distributional diagnostics (see questions).

### Questions
1. How many tokens of CoT were used in the math evaluation protocols (a maximum is provided, but not mean/SD)? I would like to understand why autoregressive sampling does not appear to pull the Next-ToBE model off the token-by-token distribution. Conceptually, since the loss divides by the probability mass assigned to the future (k)-bag, the auxiliary term should mainly reshuffle probabilities **within** that set, but I would still expect the opposite effect from what Figure 3b shows.
2. Does long-term Next-ToBE **autoregressive** sampling diverge from the training distribution—e.g., as measured by KL **to/from** a stronger language model?
3. Does this technique still work if you train from scratch rather than fine-tuning a pretrained baseline? I ask in part because the alpha weights focus on tokens that already receive probability mass, which will start **at random** when training from scratch.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Next-ToBE, a training objective to improve the anticipatory capability in LLMs. Next-ToBE modifies the standard next-token prediction objective in LLM training by replacing the one-hot target vector with a soft distribution over additional future tokens. The model incorporates a weighting scheme that assigns a dominant weight for the immediate next token, preserves the model's own anticipatory preferences, and dynamically adjusts future token weights based on temporal and semantic importance. Experiments on diverse reasoning settings (math reasoning, code generation, and common-sense reasoning) using Qwen and Llama models show consistent accuracy improvement over NTP and MTP baselines. The approach requires no architectural change and reduces peak memory usage compared to MTP methods.

### Strengths
- The method is well-motivated by the two key empirical observations. The defined future token hit rate metric and the corresponding analysis provide valuable insights into LLMs' intrinsic anticipatory behavior and how it's linked to generative accuracies.
- The objective is straightforward, and the design choices are well-motivated and supported by theoretical justification and ablation studies.
- The method is simple and requires no architectural or inference changes. It reduces peak memory usage compared to MTP methods.
- The paper provides extensive experiments across diverse reasoning tasks and model families and demonstrates consistent improvements.

### Weaknesses
- The scope of baselines could be limited. The related work section acknowledges connections to label smoothing and other MTP-related baselines, but they are not evaluated as baselines, including ProphetNet, token order prediction, and label smoothing.
- Direct quantitative and qualitative comparison with NTP is unclear. For example, the computational overhead for the weighting scheme and potential side effects (e.g., hallucinations) of Next-ToBE are not discussed. 
- Experiments are confined to relatively small models (8B), and how well the proposed method scales to larger models is unclear. Evaluation of larger models could strengthen the claims.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

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
This work proposes a novel training objective that adds a more learning signal at every time step by training to model to predict future tokens as well as the next token using the same output projection from the model. Empirically authors show how existing models are already capturing this sort of information about the future in the next token distribution and call it anitcipatory knowledge. Then they hypothesize that improving anticipatory capability in the next token distribution will improve generative performance of the model.

Authors design the objective with careful balancing between next token and future tokens learning; where they introduce weights that are based on the look-ahead distance as well as other scaling params.
Experiments show that their proposed method gives slightly better performance across the deck in most of benchmarks they consider compared to both next token and multi token prediction alternatives.

### Strengths
* well grounded motivation with empirical data point showing anticipatory knowledge already existing in the models.
* experiments are clearly described and adequate baselines and related work methods were considered.

### Weaknesses
* the balancing weights and added renormalization make the whole objective a bit cumbersome. Not very clear what might be the optimal value of K i.e. either we want to get as much anticipatory as possible or how do we optimize it? 

* its known that MTP shows the substantial gains at scaling; so this method might as well show much more gains at scale. While this might be infeasible for authors due to compute constraints, absence of larger models and on more data is a minor weakness.

### Questions
How do you think anticipatory information can be further utilized ? e.g. during inference?

### Soundness
3

### Presentation
3

### Contribution
3
