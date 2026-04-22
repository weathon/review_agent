# WETAP: Speculative Decoding with Width-Entropy Tree and Adaptive Pruning for LLMs Inference Acceleration

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
In inference acceleration of Large Language Models (LLMs), speculative decoding is used to coordinate draft model and target model, i.e., sequences are generated at the draft model and then verified in parallel at the target model, where the generation quality and speed of the draft model are the key issues. In this paper, we find that in a token tree, most of the child tokens are grew by few parent tokens with large probabilities in the low-entropy layer, and tokens with small probabilities in deeper layers also have potential to be accepted. Based on these observations, we propose WETAP, first constructing a token tree by determining the width of the next layer based on the entropy of the previous layer, then pruning it by considering both the probability and length of each token to retain the most potential ones. Experiments show that the proposed WETAP improves generation performance by up to 90% and furthermore increases up to 120% speed compared to other SOTA methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes WETAP, a training-free speculative decoding method that dynamically constructs a token tree during inference by leveraging the entropy of the previous layer to determine the width of the next layer, followed by an adaptive pruning strategy that jointly considers token probability and depth. Verification is performed from deep to shallow layers, evaluating beams rather than individual tokens to improve acceptance rates. The method requires no additional modules or training and demonstrates consistent improvements over state-of-the-art baselines in terms of generated length (up to +90%), inference speed (up to +120%), and perplexity across multiple models and tasks (e.g., MT-Bench, HumanEval, GSM8K).

### Strengths
Strengths：
Dynamic width allocation based on entropy: WETAP adaptively determines the width of each layer in the token tree according to the entropy of the previous layer, enabling more efficient exploration where it is most needed.

Depth-aware adaptive pruning: By jointly considering token probability and depth (length), WETAP preserves potentially valuable low-probability tokens in deeper layers—where draft-target alignment weakens—improving acceptance rates without sacrificing quality.

Unlike conventional token-by-token verification, WETAP verifies entire beams and starts from the deepest layer, allowing early termination and reducing redundant computation while maintaining high generation fidelity.

### Weaknesses
Increased per-step computational overhead: WETAP introduces additional inference-time computations—such as per-layer entropy estimation, dynamic width mapping, and composite score calculation for pruning—which may offset part of the speedup gains. However, the paper lacks a detailed breakdown of the time spent on these auxiliary operations versus the actual draft/target model forward passes. A latency or FLOPs analysis would help clarify whether the net acceleration remains favorable under realistic hardware constraints.

Hyperparameter sensitivity: The method introduces new hyperparameters (γ for entropy mapping, α for pruning weight). Although ablation studies are provided, their optimal values vary across tasks (e.g., α = 0.6 for HumanEval but 0.3 for MT-Bench), raising questions about robustness and the need for task-specific tuning in practice.

### Questions
On absolute speed reporting: The paper reports speed in tokens/second, which heavily depends on hardware (e.g., L20 GPU in Appendix B.4). Since this metric is not easily reproducible across setups, would the authors consider adding a vanilla autoregressive decoding baseline under identical conditions and reporting relative speedup (e.g., “1.8× faster than baseline”)? This would make the acceleration claims more interpretable and comparable.

On comparison with trained draft-based methods: The paper evaluates WETAP primarily against training-free baselines (e.g., SpecInfer, OPT-Tree) but does not include comparisons with trained speculative decoding frameworks such as EAGLE, EAGLE-2, or Medusa, which also construct token trees using learnable draft heads. Given that these methods can generate high-quality multi-token candidates through training, how does WETAP’s training-free, entropy-guided tree design compare in terms of both speed and generation quality? Would the proposed dynamic width and adaptive pruning strategies still yield advantages when the draft model is specifically trained for speculative decoding?

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
This paper introduces WETAP, a novel algorithm for Large Language Model inference acceleration via speculative decoding, focusing on dynamically optimizing the draft token tree construction and verification process. WETAP addresses limitations of fixed-width trees by determining each layer's width based on the entropy of the previous layer, thereby efficiently allocating exploration resources based on prediction certainty. It employs an adaptive pruning scheme that calculates a composite score considering both the token's cumulative probability and its depth, ensuring that deep but potentially successful tokens are retained. Furthermore, WETAP implements a "deep-to-shallow" verification strategy that processes candidate beams from the last layer forward, terminating immediately upon acceptance to minimize latency.

### Strengths
-  WETAP introduces a principled way to dynamically adjust the width of the draft token tree layers in proportion to the previous layer's output entropy. This mechanism ensures an accurate and sufficient draft exploration space is created, avoiding the computational overhead of overly wide, fixed-structure trees.

-  The method innovatively combines token probability with beam length (depth) in its pruning criterion. This approach effectively addresses the diminishing correlation between draft and target models in deeper layers, ensuring that tokens with relatively lower probabilities but high potential for long acceptance sequences are retained, leading to higher acceptance rates.

### Weaknesses
1. Questionable Validity of Deep-to-Shallow Verification Strategy. I do not fully understand the claimed effectiveness of the proposed "deep-to-shallow verification" strategy. On the one hand, the target model's verification process is inherently parallel: all draft tokens in the tree are evaluated via a single forward pass. Therefore, I fail to grasp the fundamental difference or advantage in processing the results sequentially from back-to-front (deep-to-shallow) versus front-to-back (shallow-to-deep). On the other hand, I do not understand why the acceptance of a deep-layer token implicitly guarantees the acceptance of its corresponding shallow-layer parent tokens in the beam. It is a common phenomenon that a misleading token in a shallow layer can cause the deep layers to be "misguided," leading to a draft model output that coincidentally appears similar to the target model's output. In other words, if the deep-to-shallow method can successfully identify an erroneous token upstream, then both front-to-back and deep-to-shallow verification methods should accept the same number of tokens, both halting at the first erroneous draft token.

2. Lack of Temperature=0 Experiments. Although temperature > 0 is a more commonly used strategy, the paper should provide results for  temperature =0 (Greedy Decoding). This is necessary to facilitate smooth reproduction by other researchers and enable a fairer comparison with existing baseline methods.

### Questions
- Can your proposed method be combined with existing Token Tree methods? Specifically, if it could be integrated with approaches like EAGLE-3 to further enhance its acceleration capability, the practical utility of WETAP would be significantly improved.

- Your method achieves the lowest PPL in the experimental results. Does this imply that the probability distributions of the generated results from your method's draft model and the target model are the most similar?

- How does your deep-to-shallow verification method guarantee a lossless final acceleration result, particularly in the scenario of Greedy Decoding  (temperature =0) ?

- I recommend that the authors provide performance metrics when integrating this method into practical, production-level libraries like vLLM, as this would greatly demonstrate its real-world deployability and practical value.

If you solve my problems, I'm willing to raise my score.

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
This paper improves speculative decoding of LLM based on two observations in a token tree: 

1) Most of the child tokens are grew by few parent tokens with large probabilities in the low-entropy layer.
2) The degree of correlation between the draft model and the target model decreases in deeper layers.

Correspondingly, this paper constructs a token tree by determining the width of the next layer based on the entropy of the previous layer, then pruning it by considering both the probability and length of each token. Experiments show that the proposed method improves generation performance and speed compared to SOTAs.

### Strengths
1. The key idea of proposed method is clearly represented and well motivated. It is based on two observations about the distributions of accepted tokens w.r.t entropy and layer. Those observations have not been discussed by previous literature.
2. The effectiveness of proposed method is supported by various experiments. It not only improves the speed measured by tokens per second but also improves the generation quality measured by perplexity score.

### Weaknesses
1. typos:

a) line 278, "tokenss" -> "tokens"

b) right side of figure 2, the range of Y calibration should be 6 to 16, instead of 0 to 16.

2. The implement details of tree pruning are not clearly represented. It is better to add the pseudo code of tree pruning and analyze its complexity. To maintain the tree structure, it seems that we can only prune leaf nodes.

3. The authors claim that it requires less time to verify tree from deep to shallow layers, compared with shallow to deep layers. However, there are no experiments or further analysis. In my opinion, to obtain the cumulative probability, we must traverse the tree from shallow to deep layers.

### Questions
The hyper-parameters are tuned per model/per dataset, or kept the same for all experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes extensions to popular tree-based speculative decoding methods for enhancing the speedup. The improvements include: (i) dynamically constructing the token tree by adaptively changing the width of each layer; (ii) pruning the token trees based on length as well probabilities of the tokens in each layer; (iii) verifying the tokens from right to left instead of left to right to increase the chances of acceptance.

### Strengths
- The speedup results are compelling -- the proposed method seems to be faster than comparable baselines.
- The core idea of constructing the token tree dynamically, based on the observed probabilities of tokens in each layer makes sense, and is empirically validated.
- There is a sufficiently detailed ablation analysis at the end which shows the impact of the hyperparameters introduced.

### Weaknesses
- When verifying from deep to shallow layers, it is not clear if the procedure still preserves the guarantees from speculative decoding that the resulting distribution will exactly match the target distribution. If this is not the case, there needs to be a more rigorous evaluation of any accuracy trade-offs that the method makes.
- On a related note, I find the use of perplexity as an eval metric strange here. Perplexity is generally measured against ground truth data, whereas here it is being used on the model generations itself. Hence, it is an uncertainty /entropy metric rather than telling us something about the quality of the resulting model. Combined with the fact that it is not clear if the SD guarantee holds, it is important to verify the model performance using actual downstream task metrics.
- Comparing the results in tables 1 and 2, it seems that most of speedup actually does come from the modified verification strategy -- removing the dynamic tree construction or the pruning strategy has little impact compared to the where the baselines are. Again, we need more discussion about what the modified verification strategy is actually doing to the output distribution.
- The method seems somewhat sensitive to the choice of hyperparameters -- so it would be good to see a generalization study where the gamma and alpha are tuned on one dataset and used on another one.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
