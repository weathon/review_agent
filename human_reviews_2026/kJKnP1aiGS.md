# Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Adapting language models (LMs) to new tasks via post-training carries the risk of degrading existing capabilities---a phenomenon classically known as *catastrophic forgetting*. In this paper, we set out to identify specific guidelines to mitigate this phenomenon, by systematically comparing the forgetting patterns of supervised fine-tuning (SFT) and reinforcement learning (RL), two widely adopted post-training methods. Our experiments reveal a consistent trend across LM families (Llama, Qwen) and tasks (instruction following, general knowledge, and arithmetic reasoning): RL leads to less forgetting than SFT while achieving comparable or higher target task performance.
To investigate the cause for this difference, we consider a simplified setting in which the LM is modeled as a mixture of two distributions, one corresponding to prior knowledge and the other to the target task. We identify that the *mode-seeking* nature of RL, which stems from its use of *on-policy* data, enables keeping prior knowledge intact when learning the target task. We then verify this insight by demonstrating that the use on-policy data underlies the robustness of RL to forgetting in practical settings, as opposed to other algorithmic choices such as the KL regularization or advantage estimation. Lastly, as a practical implication, our results highlight the potential of mitigating forgetting using *approximately* on-policy data, which can be substantially more efficient to obtain than fully on-policy data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper systematically compares catastrophic forgetting in language models during post-training via supervised fine-tuning (SFT) and reinforcement learning (RL). Through experiments across multiple model families and tasks, the authors demonstrate that RL exhibits significantly less forgetting than SFT while achieving comparable or better target task performance. They attribute this robustness to RL's mode-seeking behavior, which stems from its use of on-policy data, and further show that even approximately on-policy data can mitigate forgetting in SFT.

### Strengths
1. The paper provides a comprehensive and rigorous empirical evaluation across diverse tasks, making the findings highly robust and generalizable.
2. The authors offer an intuitive yet formal explanation for the observed phenomenon by modeling the policy as a mixture of distributions and linking forgetting behavior to the mode-seeking nature of reverse KL minimization.
3. The practical implication that approximately on-policy data can significantly reduce forgetting is a valuable and efficient alternative to full on-policy RL, offering a useful guideline for real-world model adaptation.

### Weaknesses
1. Forgetting is measured via average accuracy drops; other forms of degradation (semantic drift, safety loss, calibration changes) are not quantitatively explored.
2. The experiments are limited to models of up to 8B parameters, and it is unclear whether the same trends hold for significantly larger or smaller models, limiting the scalability claims.
3. While the Gaussian mixture analogy provides valuable intuition, it may oversimplify the complex, high-dimensional, and often non-Gaussian nature of real-world language model distributions.

### Questions
1. How closely must data match the current policy to be considered "approximately on-policy," and what are the precise thresholds for its effectiveness in mitigating forgetting?
2. Does the observed robustness of RL hold for capabilities beyond knowledge and reasoning, such as in multimodal or conversational tasks, where forgetting patterns might differ?
3.  Is the reduced forgetting in RL achieved at the cost of slower convergence or higher sample complexity compared to SFT, and what are the implied trade-offs for practical deployment?

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
The paper investigates catastrophic forgetting of tow prevalent LLM post-training paradigms: supervised fine-tuning and reinforcement learning. Empirical results across several model families and tasks show that RL (particularly on-policy algorithms like GRPO) achieves comparable or better target-task performance while exhibiting less forgetting on non-target tasks. To explain this, the authors propose that RL’s mode-seeking behavior (reverse KL), rooted in its use of on-policy data, helps preserve prior knowledge. A simplified mixture-of-Gaussians analysis further illustrates how SFT and reverse RL behave differently under uni-modal vs multi-modal assumptions.

### Strengths
- The paper articulates a concrete and important question, i.e., why RL fine-tuning forgets less than SFT, and provides extensive experimental results supporting the finding across architectures and datasets.

- The writing and figures are well-organized. In particular, the “gain–drop” metric and visualization (Figure 2) make results intuitive, and the toy Gaussian analysis offers a didactic explanation.

- The study disentangles potential confounders (KL regularization, advantage estimation) and clearly isolates the effect of data policy.

### Weaknesses
- The finding that on-policy learning mitigates forgetting better than off-policy learning has already been explored in both RL and alignment literature. Notably, “Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data” (Tajwar et al., 2024) also frames on-policy vs off-policy updates as mode-seeking vs mode-covering, drawing the same connection between reverse KL and improved retention. Thus, while this paper extends that reasoning to an explicit forgetting study, its conceptual contribution over the established framework appears incremental.

- Given that on-policy methods (iterative SFT, DPO, rejection-sampling, GRPO, PPO) are already standard practice, the practical insight, i.e., “on-policy mitigates forgetting”, may have limited influence on future method design unless coupled with deeper theoretical grounding or new algorithmic proposals.

- The mixture-of-Gaussians setting helps intuition but does not rigorously establish the link between KL directionality, multimodality, and empirical forgetting in high-dimensional LLMs.

### Questions
I noticed that the statement “Conventional wisdom presumes that the mode-seeking nature of reverse KL enables faster learning … while the mode-covering forward KL should maintain probability mass across modes.” cites (Chan et al., 2022; Tajwar et al., 2024b). However, upon reviewing Tajwar et al. (2024b), I couldn’t find an explicit discussion of the latter claim regarding forward KL preserving mode coverage. Could the authors please clarify whether this interpretation is directly supported by that work or derived from general understanding in the literature?

### Soundness
3

### Presentation
4

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
This paper studies the problem of catastrophic forgetting in the post-training of LLMs. It finds that reinforcement learning suffers almost no forgetting because it uses on-policy data, whereas SFT easily forgets due to off-policy training. The authors propose Iterative-SFT, which generates new data with the current model at the beginning of each epoch, achieving approximately on-policy learning. This method avoids the complexity of RL training while significantly mitigating forgetting.

### Strengths
- The paper conducts both experimental evaluations and theoretical analyses of catastrophic forgetting in SFT and RL, and the conclusions are convincing.
- It clearly shows that the reason RL resists catastrophic forgetting lies in its on-policy nature of data, rather than KL regularization or advantage estimation, which is an observation of notable value.

### Weaknesses
- The practicality of Iterative-SFT may be limited for two reasons: 1) Since the policy model generates its own training data, the generated examples may not be sufficiently challenging compared to data produced by a stronger teacher model; 2) It requires a reward model or rule-based verification methods to score and filter the data; however, because the policy model itself may not be well-versed in the target domain, the proportion of high-quality samples could be low, placing high demands on the reward model/rule-based verification methods.

### Questions
- Why are the Self-SFT and SFT curves shown as straight lines, rather than plotted after each epoch like Iterative-SFT with a stepwise curve?

### Soundness
4

### Presentation
3

### Contribution
3
