# C-Voting: Confidence-Based Test-Time Voting without Explicit Energy Functions

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Neural network models with latent recurrent processing, where identical layers are recursively applied to the latent state, have gained attention as promising models for performing reasoning tasks.
A strength of such models is that they enable test-time scaling, where the models can enhance their performance in the test phase without additional training. 
Models such as the Hierarchical Reasoning Model (HRM) and Artificial Kuramoto Oscillatory Neurons (AKOrN) can facilitate deeper reasoning by increasing the number of recurrent steps, thereby enabling the completion of challenging tasks, including Sudoku, Maze solving, and AGI benchmarks. 
In this work, we introduce confidence-based voting (C-voting), a test-time scaling strategy designed for recurrent models with multiple latent candidate trajectories. 
Initializing the latent state with multiple candidates using random variables, C-voting selects the one maximizing the average of top-1 probabilities of the predictions, reflecting the model’s confidence. 
Additionally, it yields $4.9\\%$ higher accuracy on Sudoku-hard than the energy-based voting strategy, which is specific to models with explicit energy functions.
An essential advantage of C‑voting is its applicability: it can be applied to recurrent models without requiring an explicit energy function. 
Finally, we introduce a simple attention-based recurrent model with randomized initial values named ItrSA++, and demonstrate that when combined with C-voting, it outperforms HRM on Sudoku-extreme ($95.2\\%$ vs. $55.0\\%$) and Maze ($78.6\\%$ vs. $74.5\\%$) tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces C-voting, a model-agnostic test-time voting strategy for recurrent neural models used in complex reasoning tasks. Unlike previous methods like E-voting, C-voting does not require an explicit energy function. Instead, it generates multiple candidate solutions from random initializations and selects the one with the highest "confidence," defined as the average top-1 probability across all prediction outputs. The authors also propose ItrSA++, a simple 3M-parameter recurrent attention model. Experiments show C-voting is effective: it boosts the performance of HRM on Sudoku-extreme (55.0% to 71.2%), outperforms E-voting on the AKOrN model (94.4% vs 89.5%), and, when combined with ItrSA++, achieves new state-of-the-art results on Sudoku-extreme (95.2%) and Maze-hard (78.6%).

### Strengths
1. The paper is well-written, clearly organized, and effectively motivates the work. It pinpoints the limitation of E-voting (its reliance on an explicit energy function) and proposes a logical, general solution.
2. The core contribution, C-voting, is intuitive and model-agnostic. Using the average top-1 probability as a proxy for confidence is a clean and effective way to extend test-time voting to a broader class of recurrent models that lack energy functions.
3.  The results demonstrate the method's effectiveness. C-voting outperforms specialized E-voting on the AKOrN model. Furthermore, when combined with the new lightweight ItrSA++ model, it achieves promising performance on difficult reasoning benchmarks like Sudoku-extreme and Maze-hard.

### Weaknesses
1.  The paper's core contribution, C-voting, hinges on defining confidence as the average of top-1 probabilities. This is a plausible choice, but it is one of many possible uncertainty metrics. It would be helpful to  compare this choice to other common metrics (e.g., negative entropy, sum of log probabilities). Without this comparison, it might be unclear if the chosen metric is optimal or simply one that works well for Sudoku, limiting the understanding of why C-voting is effective.

2.  The paper introduces both a new voting method (C-voting) and a new model (ItrSA++). The SOTA results (e.g., 95.2% on Sudoku-extreme) are achieved by combining them. But it seems hard to disentangle the contributions. How much of the performance gain comes from the strong baseline architecture of ItrSA++ itself, and how much is added by C-voting? It would be helpful to provide the baseline (K=1) performance of ItrSA++ on all tasks to isolate and quantify the true benefit of C-voting. Furthermore, the design choices for ItrSA++ (Cross-Attention, SwiGLU, etc.) are not well-justified with ablation studies.

### Questions
Please refer to above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
### Summary

This paper introduces C-voting (Confidence-based Voting), a test-time scaling strategy for recurrent neural networks that enables performance improvements without additional training. Unlike existing energy-based voting (E-voting) methods that require explicit energy functions, C-voting works by:
- Sampling multiple random initial latent states
- Running the recurrent model from each initialization
- Selecting the trajectory with the highest average top-1 prediction probability (confidence)

### Main contributions:

- C-voting method: A model-agnostic voting strategy applicable to any recurrent model with randomized initialization, not just those with explicit energy functions
- Integration with HRM: Demonstrates that C-voting improves HRM's Sudoku-extreme accuracy from 55.0% to 71.2%
- Comparison with E-voting: Shows C-voting outperforms E-voting on AKOrN for Sudoku-hard (94.4% vs 89.5%)
- ItrSA++: Introduces a simple attention-based recurrent model (~3M parameters) that achieves state-of-the-art results when combined with C-voting: 95.2% on Sudoku-extreme, 94.4% on Sudoku-hard, and 78.6% on Maze-hard

The paper evaluates on three reasoning benchmarks (Sudoku-hard, Sudoku-extreme, Maze-hard) and shows that C-voting provides effective test-time scaling through parallelizable sampling rather than sequential depth increases. The method's effectiveness is attributed to selecting trajectories based on model confidence, which serves as a proxy for prediction accuracy in well-calibrated models.

### Strengths
- The paper identifies a genuine problem with E-voting (requires explicit energy functions) and proposes a model-agnostic alternative that can be applied to broader classes of recurrent models like HRM and recurrent transformers.
- C-voting is simple to understand along with the theoretical justification provided in the paper of how selecting the most confident trajectory could be seen as searching for the optimal solution in Sudoku. The method itself is easy to implement and integrate into existing models without architectural changes. 
- Shows positive results across multiple models (HRM, AKOrN) and tasks, with substantial gains (e.g., HRM: 55.0% → 71.2% on Sudoku-extreme; AKOrN: 89.5% → 94.4% on Sudoku-hard). Although the paper doesn't report performance of all models across all the benchmark.

### Weaknesses
- C-voting vs. E-voting is only shown on Sudoku-hard (Figure 3), despite the authors having trained AKOrN models and evaluating on three datasets total. This makes the claim that "C-voting outperforms E-voting" (Section 6.3) inadequately supported. A complete comparison on all benchmarks is needed to validate this central claim.
- Figure 2's comparison seems misleading. it shows HRM's original performance as a flat line while scaling only C-voting's sample count. A fair comparison would show HRM's native test-time scaling method (increasing recurrence depth) vs. C-voting on a compute-normalized x-axis. The paper claims that the performance can saturate with increasing recurrence depth. However, the same is applicable for increasing the same count for C-voting and it's not clear whether C-voting would reach higher test-time optimized accuracy than HRM.
- Evaluation is restricted to three small reasoning tasks (1,000 test samples each), all involving constraint satisfaction puzzles. HRM had evaluations on the ARC-AGI task while AKOrN had evaluations on image segmentation tasks, along with the Sudoku which provided further empirical grounding to their claims on generalization. The authors could have picked another evaluation in a different domain like ARC-AGI for further empirical grounding.

### Questions
- Why is the C-voting vs. E-voting comparison (Figure 3) only shown on Sudoku-hard? You have trained AKOrN models and evaluate on Sudoku-extreme and Maze-hard elsewhere. Can you provide the C-voting vs. E-voting comparison on all three benchmarks to support the claim that C-voting outperforms E-voting more generally?
- How well-calibrated are your models, and when does this assumption break down? Equation 14's justification assumes model calibration. Can you provide calibration curves (e.g., reliability diagrams) for your models? Figure 6 suggests Maze-hard has poor calibration - what are the conditions under which C-voting fails?

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
In this submission, the authors present C-Voting, a new method to readout from recurrent neural network-based reasoning models that uses prediction confidence. The proposed method starts with stochastically initializing the first state of an RNN with K candidate states. The RNN architecture is applied to map each of the candidate states to prediction logits. Based on average confidence, the final response of the RNN is chosen to be the output corresponding to that initial state which produces the largest prediction confidence. The authors empirically show that C-Voting outperforms previously proposed methods like E-Voting on a couple of standard reasoning benchmarks (Sudoku and Maze solving). In addition, the authors also propose ItrSA++ -- a new RNN architecture that works effectively with C-Voting and is notably simpler than prior art (HRM and AKOrN).

### Strengths
Strengths:
- The proposed approach is a straightforward, simple intervention to reading out task responses from RNNs. C-Voting is broadly applicable to RNNs applied to reasoning problems with a classification head, as the approach doesn't make any further restrictive assumptions on the network's architecture or training method. 
- Empirically, C-Voting outperforms E-Voting (although some key figures lack estimates of statistical significance) on challenging versions of both Mazes and Sudoku.
- The paper is quite well-written, the explanation of C-Voting and ItrSA++ are accessible to the average reader.

### Weaknesses
Weaknesses:
- A weakness of C-Voting is that the approach might not work effectively when multiple random initializations of an RNN state don't yield  outputs with significantly varied confidence estimates. It is likely that such behavior might arise when training RNNs on increasingly challenging / high-dimensional reasoning problems. Can the authors comment on whether their experiments have explored such a failure mode of C-Voting, and how they intend to circumvent this issue?
- Figures 2-4 showing comparison of C-Voting with other baselines doesn't show performance variance. The authors should report the statistical significance of differences observed between the baselines compared.

### Questions
NA. Please refer to my above review.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes C‑voting, a test‑time voting strategy for recurrent models that generates multiple candidate trajectories by sampling random initial states and then selects the trajectory with the highest average top‑1 probability across all positions. Unlike energy‑based voting (E‑voting), C‑voting does not require an explicit energy function, making it applicable to recurrent models such as HRM and AKOrN. To verify its effectiveness, the paper also proposes a simple attention‑based recurrent architecture, ItrSA++. Empirically, the authors report sizeable gains: HRM with C‑voting improves on Sudoku‑extreme (from 55.0% to 71.2%), C‑voting surpasses E‑voting on Sudoku‑hard when many samples are used (e.g., 94.4 ± 0.1% vs. 89.5 ± 2.5% at 4096 candidates), and ItrSA++ with C‑voting outperforms HRM/AKOrN on Sudoku‑extreme and Maze‑hard. The paper also analyzes how C-voting can fail due to poor calibration of the recurrent model.

### Strengths
* The proposed method is straightforward and easy to use. The voting rule is model-agnostic and only requires per-position class probabilities. It should combine easily with other improvements to recurrent models and yield even better results.
* The proposed method enables a new dimension of test-time scaling. Existing approaches mainly scale the number of iteration steps, which can saturate at large step counts. Here, multiple random initializations provide an orthogonal axis of improvement, and should be easy to be made parallel. Consequently, the proposed method should be able to make good use of the increasing compute power. 
* The reported empirical gains are impressive.

### Weaknesses
* The effectiveness of C-voting relies on the model's calibration , which is not always satisfactory, especially when data are limited.
* The experimental coverage seems to be selective, or at least the design of experiments is not fully explained. Only Sudoku-extreme and Sudoku-hard results are reported for Fig. 2 and 3, respectively. One would typically expect all of Sudoku, Sudoku-hard, Sudoku-extreme and Maze-hard to be included across Fig. 2-4.
* Fig. 2 and 3 report the results from prior works. As the results in DL research can be highly sensitive to implementation details, it can be hard to tell if the improvements actually come from the proposed method.
* Ablations on the effectiveness of ItrSA++ are absent. It is not impossible that the improvements actually come from a better design or implementation of the model rather than C-voting.
* The idea of sampling multiple reasoning paths at test time for better output is long-standing in LLM research, such as [1] and [2]. Given the similarity, the novelty of the method is not significant.

### Questions
* In Fig. 2, why is HRM with C-voting performs exactly identical with random samples from 2 to 16?
* To mitigate calibration issues, I recommend the authors to consider standard calibration techniques. For example, temperature scaling [3] introduces a single scalar temperature for the output logits, which is calibrated on a separate validation set.

[3] Guo, Chuan, et al. "On calibration of modern neural networks." ICML 2017

### Soundness
2

### Presentation
3

### Contribution
2
