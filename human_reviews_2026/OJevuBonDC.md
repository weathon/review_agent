# Forward-Forward Learning with Dynamic Architecture Adaptation for Classification

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 0, 6

## Abstract
The Forward-Forward (FF) algorithm has emerged as a promising alternative to the traditional deep learning paradigm based on the backpropagation algorithm. However, both the original FF algorithm and several FF-based extensions rely on the quality of generated negative samples for training, which can limit their effectiveness. 
In this paper, we design an FF-based algorithm for the classification task. Specifically, we propose the concept of support neuron (SN) sets by partitioning the neurons in each layer into several sets, each explicitly corresponding to a class. The SN set with the strongest response (goodness) determines the predicted class of the input, thereby eliminating the need for negative samples. Furthermore, inspired by the functioning of the brain, we introduce neuron growth and degeneration strategies: (1) when neurons fail to achieve satisfactory performance, new neurons can grow to assist; and (2) neurons that remain inactive across all classes may degenerate. 
Extensive experiments demonstrate that our method achieves state-of-the-art performance on MNIST and CIFAR datasets compared to other FF-based approaches that also eliminate the use of negative samples. In addition, the effectiveness of the proposed neuron growth and degeneration mechanisms is empirically evaluated.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a Forward-Forward (FF)-based learning framework that introduces support neuron (SN) sets to partition neurons by class, thereby eliminating the need for negative samples. Furthermore, the authors design biologically inspired neuron growth and degeneration mechanisms to dynamically adjust network capacity. \

### Strengths
- The paper explores an interesting direction by combining FF learning with dynamic neural structural adaptation. 

- The biologically inspired neuron growth and degeneration mechanisms are interesting for dynamically adjusting network capacity.

### Weaknesses
**Ambiguity in the Combo Training Process (Table 1)**

It is unclear whether the training process using CWC for convolutional layers and PvN for linear layers is conducted sequentially (train convolutional layers first, then linear layers) or simultaneously (both optimized in the same iteration). Moreover, since the proposed method avoids standard backpropagation, it is essential to clarify how parameter updates are coordinated between these two objectives.

**Inconsistencies in Reported Results (Table 2)**

The reported accuracies for FF and CaFo on MNIST appear inconsistent with values presented in the respective original papers. Additionally, FF is only examined on MNIST is not precise. Both FF and CaFo have published convolutional and non-convolutional variants, and original results for CIFAR-10/100 are available.

**Missing Training Dynamics and Visualization **

The paper lacks visual or statistical evidence regarding the training dynamics and representation evolution. For instance, it would significantly strengthen the work to include:

- Plots of training and validation accuracy versus epochs to demonstrate convergence behavior and stability.
- Visualization of response distribution across SN sets over time to support claims about interpretability and neuron specialization.
- Optional activation heatmaps showing how neurons evolve before and after growth/degeneration.


**Theoretical Rationality of Explicit Class-wise SN Assignment**

The core assumption that neurons can be explicitly and statically assigned to specific classes deserves deeper theoretical justification. In realistic scenarios, classes often share overlapping features; thus, forcing neurons to be class-exclusive may restrict feature sharing and harm generalization.

**Minor Comments**
1. The description of  “Combo” training (CWC + PvN) could include pseudocode for clarity.
2. Consider adding figures illustrating neuron growth/degeneration workflows.
3. The transition from the critical analysis to the overall assessment is not sufficiently smooth, which makes the logical flow of evaluation appear somewhat abrupt and may weaken the coherence and persuasiveness of the review.

### Questions
See above.

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
3

### Summary
This paper proposes an extension to the Forward-Forward (FF) learning paradigm for classification. The main innovations are (1) partitioning neurons into class-specific Support Neuron (SN) sets and defining SN-set responses (goodness) as the sum of squared activations for class prediction, and (2) introducing biologically inspired neuron growth and degeneration mechanisms that dynamically adjust network width. The authors evaluate three loss functions, i.e. Positive-vs-Negative (PvN), a channel-wise competitive (CwC) SN-set-wise loss, and Cross-Entropy (CE), and a hybrid training scheme (Combo) that uses CwC for convolutional layers and PvN for the final linear layer. Experiments on MNIST, CIFAR-10, and CIFAR-100 show improvements over other FF-based methods that avoid negative samples (e.g., CFSE and CaFo).

### Strengths
Originality: Extending channel/grouping ideas to class-specific Support Neuron (SN) sets and integrates neuron growth/degeneration.

Clarity of core idea: SN-set response (sum of squared activations) and the LLR/OLR prediction criteria are clearly formulated.

Practical relevance: Eliminating negative samples addresses a practical limitation of many FF variants and may simplify implementations for resource-constrained settings.

Empirical promise: The Combo training strategy shows competitive results compared to other FF-based approaches in the paper.

### Weaknesses
Motivation: The proposed strategies, e.g. neuron growth and degeneration, seems to be ad hoc, i.e. it lacks a rigorous motivation or toy experiment to validate them.

Baseline fairness: The reported DNN baseline (65.4% on CIFAR-100) appears a little low for comparable architectures. The authors should clarify the DNN architecture, training recipe, and whether data augmentation or other standard techniques were used.

Limited comparison scope: Important recent FF variants and stronger BP baselines (beyond ResNet18 and a single DNN) are not included; this weakens the claim of its superiority.

Insufficient justification for SN response metric and hyperparameters: The square-sum choice needs empirical comparisons (e.g., L1, max, soft-assignment) and sensitivity analysis.

Growth/degeneration analysis: The degradation experiments show large and non-monotonic changes (e.g., removing two neurons sometimes collapses accuracy). Authors should analyze per-neuron contribution and report variance across different random seeds.

Reproducibility: Some low-level training details (exact optimizer schedules, seed handling, and implementation variants for baselines) should be moved into the main text or a reproducibility checklist.

### Questions
1. Could the motivation of the some strategies be presented in a more rigorous way ?

2. Please provide the exact DNN baseline architecture and full training recipe for CIFAR-100 (data augmentation, optimizer, learning-rate schedule, number of runs), since the proposed baseline is a little low. 

3. Please analyze the neuron removal experiments more thoroughly. For the cases where removing two neurons causes a disproportionate drop, can the authors (a) report per-neuron activation statistics, (b) visualize learned filters or class-specific activations, and (c) show results averaged across multiple random seeds?

4. Have the authors tried alternative SN-set response metrics (e.g., L1 norm of activations, max pooling within channels, cosine similarity on pooled features) ?

5. Please include additional baselines: (i) stronger BP baselines (well-tuned DNN/ResNet variants), and (ii) recent FF variants such as DeeperForward/Distance-Forward if implementations are available.

6. Please provide a sensitivity study for key hyperparameters: thresholds for PvN, selection rules for growth, number of neurons added per growth step, and degeneration thresholds.

7. How about the robustness of the Combo scheme (CwC for conv layers + PvN for linear layers) across architectures and dataset ? An ablation would help.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a forward-only, no-negatives training scheme that partitions each layer into per-class Support Neuron (SN) sets and defines “goodness” as sum of squared activations per set. It studies losses on per-class goodness, two prediction rules (LLR, OLR), and introduces neuron growth and neuron degeneration to adapt width. Results are shown on MNIST, CIFAR-10, CIFAR-100, compared to CFSE (Papachristodulou et al, 2024) and CaFo (Zhao et al, 2023) (forward methods without negatives) and to BP baselines.

### Strengths
- Width adaptation idea (growth and degeneration) is interesting and could be impactful with deeper evaluation.

### Weaknesses
- Conv-layer partitioning into sets is a mechanism already introduced in prior work CFSE (Papachristodulou et al, 2024). The paper acknowledges this but still lists this set partitioning as a core contribution. In my opinion the genuine novelty is the linear-layer extension and width adaptation. In addition, all three losses have already been introduced in prior work with the same name (Papachristodulou et al, 2024). The PvN and CwC as used here are already defined in CFSE (Papachristodulou et al, 2024) for channel groups (CFSE Eqs. 5–6). The manuscript should be explicit about reuse and position its difference. As it stands the mathematical formulation is the same. 
- The paper’s CwC equals CE applied to per-class goodness logits (g_l). The CE section repeats the same. This duplication confuses the contribution and makes the loss study harder to interpret. 
- Section 3.7 claims training each layer “directly on the entire dataset” and that optimization is “essentially closed-form.” The appendix algorithm trains with random mini-batches using SGD/Adam, i.e., not closed-form nor full-batch. 
- Only last-layer edits, small additions/deletions, and sensitivity (including degradations) are shown. Authors note single runs with fluctuating results in some cases.
- Reported gains over CFSE/CaFo are rather incremental (e.g., CIFAR-10 80.5% vs CFSE 77.2%; CIFAR-100 52.0% vs CFSE 48.9%). This is fine, but it does not show a qualitative jump. 
- The paper states “architectures are kept consistent” in Table 2 comparisons, but also evaluates CaFo with a revised architecture (Re). This change should be labeled carefully. 
- The paper lacks comparison and/or discussion with standard local-learning and FF-inspired methods beyond CFSE/CaFo. In particular FA, DFA, DRTP, SoftHebb. These are canonical non-BP or local-error baselines that target similar motivations (forward compute, cheap feedback, biological plausibility). The paper is missing a comparison with at least one method per family or provide a documented rationale if direct runs are infeasible.

### Questions
- Can you report some features of the different models, such as parameters, FLOPs, and memory for your method vs CFSE/CaFo/BP for better comparison?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an enhancement to the standard Forward-Forward (FF) algorithm designed specifically for classification tasks. It addresses a key limitation of the original FF model and its reliance on high-quality negative samples. They propose the concept of Support Neuron (SN) sets and integrating Dynamic Architecture Adaptation using grow and prune neurons concept. The SN sets partition neurons in each layer to explicitly correspond to class concepts, thus enforcing local class-specific representations and improving feature discrimination compared to generic FF variants.

### Strengths
- Directly solves a major weakness of the original FF algorithm, making the training process more robust and easier to implement without complex negative sample generation strategies.

- The explicit partitioning of neurons into class-specific SN sets creates a more structured and potentially more interpretable feature space, as one can directly analyze the activity of neurons associated with a particular class.
- The paper is well written and easy to follow.

### Weaknesses
- Computational overhead of the Dynamic Architecture Adaptation is unknown compared to the standard FF models. 
Provide a detailed analysis of the average time increase per training epoch (or step) compared to a standard, fixed-architecture FF model with the same number of layers and total neurons.


- The core concept relies on partitioning neurons into explicit class-based sets. This is straightforward for simple classification (e.g., CIFAR, MNIST) but breaks down for tasks without fixed, discrete classes limiting the approach's general applicability. How easy or difficult to extend this beyond classification is not known.

- The rigid, class-specific partitioning of neurons into SN sets, while enhancing interpretability, might constrain the model's ability to learn efficient, shared representations. For fine-grained classification, this forced disentanglement could be less efficient than a flexible, dense layer that naturally compresses shared features across multiple classes. Provide emprical results for fine grained classification and use Centered Kernel Alignment (CKA) to quantify the similarity between the feature representations of two highly related classes (e.g., two different bird species or types of cars) in the penultimate layer. Compare the CKA score between the Dynamic FF model and a standard, dense FF model.

- While the Support Neuron (SN) sets are designed to mitigate the reliance on high-quality negative samples, the paper might not eliminate the need for them entirely.  Compare the Dynamic SN-FF model's performance against the standard FF baseline when both are trained using trivial, low-quality negative samples (e.g. completely random noise or samples drawn from a different distribution).

- Usually, dynamically adjusting the architecture introduces instability during training. Any insights or detailed discussion regarding the model training will be useful.

### Questions
See Weakness section

### Soundness
4

### Presentation
3

### Contribution
3
