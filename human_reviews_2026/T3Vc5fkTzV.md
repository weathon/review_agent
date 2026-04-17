# KeepLoRA: Continual Learning with Residual Gradient Adaptation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Continual learning for pre-trained vision-language models requires balancing three competing objectives: retaining pre-trained knowledge, preserving knowledge from a sequence of learned tasks, and maintaining the plasticity to acquire new knowledge. This paper presents a simple but effective approach called KeepLoRA to effectively balance these objectives. We first analyze the knowledge retention mechanism within the model parameter space and find that general knowledge is mainly encoded in the principal subspace, while task-specific knowledge is encoded in the residual subspace. Motivated by this finding, KeepLoRA learns new tasks by restricting LoRA parameter updates in the residual subspace to prevent interfering with previously learned capabilities. Specifically, we infuse knowledge for a new task by projecting its gradient onto a subspace orthogonal to both the principal subspace of pre-trained model and the dominant directions of previous task features. Our theoretical and empirical analyses confirm that KeepLoRA balances the three objectives and achieves state-of-the-art performance. The implementation code is available at https://github.com/MaolinLuo/KeepLoRA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focuses on balancing three core objectives in continual learning (CL) for pre-trained vision-language models (VLMs): preserving pre-trained general knowledge, retaining sequential task knowledge, and maintaining new task plasticity. Via SVD analysis, it finds general knowledge resides in the parameter principal subspace, while task-specific knowledge is in the residual subspace. Based on this, KeepLoRA restricts LoRA updates to the residual subspace—by projecting new task gradients onto the subspace orthogonal to the pre-trained principal subspace and previous task dominant directions—and initializes LoRA via purified gradients. Theoretical proofs and MTIL benchmark experiments (11 image classification tasks) confirm its state-of-the-art performance on Transfer, Average, and Last metrics.

### Strengths
- Insightful subspace analysis: The separation of general/task-specific knowledge across principal/residual subspaces provides a clear, interpretable basis for CL design.

- Theoretical rigor: Propositions 3.1 and 3.2 mathematically validate that KeepLoRA’s updates ensure stability and align with optimal plasticity.

- Practicality: Parameter-efficient (only updates LoRA’s B matrix, no inference overhead), data-independent (no replay/reference data), and robust across task orders.

- Comprehensive evaluation: Uses well-defined metrics (Transfer/Average/Last) and ablation studies to verify component contributions.

### Weaknesses
1. KeepLoRA’s design assumes that "residual subspace updates for new tasks do not interfere with old tasks," but it does not analyze how task similarity affects this assumption: For highly similar tasks (e.g., two fine-grained classification tasks like StanfordCars and Aircraft), their residual subspaces may overlap, potentially leading to interference. 

2. Scalability gaps: Unaddressed feasibility of SVD and subspace scaling for large VLMs (e.g., CLIP-L) with massive parameters.

### Questions
1. Some typographical errors. For example, "KeppLoRA" in table 2 and row 816.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces KeepLoRA, a novel method for continual learning (CL) in pre-trained vision-language models designed to balance plasticity, backward stability, and forward stability. The authors of this study found a key insight by looking at how a model stores information, discovering that general, pre-trained knowledge is stored in the principal subspace, while specific skills are stored in the residual subspace. Motivated by this, KeepLoRA learns new tasks by restricting its Low-Rank Adaptation (LoRA) parameter updates to this residual subspace, where it essentially keeps the parameters from the principal subspace and everything it learned from past tasks, ensuring the new information is written only into the residual subspace. This approach prevents interference with existing knowledge, and the paper's theoretical and empirical analyses confirm that KeepLoRA effectively balances the three CL objectives to achieve state-of-the-art performance.

### Strengths
1. State of the Art performance: The paper demonstrates that KeepLoRA and its variant, KeepLoRA+, achieve state-of-the-art results on the MTIL benchmark, outperforming previous methods on all key metrics

2. Strong Empirical Analysis: The paper's core hypothesis is based on a clear, intuitive analysis of the model's parameter space. The finding that general knowledge resides in the principal subspace while task-specific knowledge is in the residual subspace provides a solid foundation for the method.

3.Theoretically Grounded: The authors provide a strong theoretical justification for their approach. They prove that their method of initializing and freezing the LoRA matrix At is an optimal solution (Proposition 3.2) to the problem of maximizing adaptation to the new task while remaining perfectly orthogonal to all previously learned knowledge.

### Weaknesses
1. Limited Evaluation on Language: The paper focuses on Vision-Language Models (VLMs), but the evaluation is performed on a benchmark of 11 image classification datasets. The "language" aspect is only used for zero-shot classification via class names. The method's effectiveness on more complex, language-heavy VLM tasks is unproven.

2. Your method, by design, projects new task gradients to be orthogonal to the principal subspace and all previous task directions to ensure stability. Does this strict orthogonality constraint inadvertently prevent positive forward transfer, where the model should be re-using and building upon those similar, previously-learned features, rather than being forced to find a completely new, orthogonal direction to learn?

### Questions
see weaknesses

### Soundness
3

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
This paper analyses the parameter space using SVD and finds that the principal subspace encodes general knowledge and the residual subspace encodes domain-specific knowledge. The paper proposes KeepLoRa which projects the gradients onto a subspace
orthogonal to both the principal subspace of pre-trained model and the dominant directions of previous task features. Theoretical analysis and empirical results on MTIL are provided.

### Strengths
1. KeepLoRa works with pre-trained models addressing three competing objectives: maintaining the ability to learn new knowledge (plasticity), preventing the forgetting of previously learned tasks (backward stability), and preserving the general pre-trained knowledge that guarantees general transferability (forward stability).
2. KeepLora can be implemented in a relatively straightforward manner.
3. Sensible theoretical analysis and strong empirical performance.

### Weaknesses
1. KeepLora+ outperforms KeepLora but the first mention of KeepLora+ is in Table 2 and $4.1.
2. KeepLora requires storing dominant singular vectors from tasks M but this is not listed in Table 2 nor analysed elsewhere.
3. KeepLora introduces epsilon_w, epsilon_f, r, alpha hyper parameters but only the sensitivity of epsilon_w(vision) and epsilon_w(text) is presented.

### Questions
1. What is the computational and memory overhead KeepLoRa and keepLoRa+?
2. What is the sensitivity of all hyper parameters?

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
4

### Summary
The paper addresses continual learning (CL) in pre-trained vision-language models (VLMs) by balancing plasticity and stability. It posits that general knowledge resides in the principal subspace of model weights, while task-specific knowledge lies in the residual subspace. Building on this idea, KeepLoRA confines new task updates to the residual subspace by projecting gradients orthogonally to a unified principal subspace and initializing a frozen LoRA adapter. Supported by theoretical analysis, KeepLoRA achieves comparable performance on the MTIL benchmark.

### Strengths
1. Clear  idea of separating general (principal) and specific (residual) knowledge in the parameter space.


2. Provides strong theoretical justification connecting the method to optimal, constrained gradient descent.

3. Good practicality as it adds no inference overhead, unlike architecture-extension methods.

### Weaknesses
1. Unclear if the unified principal subspace, which accumulates past task directions, can scale to a large number of tasks without prohibitive cost.

2. The method requires expensive per-task full-gradient computation and a one-time full-model SVD, which are not fully benchmarked.

3. Relies on crucial hyperparameters (e.g., $\epsilon_w$, $\epsilon_f$) whose robustness and sensitivity are not deeply analyzed.

### Questions
Refer to Weaknesses for related questions.

### Soundness
3

### Presentation
3

### Contribution
3
