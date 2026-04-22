# UnRe: Zero-Shot LLM Unlearning via Dynamic Contextual Retrieval

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Inference-time machine unlearning with only the forget data, also known as zero-shot unlearning, is becoming increasingly important for bias mitigation, privacy preservation, copyright protection, etc. Most approaches in this domain focused on query updating, decoder modification, offline module training, or reverse-generation by the forget data. Recent works found that providing offline-prepared contexts can realize in-context unlearning. However, leveraging dynamic context (conditioned on real-time queries) to achieve zero-shot unlearning has not yet been explored, which has the potential to enforce context unlearning while preserving the performance of the original LLM. In this paper, we propose UnRe, a novel unlearning framework for LLMs that employs dynamic contextual retrieval from retrieval-augmented generation (RAG) while only leveraging the forget data. Specifically, UnRe dynamically updates contexts to guide the unlearning process in a zero-shot setting. During the inference, the user query is first leveraged for online membership inference to identify a query-specific forget set. Using this set, UnRe refines the embeddings of the retrieved chunks via gradient descent, producing adaptive contexts that steer the LLM toward a query-specific unlearned distribution. We evaluate UnRe on multiple unlearning benchmarks and show that UnRe not only outperforms existing zero-shot and context-based unlearning approaches, but also preserves the original model performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an inference-time machine unlearning approach. The core idea is to apply gradient updates in the embedding space so that, during generation, the LLM avoids producing outputs that overlap with the forget data. The authors compare their approach against various baselines and report improved performance.

### Strengths
- Efficient and flexible unlearning is an important problem.
- The idea of dynamic context editing is interesting.
- However, many technical details are missing, and the writing requires substantial improvement.

### Weaknesses
1. Clarity: The paper is very difficult to follow. Variables are introduced without clear definitions, and the exposition is often confusing (see questions below).
2. Technical detail: Key details about how dynamic generation and context updates work are missing.
3. Novelty: It is unclear how the method differs from or advances standard RAG-based unlearning.
4. Efficiency: The method appears to require LLM generation, then RAG, then gradient updates, followed by another generation pass. This will likely slow inference substantially, which may make the approach impractical for real-world applications.

### Questions
- Figure 1 is very difficult to interpret. The color coding is unclear, font sizes and families are inconsistent (even in the legend), and the flow alternates between left-to-right and right-to-left/top-to-bottom. The meaning of dashed versus solid arrows is not explained. It is also unclear where the LLM is involved: there is an arrow from the language model box to (3→4), but step 4 is labeled “retrieval” rather than “generation.” The figure does not clearly indicate which components are off-the-shelf (embedding model, LLM, etc.), and the legend uses vague categories such as “language model,” “embedding space,” and “offline preparation.” Overall, the figure looks unpolished and does not meet the clarity expected for an ICLR paper. Please revise for readability, consistent notation, and a coherent flow.
- Citations are not consistently formatted in LaTeX. In many places, citep should be used instead of citet. For example: “operates during LLM inference with frozen weights and is generally regarded as suppression-intended unlearning (Ren et al., 2025).”
- Line 167: What are the input x and label y in the forget piece? This is the first mention of labels in the paper. Why is a label needed here? Is x the content to forget? If so, why do we need y?
- Line 171: The perturbed set \tilde{O}_q is introduced without sufficient context. What is its purpose at this stage, and how does it feature in the problem formulation?
- Line 173: Using M.G to denote the LLM’s generation process is unconventional. Please align the notation with standard practice.
- Lines 186–188: The connection between performing gradient descent in V_R (the embedding model space) and imposing constraints on the LLM’s output distribution is not clear. Please provide a formal justification or an empirical rationale that links embedding-space updates to output distribution shifts.
- Line 196: φ is used to denote the embedding model’s forward pass, but the embedding model itself is not clearly specified or defined. Please clarify the model choice, training status, and interface.
- Section 3.5: Why is the LLM hidden state h_δ(t) directly comparable to the embedding space produced by φ? The last hidden state of an LLM is typically optimized for next-token prediction rather than capturing the global semantics of the answer. Please justify this assumption or provide empirical evidence.
- Technical details on how the perturbation matrix is used in inference are missing. Is this matrix applied to all layers and all tokens of the transformer, or only to specific layers/states? Please provide a precise description of the application mechanism, scope, and computational overhead.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces UNRE, a framework for zero-shot, inference-time unlearning in large language models (LLMs) using dynamic, query-conditioned contextual retrieval. Unlike previous methods requiring retain sets or offline-trained modules, UNRE operates solely on a forget set and employs retrieval-augmented generation (RAG) with online membership inference to identify, adapt, and perturb context embeddings in response to real-time queries, aiming to suppress memorized or undesirable content while retaining overall model utility

### Strengths
1. The proposed method transforms the unlearning problem into a context-level intervention, avoiding the costly retraining required for parameter-level unlearning.

2. The theoretical and algorithmic design is simple yet effective, achieving context switching between forgetting and retention through embedding-space perturbation.

3. UNRE is evaluated on multiple public benchmarks, including entity unlearning and copyright unlearning, demonstrating strong experimental performance.

### Weaknesses
1. The method uses a similarity threshold ( \tau ) to decide whether to trigger unlearning based on the similarity between the model output ( y_q ) and the forget set ( O ). Since ( \tau ) is the key factor determining whether unlearning occurs, treating this decision as a binary classification may be oversimplified. Its value can be sensitive to the model, task, and data, and the paper does not explain how it is chosen.

2. The loss ( \mathrm{softplus}(N - S) ) aims to balance semantic preservation (high ( S )) and output divergence (low ( N )), which are conflicting goals. If ( S ) dominates, forgetting may be incomplete; if ( N ) dominates, the output may drift semantically. A tunable hyperparameter could help balance these effects.

3. The Projected Gradient Descent (PGD) optimization may fall into local minima or become unstable in high-dimensional embedding spaces. Exploring contrastive-learning-based optimization or low-dimensional perturbation approximations could improve stability.

4. Each unlearning step requires document retrieval, multiple PGD iterations, and re-decoding, which significantly increases inference latency. This may limit the method’s practicality in real-time applications.

5. As a context-level approach, the method may be more vulnerable to jailbreak or adversarial attacks, where an attacker could craft prompts to bypass unlearning and recover forgotten content.

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes UNRE, a zero-shot inference-time unlearning framework for large language models (LLMs). Unlike prior approaches that require retain sets, offline fine-tuning, or fixed unlearning contexts, UNRE leverages query-adaptive dynamic retrieval based on the forget set only. Specifically, it integrates an online membership inference module to identify query-related forget pieces, then applies gradient-based refinement of retrieved embeddings to steer model outputs toward an “unlearned” distribution.

### Strengths
- The paper explores a new setting of zero-shot inference-time unlearning with dynamic retrieval, extending prior in-context unlearning (ICUL) work.

- Unlike typical in-context learning methods, it operates at the embedding level to prevent the activation of information associated with the forget set.

- The motivation is well-aligned with privacy and copyright unlearning, and the approach explicitly avoids model parameter modification, retain data, or retraining, making it highly practical for real-world deployment scenarios.

### Weaknesses
- The paper could benefit from qualitative examples showing cases where UNRE fails (e.g., borderline membership detection, overly aggressive context updates) to better interpret its robustness.

- Some improvements (e.g., small PPL changes) might result from threshold or step-size tuning in Algorithm 1. Sensitivity analyses on τ and ε would clarify stability and generalization.

- The embedding-level perturbation may risk distorting semantic representations for complex queries. While PGD constraints are mentioned, an explicit analysis of semantic preservation vs. forgetting trade-off is missing.

### Questions
1. There is an incorrect bolded value in Table 1 under the Falcon3-7B-Instruct setting, where Prompt FQ (0.0970) is actually much higher than UnRE (0.0611). In addition, for the same setting, it is unclear why the MU score of UnRE (0.0644) is significantly lower than that of the other methods  (0.66).

2. Could you please bold the best value in Table 2 for clarity?

3. How sensitive is UNRE to the similarity threshold (τ) and update budget (ε)?

4. Could the authors provide ablation studies showing the contribution of each component — membership inference, gradient-based update, and semantic loss — to the overall unlearning performance?

### Soundness
2

### Presentation
1

### Contribution
2
