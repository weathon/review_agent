# Beyond Unidirectional Flow: LLM Reasoning with Bidirectional Cycle-Consistent CoT

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Small-large model collaboration is a promising approach for efficient reasoning, where lightweight assistant models generate intermediate representations to guide larger, more capable models. However, this paradigm encounters two key challenges: \textbf{representation heterogeneity} between different model architectures and \textbf{unidirectional information flow} that prevents mutual learning. 
Small assistant models and large base models develop distinct geometric structures for encoding similar concepts, making direct alignment difficult and leading to information degradation. 
Additionally, unidirectional flow creates asymmetric dynamics where assistant models cannot benefit from large models' superior representational capacity. 
We introduce \textbf{CycleCoT}, a bidirectional framework that addresses these bottlenecks through cycle-consistent soft thought alignment. Our approach uses dual residual transformation networks to establish invertible mappings between heterogeneous model spaces through three mechanisms: (1) expressive mappings between different model representations, (2) bidirectional alignment objectives enforcing semantic consistency in both directions, and (3) cycle consistency constraints preserving information during round-trip transformations. This enables large models' knowledge to enhance assistant models' soft thought generation, creating symbiotic collaboration. Evaluation on LLaMA-$3.1$-$8$B-Instruct and Qwen$2.5$-$7$B-Instruct across mathematical, commonsense, and symbolic reasoning benchmarks demonstrates consistent improvements over unidirectional baselines, with gains up to $5.5\%$ on mathematical reasoning tasks. 
Our analysis reveals that alignment quality surpasses quantity: fewer, well-aligned soft thoughts outperform longer sequences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CycleCoT, a bidirectional, cycle-consistent framework designed to enable efficient and robust reasoning in large language models through collaboration with small auxiliary models. The core idea is to overcome the challenges of representation heterogeneity and unidirectional information flow present in prior work by introducing dual residual projection networks. These networks map soft thought vectors between the auxiliary model space and the base model space, regularized via bidirectional alignment and cycle-consistency losses. Comprehensive experiments using LLaMA and Qwen model families on mathematical, commonsense, and symbolic reasoning tasks demonstrate significant performance improvements over strong baselines, particularly in mathematical reasoning.

### Strengths
1. This paper clearly identifies two major bottlenecks in small-model collaboration for LLM reasoning: representation heterogeneity and unidirectional information flow, thereby motivating the need for a bidirectional approach.

2. The CycleCoT framework is conceptually inspired by cycle consistency and dual learning, and is concretely instantiated through a residual projector architecture with explicit mapping, alignment, and reversibility constraints.

### Weaknesses
1. The paper lacks evaluation on more challenging mathematical benchmarks, such as AIME24, AIME25, or MATH500, which would better demonstrate the robustness and generalization of the proposed method.

2. The experiments do not include results on larger and more powerful base models, such as Qwen3-8B-Base or Qwen2.5-32B-Base, limiting the assessment of the framework’s scalability and effectiveness across model scales.

3. The comparison with stronger baseline methods is insufficient. In particular, the paper should include results from supervised fine-tuning (SFT) and reinforcement learning approaches (e.g., GRPO, DAPO) to provide a more comprehensive and competitive evaluation.

4. The paper does not discuss or compare with recent representation-enhanced Chain-of-Thought approaches, such as:

[1] Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models

[2] Amplify Adjacent Token Differences: Enhancing Long Chain-of-Thought Reasoning with Shift-FFN

[3] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering

[4] Logit Arithmetic Elicits Long Reasoning Capabilities Without Training

These works are highly relevant and should be included in both the related work section and experimental comparisons to better position the proposed method within the current research landscape.

### Questions
Could the authors provide a case study or qualitative analysis to better illustrate where the proposed method demonstrates its advantages? For instance, how does the bidirectional reasoning process with cycle consistency concretely improve the quality or coherence of the generated thoughts compared to existing approaches?

### Soundness
2

### Presentation
2

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
The paper proposes CycleCoT, a bidirectional, cycle-consistent framework for coordinating a small assistant model and a larger base model in latent Chain-of-Thought reasoning. It aligns the models’ heterogeneous hidden spaces using dual residual projectors with alignment and cycle-consistency losses; at inference, only the forward mapping is used, so runtime matches prior SoftCoT methods. Empirically, pairing a 1B assistant with LLaMA-3.1-8B and Qwen2.5-7B yields consistent gains on GSM8K, ASDiv-Aug, AQuA, StrategyQA, CommonsenseQA, and Date Understanding (about 2.5–6% on average, strongest on math). Ablations show each component is necessary, and analysis finds that a few well-aligned soft thoughts (around 2–4) outperform longer sequences. Overall, CycleCoT turns the large model from a passive consumer into an active teacher, enabling symbiotic small–large model collaboration with minimal inference overhead.

### Strengths
1. Unlike SoftCoT/CoCoNut-style pipelines that project assistant thoughts **only forward** into the base model’s space, CycleCoT introduces **bidirectional, cycle-consistent** alignment—reducing representation mismatch and enabling feedback from the large model, a design not explored in those works. 

2. The paper backs its claims with broad, controlled experiments and ablations across model families and task types, while preserving SoftCoT-like **inference cost** by using only the forward map at test time—making adoption straightforward in existing latent-CoT pipelines.

3. Bringing **cycle consistency** into LLM representation alignment leverages a proven principle from cross-domain mapping (e.g., CycleGAN) and dovetails with recent latent-CoT developments (e.g., SoftCoT/SoftCoT++), positioning the method as a simple drop-in that complements ongoing test-time scaling work.

### Weaknesses
1. The paper repeatedly invokes geometric/representation heterogeneity but does not give a precise, self-contained definition or diagnostic. *Actionable:* formalize the notion (e.g., specify the object being compared, the metric family, and the hypothesis about how mismatch harms reasoning), provide standard diagnostics and visualizations, and state testable predictions that link reductions in the defined mismatch to downstream gains. 

2. Because the geometric notion is not operationalized, it is unclear whether the chosen benchmarks (mostly QA-style reasoning with 7–8B bases and 1B assistants) are the right stress tests or how conclusions transfer to planning, tool use, or long-form generation. *Actionable:* broaden tasks and scales, and—crucially—tie results back to the formal diagnostics in(e.g., show that tasks with larger measured mismatch benefit more, and that improvements track the proposed metrics). 

3. While inference overhead matches SoftCoT (forward map only), the training cost and stability of dual projectors are not quantified, and the indispensability of the cycle-consistency term is not established beyond limited ablations. *Actionable:* report training FLOPs/wall-clock/memory, add stability statistics, and include targeted ablations (weaken-cycle, reverse-only, invertibility constraints) to demonstrate necessity rather than generic regularization effects. )

### Questions
# Questions*

1. How do you formally define the “geometric structure” mismatch between models, how is it computed, and how do you visualize it to support the claimed link to reasoning gains?

2. **(Lines 180–192):** How do you ensure the 1B assistant’s rationales/soft thoughts are of sufficient quality, and how sensitive are results to the small-LLM instruction template and its diversity?

3. **(Figure 3):** Are you only training the MLP projector layers while keeping LLM_A and LLM_B frozen, and if so, how do you ensure the stated geometric-structure alignment holds without updating the backbones?

4. **(Figure 4):** Since cosine similarity appears to drive the gains while other metrics can hurt performance, does this imply the benefits stem mainly from bidirectional semantic alignment rather than the cycle-consistency objective?

### Soundness
2

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
3

### Summary
In this paper, the authors proposed techniques that advance latent CoT reasoning by introducing a more symmetric, self-consistent interaction between models. The approach is conceptually interesting and shows potential to improve reasoning capability and interpretability.

### Strengths
This paper presents an interesting and well-motivated approach for bidirectional alignment in the collaboration between large and small models. The proposed cycle-consistent soft thought alignment effectively establishes invertible mappings across heterogeneous model spaces, enabling deeper interaction and mutual understanding between models. Moreover, the experimental study is thorough and convincing, demonstrating the effectiveness and practical value of the proposed design.

### Weaknesses
The paper should provide more justification for why a two-layer projector is sufficient to ensure high-quality mapping across models.
It is unclear how essential continuous tokens are to the proposed design—whether they are fundamental to the framework or primarily used to enhance performance.

Figure 2 lacks clarity, as the meaning of the purple arrow is not explained.

In Equation (7), it remains unclear whether the transformation could alternatively occur on the base model side (e.g., using T_B'').
The necessity of the loss term in Equation (7) is not fully explained, especially if minimizing the loss in Equation (6) might already suffice. Relatedly, it is worth clarifying whether T_A'' could be replaced with T_B' .

In the efficiency analysis (Figure 5), the discussion does not clearly quantify the trade-off between performance gains and the additional computational cost introduced by the projection and alignment steps.

### Questions
see the weaknesses section

### Soundness
2

### Presentation
3

### Contribution
2
