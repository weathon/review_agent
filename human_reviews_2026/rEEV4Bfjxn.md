# On structured sparsity and dual lottery tickets for Robust Continual Multi-task Learning

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Continual learning for large language models (LLMs) faces a critical challenge: adapting to new tasks often results in catastrophic forgetting of prior knowledge and destructive interference across tasks. While sparse adaptation methods, such as Lottery Ticket Adaptation (LoTA), have emerged to mitigate these issues by optimizing only sparse subnetworks, they often rely on data-dependent mask calibration or random pruning. LoTA, for instance, identifies sparse subnetworks to avoid destructive interference and enables model merging, demonstrating improved performance over full fine-tuning (FFT) and low-rank adaptation (LoRA) in multi-task scenarios. Its extension, LoTTO, further enhances sequential training by learning mutually sparse masks to prevent overlap between tasks.

Building upon these insights, our work introduces a novel approach for robust continual multi-task adaptation, specifically designed to achieve high accuracy on two or more tasks (A and B) without catastrophic forgetting. Our technique distinguishes itself by first selecting subnetworks based on inherent structural properties using expander graph masks, rather than relying on data-dependent or purely random selection. These expander masks provide a principled and structurally sound basis for defining initial sparse subnetworks. Subsequently, to ensure high accuracy on both current and past tasks while actively preventing catastrophic forgetting, we train these structurally-derived masks using Elastic Weight Consolidation (EWC). EWC selectively regularizes the parameters deemed important for previously learned tasks, thereby preserving critical knowledge and enabling efficient adaptation to new objectives.

This combined methodology not only yields demonstrably higher scores across multiple tasks but also offers a compelling multi-task extension of the Dual Lottery Ticket Hypothesis (DLTH). In this context, we claim that any two random expander masks can be transformed into highly trainable subnetworks, achieving high degrees of accuracy on distinct tasks. Our approach provides a powerful and efficient framework for robust continual learning in LLMs, addressing the core challenges of destructive interference and catastrophic forgetting through structured sparsity and intelligent knowledge preservation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a continual learning framework that combines structured sparsity (via expander graph masks) with Elastic Weight Consolidation (EWC) to improve robustness against catastrophic forgetting. The authors argue that expander graphs provide a principled, data-independent way to define sparse subnetworks, while EWC regularization preserves task-critical knowledge. The method is empirically evaluated on LLaMA-3-8B and RoBERTa-Base across multiple tasks and claims to extend the Dual Lottery Ticket Hypothesis (DLTH) to multi-task scenarios.

### Strengths
1. The paper addresses an important problem — catastrophic forgetting in continual learning for LLMs.
2. The combination of structural sparsity and regularization (EWC) is conceptually reasonable and aligns with theoretical intuitions in network theory.
3. The theoretical section attempts to connect expander-based masks to probabilistic continual learning formulations and provides analytical bounds.
4. The experiments include both large-scale (LLaMA-3-8B) and smaller (RoBERTa-Base) models, which is a good choice for verifying generality.

### Weaknesses
1. Lack of expressiveness and visual explanation. The paper contains no figures or conceptual diagrams, which makes it hard for readers to grasp the architecture, workflow, or the role of expander masks intuitively.

2. Limited novelty. The core components (EWC and expander-based sparsity) are both existing techniques. The combination is incremental, and the work does not substantially innovate beyond baseline LoTTO according to Table 2.

3. Insufficient methodological clarity. The procedure for constructing and applying the expander masks is not described in sufficient algorithmic detail. It is unclear how graph generation interacts with model layers or parameters.

4. Experimental analysis is weak. 
    a. Tables are presented without qualitative or statistical analysis. 
    b. Table 1 only lists raw numbers with no discussion on reasons why the proposed method outperforms other baselines .
    c. In Table 2, performance is roughly comparable to LoTTO, and the improvements are minor.
    d. No ablation studies are provided to isolate the effects of EWC, mask overlap, or Cheeger constant.

### Questions
1. Could the authors provide a detailed algorithm or pseudocode illustrating how expander masks are generated and applied layer-wise?
2. How sensitive is performance to the sparsity ratio or the choice of graph parameters (e.g., degree d, λ)?
3. Have you compared against more recent continual learning baselines？
4. Can you provide ablation results showing the contribution of EWC versus expander masks alone?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript proposes a continual learning (CL) framework that combines structured, data-independent "expander graph masks" for task separation with Elastic Weight Consolidation (EWC) for knowledge preservation. The authors claim this method mitigates catastrophic forgetting in LLMs and serves as a multi-task extension of the Dual Lottery Ticket Hypothesis, presenting experiments on Llama-3-8B and RoBERTa.

### Strengths
The paper's primary strength is its interesting conceptual synthesis, combining data-independent structural sparsity (expander graphs) with classic regularization (EWC) to tackle the stability-plasticity dilemma. The authors attempt to ground this in CL theory and provide ablation (Table 4) showing that expander masks outperform random masks, supporting the premise that mask structure is important.

### Weaknesses
1. Key continual learning results (e.g., Table 3, Table 5) report numbers only for the proposed method, completely omitting comparisons to sota methods. This makes it impossible to evaluate the method's performance on forgetting or plasticity.

2. The paper evaluates its LLM experiments on a self-defined sequence of "capability" tasks. These are not standard benchmarks in the CL literature, making it difficult to compare the method's performance against the vast body of existing CL research, which typically uses established task sequences.

3. The method is a combination of two existing techniques (expander graphs, EWC), and the theoretical justification relies on tenuous leaps (e.g., from single-layer linear models to deep LLMs) that feel post-hoc.

### Questions
Refer to Weaknesses for related questions.

### Soundness
3

### Presentation
2

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
This paper proposes a framework that combines structured sparsity using expander graph masks with Elastic Weight Consolidation (EWC) to enable robust continual multi-task learning in large language models (LLMs). The key idea is that expander-based masks provide data-independent structured sparsity that preserves gradient flow and connectivity, while EWC constrains important weights to mitigate catastrophic forgetting. The approach is validated on the Meta-Llama-3-8B model and RoBERTa Base, showing improvements over prior methods like LoRA, LoTA, and LoTTO across multiple tasks such as reasoning, math, safety, and instruction following.

### Strengths
1. Novel combination of techniques: The paper introduces a new combination of structured sparsity via expander graph masks and EWC regularization for continual learning. This is a principled approach that uses principles from graph theory to introduce sparsity and differs from prior sparse adaptation methods which relied on random or data-driven mask selection. Coupling this with EWC is a good way to protect previous task knowledge. This dual strategy is well-motivated and to my knowledge original in the context of LLM Continual Learning.
2. Theoretical Grounding: The paper provides a theoretical justification for the approach. It derives a bound on the forgetting of a previous task A after learning a new task B, showing that the increase in Task A’s loss is bounded by terms proportional to the mask overlap between tasks and the gradient norm weighted by the Fisher Information Matrix. This result formalizes why using disjoint masks (small overlap) and EWC (penalizing changes in directions of high Fisher for Task A) should minimize forgetting. Moreover, the paper frames a multi-task extension of the Dual Lottery Ticket Hypothesis (DLTH), hypothesizing that two random expander subnetworks can be transformed into high-performing tickets for different tasks via EWC-guided training. This is a conceptually
interesting contribution, connecting lottery ticket theory to continual learning.
3. Empirical Efficacy: The proposed method achieves state-of-the-art empirical results on multi-task learning with a large LLM (Llama-3-8B). However the margins are slim.
4. Addresses Catastrophic Forgetting Directly: The method explicitly targets both of the core continual learning issues (catastrophic forgetting and interference) in a straightforward way. By freezing most weights and only training a task-specific
subnetwork, interference is naturally limited (especially if masks barely overlap). And by applying EWC to all parameters (especially those important to prior tasks), the approach actively preserves past knowledge. This dual approach is more direct and potentially
more effective than baselines like LoRA (which doesn’t explicitly prevent forgetting) or
LoTTO (which uses disjoint masks but no regularization).

### Weaknesses
1.	Sparse Mask construction details are missing: It is not mentioned how the expander graphs are calculated for the LLM parameters in question. The author mentions that there are standard algorithms to construct ramanujan graphs but does not divulge into the details such as layer-wise mapping, degree selection per tensor or other implementation details. Without this, “expander properties” are asserted at a high level but it’s unclear how they commute with real parameter layouts.
The authors also mention that constructing these graphs are computationally non-trivial for large graphs, however they do not show calculations on how much wall-clock time is actually needed for these computations for their chosen models.
2.	EWC/ Fisher estimation details are missing: The authors do not describe how they calculate the FIM, which approximations they use, whether the entire dataset is used or not. How the FIM is updated across tasks is also not discussed.
3.	Result Mismatch: The authors note that they were unable to reproduce the results for Instruction Following task from LoTA paper (LoTA paper reported values are higher than this papers perf figures) , and given the marginal improvements in other tasks it’s a bit of concern on whether the proposed methodology actually beats the SoTA. 
4.	Inconclusive Testing: The authors only test their proposed methodology on sequences of two tasks at once. This is extremely different from the real world scenarios where the model might have to face multiple tasks sequentially. They also did not test their methodology on multiple LLM backbone models which further questions whether their methodology works in general or not. The lack of variance figures in the results are also concerning. Finally the authors should also compare their methods to other traditional CL strategies such as replay based methods and other advanced versions of LoRA such as AdaLoRA etc.
5.	Inconclusive training time and parameter requirements: The authors mention that the parameter count with 10% sparsity is comparable to a 256 rank LoRA. However 10% of a 8B parameter model implies about 800M trainable parameters which is much higher than a traditional 256 rank LoRA setup (on Q and V only) that requires about 130M parameters for the same model, even if we compute LoRA for Q,K,V,O and the MLP projectors which is extremely unnecessary and uncommon that requires a parameter count of 600M which is still lower than their 10% sparsity count. 
On top of that they mention that EWC is applied on all parameters. If so then that requires optimizer states for all params and increases the memory and bandwidth overheads significantly compared to other PEFT methods such as LoRA. This is a point of contention as that would significantly diminish the paper’s claim of being an efficient way to train LLMs in a Continual Multitask setup.

### Questions
1.	Fisher Information Matrix calculation: 
How was the FIM calculated for the Llama-3-8B model, which approximations were used. What data was used to calculate it and the computational overhead required.
2.	Mask construction and Application: 
	How are the expander graphs calculated in this LLM NN context? 
	Is it calculated layerwise?
	Is a separate (n,d,\lambda) expander graph generated for each layer and if yes 
	How do you ensure global structural properties?
	How is minimal overlap (low jaccard Index) between masks ensured?Is this done 
	Through a sampling-rejection scheme or some disjoint mask generation method?
3.	Training dynamics outside the active mask:
	 Since L_B only depends on the current mask but the EWC penalty is on all 
	Parameters, are params outside m_B updated solely by the quadratic EWC 
	Regularization term? Did you compare against simply freezing them?
	Also if the graphs are sparse enough shouldn't that be enough to prevent CF?
	Is any regularization applied to the Fisher matrix (e.g., damping, clipping),
	especially given potential noise and numerical instability at this scale?
4.	Scaling and Generalization: 
	Have you tested the method beyond two sequential tasks (e.g., three or more)? If
	so, does performance degrade or saturate with accumulated masks?
	How does this methodology scale to other LLM models of different sizes?
	How does your method compare against other traditional CL methods such as
	Replay based methods and other improvements on LoRA such as AdaLoRA etc.
5.	True parameter count and training dynamics:
	What are the different training overheads related to mask calculation and training
	Both in terms of clock time and compute?
	What is the actual number of trainable parameters and compute load and how
	Does it compare against the other baselines mentioned?
6.	Safety and Evaluation nuances:
	Define the “% safe outputs” used in the experimental results.
	Report variances for the different performance metrics.
	Some ablation results are there for certain tasks and not for others (e.g. the 
	Comparison against the random masks is only done for the GLUE tasks on
	RoBERTa and not for the other tasks done on the Llama-3-8B model. Detailed
	Ablation study required.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a continual learning method based on Lottery Ticket Adaptation (LoTA) and Elastic Weight Consolidation (EWC). The proposed method can be viewed as an extension of the Dual Lottery Ticket Hypothesis, where the expander masks can be co-adapted for successful continual learning. The proposed method is evaluated by several finetuning tasks in the language domain, using Llama-3-8B model. The evaluation tasks consist of single-task finetuning and continual learning. In single-task finetuning tasks, the proposed method outperforms baselines including FFT, LoRA, and LoTA in most cases. In the continual learning evaluation, a sequence of two tasks is evaluated, and the proposed method outperforms the baselines in most cases. The authors also provide a light theoretical justification of the method in the appendix.

### Strengths
1. The motivation for network sparsity is promising.

2. The evaluation benchmark tasks are inclusive, covering many different types of tasks in language modeling.

### Weaknesses
1. The proposed method lacks justification on why and how disjoint (or as disjoint as possible) binary masks will contribute to the continual learning performance

2. The continual learning evaluation only involves a sequence of two tasks. It's hard to tell whether the proposed method works in continual learning settings with more tasks.

3. 10% of the model is masked as reported in Table 2. However, how the ratio has been selected is not reported. 

4. More ablation studies are needed to prove the effectiveness of the approach, e.g.,  which part of the proposed method contributes more -- lottery ticket or EWC. Moreover, EWC should be included as a baseline.

### Questions
1. In L224, Meta’s Llama-3-8B model (see model card) -- which model card?

For the rest of the questions, please refer to Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
