# Recast Your Input via a Mapping Function for Alignment

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Alignment is promoting its critical role among the large language model (LLM) scenarios, which ensures safety, controllability, and trustworthiness of the generation. The popular alignment methods, that is, reinforcement learning from human feedback (RLHF), direct preference optimization (DPO) and such series, usually change weights of the model by elaborate algorithm. Nevertheless, they suffer from the compute drain for training, especially when the parameters' size getting huge. Worse still, people typically do not have access to the weights of the SOTA models, such as GPT-4, which consequently renders the aforementioned algorithms unimplementable. In this paper, we propose to employ a separate LM as the Refiner, an input mapping function essentially, to transform the original query into a novel formulation that impels the final generation to align with the expectations. During optimization, an evolution strategy, namely CMA-ES, is leveraged to fine-tune the LM with linkage to the generation model. We conduct extensive experiments on various refiner and generation types, and achieving surpassing results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes using an additional Language Model (LM) as a Refiner to improve the input, which ultimately leads to outputs with better alignment. Experimental results have demonstrated the effectiveness of this method.

### Strengths
1. The method reduces parameters requires training, which is more efficient for preference alignment.
2. The effectiveness of the proposed method is validated through experiments.

### Weaknesses
1. The approach bears a strong resemblance to existing methods that improve the performance by refining the prompt, which raises concerns about the novelty of the paper.
2. Ablation studies suggest that the impact of the CAM-ES module is marginal or not statistically significant.

### Questions
While the parameter efficiency is a stated advantage, the paper lacks a theoretical justification for a key finding: why does this method outperform full fine-tuning approaches like DPO and BPO? Especially since your experiments seem to confirm this counter-intuitive result, I'm confused about this experimental result.

### Soundness
3

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
The paper under review proposes a novel approach to align large language models (LLMs) with human preferences by introducing an input refiner module. This module employs a latent variable to transform the original input into a refined version that better aligns with the desired output. The method utilizes the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize the refinement process, ensuring that the generated responses meet expectations without requiring access to the model's internal parameters.

### Strengths
The introduction of an input refiner using latent variables and CMA-ES is a novel contribution that addresses the limitations of existing alignment methods, particularly those requiring access to model weights. By avoiding direct manipulation of model parameters, the proposed method reduces computational overhead, making it feasible for use with state-of-the-art models like GPT-4. The method is adaptable to different LLMs and can be integrated with various generation models.

### Weaknesses
1. The concept of alignment through refining prompts is not new and somewhat outdated, as it has been extensively explored in previous literature. The authors should provide a more comprehensive discussion on how this paper distinguishes itself from prior works [1].

2. Though the LLM can only be a black-box module for some closed-source LLMs like GPT-4, as you can revise its input prompt, you can also revise its output response for alignment. Doing them together could be more effective than only doing it in the input side [2].

3. Posterior regularization can conflict with alignment goals if the output y is not aligned with the input x. This misalignment can introduce bias in optimizing the learning of z in Eq. (8). The authors might consider using a preference model to decide whether to apply regularization with y. The current results are not convincing and should be evaluated against more complex alignment benchmarks, such as diverse preference sets.

4. The authors should provide a more detailed explanation in Section 3.1, particularly regarding Eq. (5), which lacks rigor. Variables z and x′ should be sampled from distributions rather than being deterministically projected.


[1] A systematic survey of prompt engineering in large language models: Techniques and applications.

[2] EFFICIENT LLM ALIGNMENT VIA HIERARCHICAL COARSE-TO-FINE REFINEMENT.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method to align black-box large language models (LLMs) by learning a "mapping function" to "recast" the user input. This function is optimized using the CMA-ES algorithm to steer the model towards more aligned responses.

While the paper addresses an important and timely problem, its execution suffers from significant flaws in clarity, methodological explanation, and experimental validation, making it difficult to assess the true merit of the proposed approach.

### Strengths
+ The paper tackles a highly relevant and challenging problem: the alignment of proprietary, black-box LLMs where access to model weights is not available.
+ The basic idea of refining the prompt using a small LLM, like 1b/3b, seems to be practical and valuable.

### Weaknesses
+ The paper is quite difficult to follow. The writing is disorganized, and key concepts are not introduced clearly. Notations are not defined in the proper place. The methodology section, along with Figure 2, is nearly incomprehensible.
+ The core method is not adequately explained. Algorithm 1 is presented as the main framework but lacks a clear, step-by-step textual explanation to accompany it. The "mapping function" itself is not clearly parameterized, making it hard to understand what is being optimized. The paper would greatly benefit from a simple, concrete "with/without" example to help the reader build intuition for what the input refiner is doing.
+ The figures and tables are not up to publication standards.
  - Figure 2 is low-resolution and difficult to read.
  - Tables are populated with a large number of raw scores but lack clear summary statistics (e.g., averages, confidence intervals). This makes it very difficult to interpret the results or draw conclusions.
+ The experimental setup and analysis are insufficient.
  - The proposed method fails to outperform the `Best-of-N` baseline, a key result that is not discussed.
  - More troublingly, the tables show the *original* base model performing better than both BPO and DPO. This is a highly unusual result that suggests a fundamental problem with the baseline implementations, yet it is presented without comment.
  - The very low "tie" ratio in the pairwise comparisons is also not analyzed, as it does not always appear in alignment papers.
  - The results are evaluated using "gpt-4o-turbo" as an LLM-as-a-judge, which may not be strong enough for the evaluation.
  - With the counterintuitive results, some experimental setups are missing. For instance, the paper uses DPO as a baseline but provides no information on how it was trained.

### Questions
How do the authors choose the model for experiments? In the paper, llama 3, llama 3.2, mistral v0.3, and qwen 2.5 with different model sizes are included.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes a method for LLM alignment that bypasses traditional parameter-updating approaches like RLHF and DPO. The core idea is to use a separate "Refiner" model that transforms user queries into refined inputs that better align with desired outputs from black-box generation models. The method employs CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to optimize the refiner's behavior based on feedback from the generation model, combined with posterior regularization using preference data.

### Strengths
* Practical problem setup that addresses the difficulty of finetuning closed-source models
* Novel viewpoint of alignment by modifying inputs to achieve the alignment goal. The latent variable approach is also generally interesting.

### Weaknesses
* Limited interpretation of latent variables. The paper claims it represents "user preference with diversity" and "reasoning paths," but provides no theoretical or empirical evidence to support this.
* Presentation and Clarity Issues. Figure 2 is confusing to understand given the fact that math notions appears much later.
* Using only 256 samples for CMA-ES optimization seems insufficient for robust optimization, especially given CMA-ES's sample complexity. It would be good to have variance (or similar stuff) reported.
* Technical confusion. Please see questions.

### Questions
* The constraint function B(.) seems crucial but is barely explained. How was this specific form derived?
* Given the uncertainty of latent variable z, why is the latent variable z necessary? Can we get rid of z?

### Soundness
3

### Presentation
2

### Contribution
2
