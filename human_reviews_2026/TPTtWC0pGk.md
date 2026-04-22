# Can Language Models Discover Scaling Laws?

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 4, 8

## Abstract
Discovering scaling laws for predicting model performance at scale is a fundamental and open-ended challenge, mostly reliant on slow, case specific human experimentation. To investigate the potential for LLMs to automate this process, we collect over 5,000 experiments from existing literature and curate seven diverse scaling law discovery tasks. While existing agents struggle to produce accurate law formulas, this paper introduces SLDAgent, an evolution-based agent that co-optimize the scaling law model and the parameters, enabling it to autonomously explore complex relationships between variables. For the first time, we demonstrates that SLDAgent can automatically discover laws that exhibit consistently more accurate extrapolation than their established, human-derived counterparts across all tasks.
Through comprehensive analysis, we elucidate why these discovered laws are superior and verify their practical utility in both pretraining and finetuning applications. This work establishes a new paradigm for agentic scientific discovery, showing that AI systems can understand their own scaling behavior, and can contribute novel and practical knowledge back to the research community.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SLDBench, a benchmark for evaluating scaling law discovery across seven tasks. The paper also introduces SLDAgent, an agent that co-evolves both the generator that proposes law expressions (including their functional forms) and the optimizer that sets values of parameters in the laws. SLDAgent works by prompting an LLM to propose modifications to both of these functions. 

Results: The paper finds that SLDAgent outperforms all existing baselines, including agent-based and LLM-based methods augmented with command-line code agents. They also find that when SLDAgent is paired with GPT-5, it outperforms human experts on all tasks in SLDBench. They show that SLDAgent can be used downstream to optimize hyperparameters for pretraining and to select a pretrained LLM for finetuning.

### Strengths
1. Experiments are extensive, evaluating a variety of baselines on many tasks. Experiments with SLDAgent + different models show that SLDAgent performance can scale with model size. Analyses are thorough, and the paper also shows that SLDAgent can be used downstream in real-world settings (hyperparameter selection, LLM selection).

2. Novelty: While existing works have studied AI agents for science, none have been proposed specifically for evaluating scaling laws.

3. The paper is well-written and easy to understand.

### Weaknesses
1. The comparisons between SLDAgent and other methods is not totally fair.

     - 1a. Comparisons between SLDAgent and other baselines do not control for the number of LLM calls. Results for SLDAgent reflect performance after 50 rounds – do these SDLAgent runs then involve many more LLM calls (or other forms of compute) than other agent baselines? 

    - 1b. It is hard to know how much of the performance of SLDAgent is driven by the LLM versus the optimization and generator subroutines; while the experiments evaluating SLDAgent with different LLMs suggest that the LLM's evolution decisions have an impact on SLDAgent's performance, the paper does not compare these SLDAgent experiments to the performance of a human who had access to the same subroutines. 

     - 1c. We also do not know the impact of *storing programs with high-performing scores* in a database and returning the best program at the end; if baselines or humans had access to such a database, would they perform better?

2. Section 4.3 reports analyses of the differences between human vs LLM written scaling laws. Instead of just qualitatively analyzing a few manually picked examples, it would be helpful to understand patterns in the differences through systematic analyses. Some ideas:
   - human experts rate which of a human-written vs LLM-written scaling law is better
   - analyzing # of parameters and other patterns in functional forms
   - quantitatively analyzing asymptotic behavior

### Questions
1. Are there any patterns in the kinds of changes that LLMs make over rounds? How much diversity is there for a single task/SLDAgent setting across seeds?
2. Line 289: The choice to clip does not feel well-justified; how do the results look if there is no clipping?
3. Do some of the baselines similarly store candidate programs and perform the best performing program? (see 1c in weaknesses)

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
3

### Summary
This paper proposes a new benchmark for LLM agents, namely *generating scaling laws* for language modeling. The benchmark (**SLDBench**) is constructed by collecting scaling law tasks (e.g., predicting loss from $N$ and $D$) and data points from LLM experiments in existing literature. The agent is **given the scaling law task and the set of experimental results**, and **responsible for generating a mathematical equation that fits the given data**.

The benchmark contains seven tasks:
1. Parallel: scaling w.r.t. parallelism $P$ and model size $N$
2. Vocabulary: scaling w.r.t. non-vocab model size $N$, vocab size $V$, and dataset size $D$
3. Supervised finetuning: scaling w.r.t. dataset size $D$
4. Domain mixture: scaling w.r.t. proportions of different domains
5. MoE: scaling w.r.t. network size $N$ and number of experts $E$
6. Data-constrained: scaling w.r.t. network size $N$, dataset size $D$, and unique tokens $U$
7. LR and batch size: scaling w.r.t. learning rate $\ell$, batch size $b$, and dataset size $D$

Each data point under each task contains the parameters of the experiment $\mathbf{x}$ (e.g., dataset size, batch size), the result $y$ (e.g., loss), and a control index (e.g., a particular model). **SLDAgent is a program that iteratively refines an $\operatorname{Expression}$, which implements $f_\bf{\theta}:\bf{x}\mapsto\hat{y}$, and an $\operatorname{Optimization}$ routine, which finds the best-fit parameters $\hat{\bf{\theta}}$.** It begins with an initial pair, namely the power-law function for $\operatorname{Expression}$ and BFGS optimizer for $\operatorname{Optimization}$, which is added to the initially empty database. Then, at each step, some programs are selected from the database (according to pre-defined heuristics), formatted as a prompt, and the agent proposes a modification by altering the law form or changing the optimizer. The resulting child program is executed and added to the database. The loop terminates after a fixed number of iterations. At the end, the program with the highest score in the database is returned.

Evaluation on SLDBench is performed via correlation between the prediction of the scaling law and the ground truth on held-out experiments under each task. The authors show that **SLDAgent paired with existing LLMs outperforms the corresponding provider-specific agents (e.g., Gemini + SLD Agent beats Gemini-CLI) as well as the scaling law proposed in the original paper**.

### Strengths
1. The authors study a new application of LLM agents, namely the discovery of scaling laws for foundation models. This contributes to the growing body of literature on the potential of LLMs to augment scientific discovery.
2. Discovering better scaling laws has the potential to feed directly back to improving LLMs.
3. The paper is well-written and clear (apart from being too broadly scoped in the introduction, as I complain about below).

### Weaknesses
1. The biggest challenge of scaling laws is actually *formulating the set of independent & dependent variables* that make sense to model. For instance, recent works had the insight to propose new axes of scaling such as the vocabulary size $V$ (Tao et al., 2024), the amount of unique data $U$ (Muennighoff et al., 2024), or the domain mixture (Ye et al., 2024). Therefore, the sense of "scaling law discovery" here is limited — the agent is not asked to discover entirely new axes of scaling, but instead to **generate an equation that fits the given data**.
2. Arbitrarily expressive equations can get arbitrarily close to modeling the given data perfectly. Therefore, I find that the experimental setup (which only measures predictive accuracy) does not really evaluate whether models generate *plausible* scaling laws that are consistent with prior knowledge. The authors provide some qualitative examples of human-discovered and agent-discovered scaling laws, but I do not have the expertise to compare their plausibility, and defer to the judgment of my fellow reviewers.
3. The above points themselves are not themselves reasons for rejection, but I find some of the writing in the abstract & introduction to be not well scoped (or even misleading). The authors should make it clear right away that the agent is *given* the scaling law setup and experimental results; they are not being asked to formulate the variables of interest or execute the experiments. A re-ordering of the introduction would also be fitting: rather than starting with the importance of scaling laws and then discussing progress on AI agents, I think it is better to start with the discussion of AI agents and identify scaling law discovery as a particular testbed. This is because the main contribution is actually about agent evaluation, and not knowledge on scaling laws.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper aims to formulate scaling law discovery as a task for scientific agents, and proposes a new agent model, SLDAgent, for this task that outperforms existing agents on this task. I think this is good work that would be appeal to several subsets of the ICLR community that makes data, modeling, and (promises of) application contributions.

### Strengths
- Overall I enjoyed reading the paper - I noticed that a lot of the implementation detail questions I wrote down during my reading got resolved as I read through it, which suggests that the grounds were well covered. 
- The construction of the training-extrapolation set is sensible and reflective of real world scenarios ("For each task, we hold out an extrapolation test set from the dataset by selecting data corresponding to the largest model or dataset sizes.")
- Good controlled comparisons: the results are reported on both fixing the base LLM and varying the agents, and on fixing the agent and varying the base LLMs.
- I appreciated the existence of a human baseline.
- I liked the design choice that the information given to the model includes the task context and the semantics of the parameters involved - I think this will be critical for predicting "harder to predict" scaling laws that I will discuss in the Questions section.

### Weaknesses
The weaknesses below are not critical:
- Sparse discussion of the possible effects of contamination and measures taken to prevent this
- Coverage of unexpected or harder-to-predict scaling laws like inverse scaling or U-shaped scaling

I provide more context about these points in the question section, but I think neither of these limitations are grounds for rejection.

### Questions
- How to deal with possible future data contamination? (or existing contamination because the human discovered scaling laws are probably already online and/or part of the model training data and also are searchable on the web?) Maybe agents have an upper hand to the humans because they can springboard from the human solution that is available to them (at least for the non lr_and_bsz tasks). The fact that they are able to do this is cool of course, but if this were the case the "superhuman" claim should be interpreted with more nuance.
- Maybe slightly more seriously, the extrapolation data might also be available to the models due to them being published. There is really no way of controlling for training-level contamination, but were any measure taken to blacklist access to such information at inference time (e.g., PaperBench does this, if I remember correctly), or were any post-hoc analyses of the agent trajectories done to examine whether they were accessing/making use of such information?
- Do you think it would be useful to consider "harder to predict" scaling trends like inverse scaling or U-shaped scaling that have been observed in the literature and would the agents do well? I think these kinds of settings are cases for which the abstract nature of the task itself is critical information that humans and agents need to make use of in predicting extrapolation trends, more so than the ability to curve-fit the training set's trends well. For instance, we know that "overriding existing definitions" is a category of task that is likely to inverse-scale (at least in some parts of the curve), and this abstract intuition seems important especially if the available datapoints only point to monotonic increase.

Nitpicky comments:
- L076: "the objective is clear, continuous, and unbiased" I'm not sure any objective can be "unbiased"
- L079: "As existing laws are designed based on a series of continuous and iterative human research efforts, existing agentic systems lag behind these laws on difficult SLD tasks, and the human-derived laws could also give negative-correlated predictions under challenging scenarios." -- hard to comprehend logical flow
- L183: "focus on rediscover a pre-known formula" rediscovering*

### Soundness
3

### Presentation
3

### Contribution
3
