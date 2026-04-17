# ChemReason: A Chemical Code-Driven Reasoning LLM via Verifiable Reinforcement Learning

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
In chemistry, most research on large language models has centered on knowledge question answering and retrieval. However, these approaches fall short on core tasks such as open molecular generation and optimization: they lack explicit reasoning processes, and their outputs cannot be systematically verified, leading to severe issues of scientific hallucination. To overcome these limitations, we propose ChemReason, a chemical LLM grounded in generative code reasoning. Unlike non-reasoning LLMs, ChemReason is a code-driven reasoning model that provides transparent inference for molecular editing, generation, and optimization. By dynamically generating and executing chemical verification code during reasoning, the model validates each step, ensuring the scientific reliability of its answers under open-domain conditions. On the TOMG benchmark, ChemReason achieves state-of-the-art performance and demonstrates end-to-end verifiable inference for open molecular tasks. More broadly, the proposed "code–verification–reflection" paradigm offers an extensible pathway for AI for Science, providing a generalizable architecture for addressing complex scientific computing challenges.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce ChemReason, a tool-augmented reasoning LLM fine-tuned for molecular editing, optimization, and generation. Grounded in generative code reasoning, it verifies and reflects on its own steps by auto-generating and executing verification code during the chain of thought.

### Strengths
- S1: The paper tackles a real, high-impact challenge in chemistry/drug discovery: language-based molecular optimization.
- S2: Solid pipelines to generate traces for SFT and model training with ablation study on the main components: access to tool calls and RL.
- S3: Results are strong: the method outperforms much larger general-purpose models on the TOMG benchmark.

### Weaknesses
### Major:

- **W1: Potential task overfitting vs. genuine chemical reasoning gains.**
The model is explicitly trained on the same tasks it is evaluated on, unlike the baselines. It’s unclear whether improvements reflect overfitting or superior chemical reasoning. Please add:
    - *Task-transfer tests*: Without retraining, does the model generalize to related objectives (e.g., solubility/TPSA optimization, scaffold hopping)?
    - *Capability retention*: Does it retain general language-modeling abilities, or have these degraded?

- **W2: Lack of uncertainty and significance.**
The significance of results cannot be assessed without variability estimates. If these are Pass@1, please run ≥3 independent runs with different seeds and report mean ± s.d., plus ideally paired significance tests.

- **W3: Unclear differentiation from tool-enabled general LLMs.**
Conceptually, your model is an LLM with Python + *RDKit* tool access, capabilities also available in recent GPT models. What differentiates your model from them? I know that they are expensive models but having this comparison would be very valuable.

### Minor:

- W4: Missing Limitation section.
- W5: Related work coverage (contemporaneous). Although these are near-contemporaneous, please mention early preprints on chemical reasoning LLMs (e.g., ether0, ChemDFM-R) in the Introduction/Related Work for completeness.
- W6: Terminology: “cold-start.”  The term is used inconsistently in the LLM/RL literature, sometimes for an SFT phase (as in your case) and sometime for RL without an SFT phase. Currently, you introduce “cold-start” at L68 and only clarify SFT at L240. Please clarify the first time you use the term that you mean an SFT phase.
- W7: Missing citation or Synapse (L127).
- W8: Type in "Methodology" (L146).
- W9: Figures and Tables are not self-contained. E.g., Fig 2: Add a legend or panel description clarifying symbols/labels (T, C, R, CT, M, etc.) so the figure is self-contained, and Table 1:  Define Aut., Bon., Fun. in the caption; state the metric(s); explain what bold and underlining denote.

- What the difference between Qwen3-8B-SFT (original data)/ SFT(ori), C-SFT and C-SFT models? Aka what is the original data, once again I tought that you only had traces with code-calls. Did I misunderstand something? Also, do I understand correctly that the difference between C-SFT and C-SFT+TIR, is that both the trained on the traces with code but only +TIR had access to tool calls during evaluation? As you can see I'm bit confused on differences between your models and SFT training data. Please clarify these. OK, it seems that you give of these explanations in 4.3 after the main results. Please explain to the different models in the main table before showing the main results or at least give a hyperlink to this sections or simply remove the ablation study results from the main table as you anyway show them in Fig 4.

### Questions
- Q1: On L277, you mention that GRPO doesn't require a reward model, but in Fig. 3 you have a reward model in the C-GRPO pipeline... 
- Q2: Not sure, I understand the Eq. 1 and what process above does it refer to? The trajectory generation for the SFT? For me it reads as the likelihood of answer given a set of tasks $\mathcal{T}$, but probably it should for a one specific task? And do I understand correctly that $k$ is number of reasoning-code-result iterations, or what exactly is a reasoning step?
- Q3: Not sure I understand section 3.3.2 and Eq. 2 correctly. From previous sections, I understood that all your COTs contain a tool call. However, from Eq. 2 I would understand that you have cases where you have tools calls and others that do not. Please clarify.
- Q4: Please clarify the variants before the main results: What exactly are Qwen3-8B-SFT (ori), C-SFT, and C-SFT+TIR? What is "original" data? Also confirm whether C-SFT and C-SFT+TIR share the same code-trace training data, with +TIR differing only by tool access at evaluation, as for my TIR doesn't make sense during SFT. Add a short explanation of each of variants before Table 1 (or at least cross-link to §4.3), or move the ablations out of the main table, as the ablation results are already shown in Fig. 4, to avoid confusion.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Chemical Reasoning LLM, ChemReason, for 9 different tasks spanning from molecular generation and molecular optimization to molecular editing. The core idea of this paper is to incorporate codes as verification tools to allow Chemical LLMs to verify and reflect during thinking. They construct such trajectories and train Qwen3-8B with cold-start SFT and RLVR, and evaluate their model using TOMG-Bench.

### Strengths
They are the first to implement the tool RL and use code as a tool in the field of Chemistry. The results on TOMG-Bench are promising.

### Weaknesses
1. The paper is poorly written and organized, to the point of significantly hindering readability. Even several fundamental requirements for an academic paper have not been properly achieved. For example:
    * A great number of citations are missing, such as TOMG-bench, Synapse, and all the baseline models used in their experiments. (Also, the format of all the citations is incorrect.)
    * The writing logic of the Related Work section is confusing, and some parts read more like content that should belong in the Introduction section.
    * Several concepts in the paper are insufficiently or inadequately explained, such as the method to assign difficulty levels to the problems in TOMG-Bench, the role or function of R' in Figure 2, and the differences between C-GRPO and GRPO.
2. Their description of the advantages and core methodology of GRPO is almost entirely incorrect. The key improvement of GRPO lies in estimating the expected reward of the current state, V(S), through group-wise averaging, thereby eliminating the need for a Critic Model as used in PPO. However, in this paper’s discussion:
    * “Instead of assigning absolute rewards, GRPO normalizes rewards within each group.” Other algorithms, such as REINFORCE-Baseline and PPO, also compute the advantage rather than directly using the absolute reward, so this is not a distinguishing feature of GRPO.
    * “While other RLHF approaches require a dedicated reward model, GRPO can flexibly rely on any scoring function or even a stronger LLM to assess solution quality.” The authors seem to conflate the RL optimization algorithm with the reward system. For all algorithms that support optimization using outcome rewards, such as REINFORCE++, PPO, and GRPO, the method of obtaining the reward is irrelevant to the optimization process itself. As long as an outcome reward can be provided, these algorithms can be used for training.
3. Dedicated design of code templates seems to hinder the generalization capability of the model.

### Questions
1. The format of all the citations is incorrect. You should use \citep, not \citet, in most cases.
2. Please provide more explanation of the questions in the third bullet point of Weaknesses 1.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a model with native tool usage. Specifically, during trace generation, the model produces Python-based verification tools to check its own outputs. The generated code is executed in a sandbox environment, and the resulting outputs are incorporated back into the reasoning trace. Based on these tool results, the LLM learns to either initiate another reasoning round or stop and produce the final answer.
The authors evaluate their model on the TOMG benchmark, where it outperforms baseline methods. An ablation study further shows that the proposed multi-stage training procedure contributes to the observed performance gains.

### Strengths
* **(S1) – Relevance of the tackled research question.**
The proposed approach rests on a critical assumption: compared to external tool usage, an LLM’s reasoning capabilities might improve when it is required to generate tools on the fly. This assumption appears plausible, as generating executable code forces the model to make explicit design choices and reason about program correctness. In doing so, the LLM may develop a deeper understanding of the problem, which could also enhance general reasoning, e.g., reasoning about orchestrating tool usage and arriving at final solutions. Consequently, investigating whether this assumption holds is highly relevant and could yield genuinely novel research insights.

* **(S2) – Experimental results indicate potential significance.**
Experiments on the TOMG dataset and corresponding ablation studies indicate that the staged training procedure (SFT followed by RL) is beneficial. Models trained with both stages outperform their respective baselines. Compared to other base models, the proposed approach also seems to perform well. The choice of the TOMG dataset is appropriate, as it enables evaluation across three relevant tasks: molecular editing, molecular optimization, and conditioned generation.

### Comments
* A notable advantage of the proposed approach is its ability to verify the validity of generated SMILES strings (i.e. RDKit processibility). However, this benefit is conceptually equivalent to that of LLMs employing external tool usage, where RDKit acts as the external verifier.

### Weaknesses
* **(W1) - The assumptions made are not well supported, diminishing both the manuscript quality and the overall relevance of the work**.
    - Statements about tool-augmented models: a) "they remain far from realizing the vision of autonomous scientific assistants" (l132) and b) "They still struggle to unify reasoning, tool invocation, and execution into a coherent, learnable process" (l133f). The reference provided for (a) does not substantiate this statement and does not focus at all on tool-augmented agents. For (b), no reference is provided. Consequently, the validity of both statements remains highly unclear.
    - Verifiability: In a strict sense, the correctness of the generated tools cannot be guaranteed. While in most practical cases it might be sufficient to check only for the correct answer, the generated tools might rely on Clever-Hans effects rather than on a correct implementation.

* **(W2) - The central research question is never answered due to a missing experiment**. The interesting research question is: Does generating tools on the fly within the reasoning trace help to boost reasoning capabilities compared to models using external tools? However, this question is never answered because no model with external tool usage has been trained and evaluated. To answer the raised question, reasoning traces for both approaches would need to be compared. A standard RL-trained model without tool usage would also be necessary to evaluate whether tool usage is required at all to solve the tasks. Currently, the success rates of standard chemical reasoning models remain unclear.

* **(W3) - The manuscript includes several mistakes, typos, and unclear passages**.
    - Brackets for references are missing.
    - Typos: 
        * l092: introductionaa
        * l134: missing white spaces
        * l146: Methodolog
    - The paragraph in Section 3.1.1, which introduces the TOMG tasks, is understandable but needs rewriting.
    - Formula (1) is cluttered.
    - Clarity issues:
        * l199: "we prompt a strong code model" is not specific enough. The authors should describe the model used. 
        * Figure 2: Used abbreviations, e.g., T'_2, should be introduced.
        * l264: "data portion" is unclear. The authors should specify the data used.
        * "C-GRPO": In the RL step, the model is essentially trained with standard GRPO. The rebranding to C-GRPO adds confusion.
        * "Code success nums" (l306f) is unclear and needs further explanation.
        * The abbreviation "Aut" (l381) in Table 1 is unclear.
        * Table 1: The included models SFT(ori), C-SFT, and C-SFT-TIR are insufficiently explained, despite the ablation study section allowing an educated guess as to their meaning. 

* **(W4) - Missing error bars and statistical tests reduce the significance of the results**. The tables and figures are reported without error bars or statistical tests, making it unclear how much of the observed variation could have arisen by chance. The authors should include error bars and perform appropriate statistical tests.

* **(W5) - Related work and experiments lack consideration of other chemical reasoning models**. Reasoning models for chemical tasks is the core of the manuscript, yet the authors completely ignore large parts of relevant prior work in this area, e.g., [1,2].

### References
* [1] Narayanan. Training a Scientific Reasoning Model for Chemistry
* [2] Zhao. MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs

### Questions
* How different is the generated code from the associated templates? Could the code also be fully pre-written, with the LLM only required to execute it? If so, how would the training behavior differ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on training an LLM that can do better at reasoning, leveraging its chemical code generation ability, tool usage, and training strategies. It leverages a few chemical tools that can automatically return results with chemical code as input, construct a reasoning output format, and think of ways to deal with the cold start problem. Experiment results show that the trained LLM with the proposed training scheme, can outperform close-source LLMs such as Claude-3.5 and open-source LLMs such as DeepSeek-R1.

### Strengths
1. The proposed methods are useful ways to augment an LLM on its ability such as reasoning, tool usage, and reflection.

2. The experiment results are very impressive.

### Weaknesses
1. The proposed methodologies are generally known methods for training an LLM (cold start, training an LLM to learn better reasoning and tool usage). It seems like the contribution lies in more on adapting these known strategies and implement them with probably good engineering to a specific chemistry task. In this sense, the contribution is not so significant interms of brining new knowledge to the community. 

I would suggest this paper to rewrite its contribution section, maybe to think of what are novel findings / insights during implementation, and what are the difference between the implemented methodology with the exisiting ones, and what are the insights for training a 8B model to be much better than the powerful public LLMs in some metrics. These unique insights may count as good contributions.


2. The writing and presentation can be improved. For example:

(a) section 3 is named with "methodolog", lacking a "y"

(b) line 259 should use \citep{} instead of \citet{}.

### Questions
Which specific LLM is Claude 3.5? Sonnet or Haiku?

### Soundness
3

### Presentation
2

### Contribution
2
