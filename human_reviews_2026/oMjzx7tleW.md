# AutoStat: DSL-based Automated Statistical Modeling from Natural Language

- Avg Score: 2.80
- Decision: Reject
- Scores: 2, 2, 2, 2, 6

## Abstract
Statistical modeling plays a critical role and is widely used in data analysis across diverse domains. Despite its importance, existing workflows remain cumbersome: they rely on fragmented programming environments and domain-specific probabilistic programming languages that are verbose and difficult to use, especially for non-experts. Although many efforts have been made toward automated statistical modeling, the methods still suffer from low accuracy, high computational cost, and heavy reliance on manual intervention. To address these challenges, we present \textbf{\textit{AutoStat}}, a novel Domain-Specific Language (DSL)-based framework for automating statistical modeling. AutoStat leverages \textbf{\textit{StatModelDSL}}, the first compact and structured DSL that specifies complete modeling tasks in a unified and portable form. AutoStat further enhances the automated process via interactive modeling by integrating two agents -- StatModelChatbot, which interactively refines underspecified user requirements, and StatModelCopilot, which generates executable DSL programs. 
With StatModelChatbot clarifying intent and StatModelCopilot emitting executable DSL, AutoStat compiles and executes the specification end-to-end, delivering the complex statistical models directly from natural-language dialogue.
We demonstrate that the proposed StatModelDSL affords both LLM amenability and practical usability: when instantiated with GPT-4o, it yields a \textbf{91.59\%} reduction in error rate and a \textbf{5.89\%} uplift in user preference over a Stan-based workflow. Meanwhile, AutoStat achieves a \textbf{100\%} syntax correctness rate for DSL generation and a \textbf{98.76\%} semantic passing rate, significantly surpassing previous methods. Our dataset, codes, and models will be publicly released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The presents a domain specific language framework, called AutoStat, to automate statistical modelling. The framework consists of:
- A novel DSL named StatModelDSL, which can compile to multiple PPL frameworks, specifically listed are Stan and PyMC3
- An agent/fine-tuned LLM, StatModelChatbot, which refines user queries
- An agent/fine-tuned LLM, StatModelCopilot, which generates programs in StatModelDSL from the user specifications
Empirically, the paper evaluates AutoStat framework on a test set of synthetically generated modelling problems. AutoStat is able to provide higher accuracy in generating syntactically valid DSL code, as well as, providing output programs that more closely match the specified target program, as evaluated by an LLM judge.

### Strengths
The task of improving how non-experts can leverage PPLs for improved statistical modelling is definitely valuable. Furthermore, the authors seem to clearly but a lot of work into the non-trivial task of generating a dataset for statistical modelling task, even though it is fully synthetic (a downside discussed in the weaknesses).

In general, the paper is easy to read and the main ideas can be followed easily.

Overall, I am skeptical though of the benefit of using a purpose-built DSL for LLMs for this use case which I discuss in the weaknesses below.

### Weaknesses
Multiple high-level weaknesses of the paper currently make me vote for rejection:

1. Overall, I don't see a need for designing a novel DSL that is mainly designed for the usage by LLMs. The theoretical arguments and the experimental results provided in the paper do not convince me otherwise. I base this judgement on the following three points:
  - The paper says the proposed DSL offers "Completeness" (L.70) of being able to represent the full data analysis pipeline necessary for statistical modelling. But there is no proof  (either theoretical or empirical) provided in the paper that the DSL is actually in any sense of the word "complete" and the experimental evaluation is exclusively done on synthetic data. Fundamentally, any DSL will have edge cases that cannot be represented with it. I think this downside of choosing to formulate a new DSL is not discussed enough in the paper.
  - The DSL supposedly provides "Portability" (L.71) because it can compile to Stan and PyMC but in practice there will only be a limited number of PPLs that the DSL can compile into. Effectively, there is still a lock-in into the PPLs that AutoStat has as compilation targets.
  - LLMs are already improving rapidly on coding tasks, making it unclear whether developing a DSL specifically for LLMs and statistical modelling is necessary. This is backed up by Table 1. GPT-4o with in-context learning outperforms both the AutoStat-1B and AutoStat-3B models. I would assume that more recent GPT or Claude models would have performance that is competitive with AutoStat-8B.

2. The paper presents a framework for automated statistical modelling *but the generated statistical models are never evaluated based on their statistical properties*. None of the following questions below are answered:
- What is the test set/cross-validation performance of the generated statistical models? 
- How easy is statistical inference in the generated models (e.g. number of divergences in HMC samples, effective sample size (ESS), etc.)?
- What is the performance of the whole set-up on real-world data and modelling tasks? As far as I can tell, the evaluation is limited to synthetically generated problems.

3. One of the headline metrics to praise the performance of AutoStat is that it achieves "100 % syntax correctness rate for DSL generation" (L28). But this does not seem to be a very difficult goal to achieve? Tools such as outlines can guaruantee that you will generate syntactically valid programs with arbitrary base LLMs.

### Questions
There are very few details on the conducted user study in the main text and the appendix. Additionally,  only 17 participants for a user study seems to be quite small to have conclusive results and there is not much detail on the study design and what was done to ensure the results are statistically significant. Additionally, it would be good to have more (anonymised) details about the study participants, are these professional software developers or undergraduates, etc.?

Minors:
L. 129: Missing whitespace in front of citation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
A framework for automatic statistical modelling is proposed. The framework architecture has free major components: a statistical modelling DSL, an interactive bot helping produce a description of the statistical model and analysis, and the DSL code generator. The DSL code is further translated to either PyMC or Stan to execute the statistical analysis. The implementation is trained and evaluated on synthetic data.

### Strengths
The paper considers automation of statistical modelling and analysis, a long-standing and active research subject.

### Weaknesses
1. The framework is trained on a synthetic dataset generated by an LLM and evaluated on a synthetic dataset generated by an LLM. This gives little indication of the framework's performance on real problems. This should be evaluated on real problems harvested from publications and online resources instead. 

2. The DSL, claimed to be a better representation of probabilistic programs for generation purposes, and thus consituting a crucial part of the framework, is not defined in the body of the paper. There is BNF in the appendix. There is no DSL operational semantic anywhere to be seen. 

3. Based on the BNF, the DSL is less expressive than Stan (and PyMC for that matter), for example it does not have conditional execution of any form. With DSL being a simpler language that that generated DSL has fewer errors than generate Stan is of little surprise.

4. DSL being less expressive than Stan/PyMC, it would be expected to show what subset of Stan/PyMC can be translated into the DSL (the opposite direction). This is missing in the paper entirely.

Overall, the contributions are not clear, and evaluation is insufficient and biased.

### Questions
How does a DSL snippet corresponding  to the following Stan code  look like?

```
real lpdf = baseline_lpdf(y[n] | theta);
if (y[n] == 0) {
  // Contributions from both components
  target += log_mix(lambda, 0, lpdf);
} else {
  // Contribution from only the baseline component
  target += log(1 - lambda) + lpdf;
}
```

I took this snippet from Betancourt's mixture model tutorial.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a framework called AutoStat, which is used for statistical modeling. The main idea is to introduce a new domain-specific language (DSL) called StatModelDSL, from which Stan or PyMC programs can be generated. The aim is to make it easier for LLMs to generate models in StatModelDSLs, compared to generating probabilistic programs in Stan or PyMC directly. The framework includes two components: (i) the StatModelChatbot that interacts with the user when performing the modeling, and (ii) the StatModelCopilot that is used for generating the DSL program.

### Strengths
- Trying to do statistical Bayesian modeling from natural languages is an extremely hard, yet relevant problem.

### Weaknesses
- There are very strong negative claims about probabilistic programming languages (PPLs) in general, such as "workflows for statistical modeling remain overly complex and unfriendly to users" or "verbose PPL syntax". Most PPLs encode Bayesian models as programs, directly encoding sample and observation statements in a concise and precise way (e.g., WebPPL, Stan, Anglican etc.). Saying that PPL syntax in general is verbose is an unjustified statement. 

- On line 130-132 it says "while PPLs are powerful for statistical modeling, the complete workflow still requires external tools (e.g., Python or R) for data processing and analysis." It sounds like this is a drawback. However, this design is typically very intentional. Why would a new PPL introduce its own framework for data processing and visualization, when there are already excellent software packages in Python and R that the intended users are already used to?

- The paper contains several unjustified and strong claims, such as line 137 "Leveraging LLMs, we make the workflow easy and reliable." The paper does not justify such strong claims. It is, by definition, very hard to scientifically show that a workflow is "easy" and "reliable". Hence, a scientific paper should not contain such unjustified strong claims.

- A key part of the suggested approach is to design a new DSL for statistical modeling. But, it is unclear what is unique about it. All existing DSLs are by themselves DSLs, although most of them are embedded DSLs in a specific host language. The key idea and novelty of the proposed DSL are not clearly expressed in the paper.

- Statistic modeling is a very complex task in general, and traditional PPLs have made a huge step in simplifying this process by separating statistic modeling and inference. This paper makes a very strong claim about automatic modeling using LLMs, but there is no clear evidence of how this is useful or can work accurately in any real statistical modeling tasks. The papers present numbers comparing the framework with baseline GPT-based standard LLMs, but this does not say anything about the actual modeling capabilities compared to actual real modeling and problem-solving using PPLs.

### Questions
- Please provide concrete examples and justification for where LLM-based modeling of probabilistic programs actually works. That is, examples where the approach can be used to model non-trivial statistical models that have not been directly seen before.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents (i) a DSL for data analysis, (ii) a two-agent LLM-based architecture for generating DSL code from task descriptions, and (iii) a simple methodology for supervised fine-tuning the LLM to generate the DSL code more accurately. The two-agent architecture includes a (i) an agent that elicits detail from a user in a back-and-forth conversation, and (ii) an agent that generates the DSL code from a complete task description. Evaluation is done on a synthetic dataset (not provided) generated using GPT-4o, and quantitative evaluation is done using GPT-4o.

### Strengths
Significance: The general topic (LLM-assisted data analysis and data modeling) that the paper explores is interesting and important and has potential for impact. Leveraging LLMs to assist in the general of PPL modeling code seems promising.
Originality: The type of user study explored is interesting and in the right direction (evaluating how easily a user can make a modification to a model).

### Weaknesses
The task presented to StatModelChatbot (eliciting details from a user) potentially includes meaningful work its purview, but the paper does not provide enough information to make this clear. The biggest challenge in LLM-assisted data analysis lies in the translation of a user data analysis goal or high-level task description into a statistical model. The method shown appears to only assist users who already know the model they want to write. That is not a significant contribution. Appendix C does not include a complete input the StatModelChatbot from the evaluation dataset, so it is impossible to tell whether this module is doing a meaningful task, or is only doing a fairly trivial translation task (like the StatModelCopilot, see below). Because the paper does not include example tasks, it is impossible to determine how realistic the tasks are. Do modern LLMs understand statistical modeling enough to assist users? The paper ignores this fundamental question.

The task that the code generation agent (StatModelCopilot) performs (and is evaluated on) is too narrow and does not capture the difficult parts of the LLM-assisted data modeling. Specifically, the code generation model is evaluated on the task of translating a low-level task description that appears to already include almost-verbatim snippets of the target code (see "diff_consumption ... " in the "Output task definition" on page 27-28) into code. This appears to be a trivial language-translation task for modern LLMs.

The DSL does not appear to be a novel contribution. It appears to be a minimal extension of existing PPLs (e.g. borrowing the transformed_parameters and transformed_data idioms from Stan). It appears to be roughly Stan with a simple JSON-like schema for selecting a few parameters for existing PPLs (e.g. selecting an inference engine from an enum, specifying file paths to load data from).

There is some minor novelty in use of a two-stage training curriculum, but the supervised fine-tuning training methodology is standard and is generally not novel.

The paper has major clarity weaknesses. For example, reading the caption for Table 3 literally indicates that the base model Llama 3 8B performs identically to the trained Llama 3 8B model (Autostat 8B in Table 1). I can infer that Table 3 reports results after fine-tuning each of those base models, but this is never stated in the caption or the text.

Also, the use of GPT-4o for generation of synthetic data (including "random parameters") makes me think that the examples that are used for training and evaluation may not reflect realistic tasks. This comports with the paper's general lack of concern with demonstrating the effectiveness of the end-to-end system on real-world end-to-end use cases.

The use of GPT-4o for quantitative evaluation of correctness, without an evaluation of this methodology's soundness, reduces the soundness of the paper.

### Questions
1. The paper would be improved with at least one complete example (ideally more) of a train task and a test task (including the complete input to the StatModelChatbot).
2. I suggest that the authors use an existing benchmark for data science tasks, or create a more realistic evaluation, to highlight the significance of their work.
3. I suggest that the authors focus on the StatModelChatbot part of their workflow, which seems to be where the potential for an LLM to understand the consequences of modeling decisions and to add value to a data analysis user lies. Reformatting or simple translation with an LLM does not seem challenging enough to make for a meaningful contribution. The back and forth between the agent, the user, and the consequences of their modeling assumptions, seems to be where the potential for novelty lies.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
AutoStat presents a framework for automating statistical modeling from natural language by introducing StatModelDSL, the first Domain-Specific Language designed specifically for statistical modeling workflows. The paper addresses three problems with existing approaches ( insufficient specifications when using PPLs, fragmentation across heterogeneous environments, and lack of portability between inference engines. AutoStat consists of two main agents, StatModelChatbot to capture complete task specifications and StatModelCopilot to translates the refined natural language descriptions into executable DSL programs. The DSL itself provides a structured, block-based representation of the entire modeling pipeline including data preprocessing, model specification, inference configuration, and output analysis, and can be compiled to multiple backends including Stan and PyMC. 

To train the system, the authors construct StatModelDataset through a three-stage process: generating synthetic data. The StatModelCopilot is trained by first learning DSL syntax from concise descriptions then learning to capture fine-grained details from comprehensive specifications. Experiments demonstrate that when instantiated with GPT-4o, StatModelDSL achieves a 91.59% reduction in error rate and 5.89% uplift in user preference compared to Stan-based workflows, while the trained 8B StatModelCopilot achieves 100% syntax correctness and 98.76% semantic passing rate, significantly surpassing baseline methods including few-shot prompting with GPT-4o.

### Strengths
The StatModelDSL design is well-motivated with benefits such as completeness and clarity through explicit block-based representation, unification and portability by enabling compilation to multiple backends, and LLM-friendliness through structured design that makes fine-grained details explicit. This addresses real pain points in current workflows where practitioners must context-switch between Python for preprocessing, Stan or PyMC for modeling, and back to Python for analysis.

The interactive modeling approach through StatModelChatbot is a good design choice that acknowledges the inherent complexity of statistical modeling tasks. 

The ablation studies show that both training stages contribute to final performance, with Stage One establishing syntactic correctness (100% pass rate) and Stage Two enabling semantic alignment (98.76% pass rate).

The experimental evaluation is comprehensive across model sizes, few shot and finetuned approaches. The paper demonstrates strong software engineering with a working compiler that parses DSL into AST using Lark, executes data processing in Python, and handles model inference through target PPLs. The examples in Appendix B showing the same linear regression task compiled to both Stan and PyMC demonstrate the portability claim concretely. The provision of detailed execution EBNF and block descriptions shows technical rigor and reproducibility.

### Weaknesses
The evaluation scope is limited to relatively simple statistical models.

The reliance on GPT-4o as both the dataset generator and evaluation judge raises concerns about evaluation circularity and bias, as the same model family might be implicitly reinforcing its own conventions. 

The baselines are limited to raw LLM prompting with Stan or PyMC rather than comparisons with specialized automated statistical modeling systems like AutoBayes, Bambi, or Turing.jl pipelines, which weakens the claim of outperforming state-of-the-art methods. Moreover, the user study, though promising, is small in scale (17 participants) and focused on short modification tasks rather than long-term usability or learning curves. The paper also doesn't discuss analysis of end-to-end latency and system efficiency and does not discuss how AutoStat handles runtime errors, debugging, or real world data irregularities.

### Questions
How expressive is StatModelDSL relative to full PPLs? For example, can it support hierarchical, mixture, or nonparametric models? 

Did you take any steps to mitigate using GPT4o for both data generation and evaluation?

What are the system’s runtime and interaction costs in practice?

Have you compared AutoStat’s modeling accuracy and computational efficiency against other frameworks?

### Soundness
3

### Presentation
2

### Contribution
3
