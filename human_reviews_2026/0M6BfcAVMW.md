# TusoAI: Agentic Optimization for Scientific Methods

- Avg Score: 3.60
- Decision: Accept (Poster)
- Scores: 2, 2, 4, 4, 6

## Abstract
Scientific discovery is often slowed by the manual development of computational
tools needed to analyze complex experimental data. Building such tools is costly
and time-consuming because scientists must iteratively review literature, test mod-
eling and scientific assumptions against empirical data, and implement these in-
sights into efficient software. Large language models (LLMs) have demonstrated
strong capabilities in synthesizing literature, reasoning with empirical data, and
generating domain-specific code, offering new opportunities to accelerate com-
putational method development. Existing LLM-based systems either focus on
performing scientific analyses using existing computational methods or on de-
veloping computational methods or models for general machine learning without
effectively integrating the often unstructured knowledge specific to scientific do-
mains. Here, we introduce TusoAI, an agentic AI system that takes a scientific task
description with an evaluation function and autonomously develops and optimizes
computational methods for the application. TusoAI integrates domain knowledge
into a knowledge tree representation and performs iterative, domain-specific op-
timization and model diagnosis, improving performance over a pool of candidate
solutions. We conducted comprehensive benchmark evaluations demonstrating
that TusoAI outperforms state-of-the-art expert methods, MLE agents, and scien-
tific AI agents across diverse tasks. Applying TusoAI to two key open problems
in genetics improved existing computational methods and uncovered new biology
missed by previous methods.  Our code is publicly available at https://github.com/Alistair-Turcan/TusoAI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Summary
- The paper introduces TusoAI, an LLM agent for developing computational methods for domain-specific scientific tasks. They apply it to several biological and deep learning problems and demonstrate improvement over other agentic and LLM models. They also demonstrate its performance on iterating on existing state-of-the-art methods and show new scientific insights.

## Recommendation
- Reject. While results comparing with other agentic models are there, the actual performance isn't that much better compared to existing models on the specific tasks. The most interesting result is the case studies, but the other models were not run on the same case studies. There is a lack of case studies in other domains. Code is poorly written and not amenable to open source.

### Strengths
## Strengths
- Model architecture is properly motivated and described in sufficient detail.
- The utility of a model is conveyed properly, I can see the usefulness of this.
- Code is provided.

### Weaknesses
## Weaknesses
- TusoAI is designed to work with any domain as long as domain knowledge is provided, however, there is a strong focus on biological applications. This paper would benefit from showing TusoAI's performance in other scientific domains similar to the two biological case studies. I see that it worked on deep learning but only a cold start was performed. I am specifically asking for a warm start case study in a non-biology domain.
- Diversity in methods produced is a core architecture consideration, but it is not substantiated in the text. While evidence is provided that TusoAI writes diverse code, there are no references supporting the assumption that generating more diverse code is better.
- Code is poorly written / formatted.

### Questions
## Questions
- Was an ablation study performed on the number of points saved for each paper's methods in Step 1 "Gather domain knowledge"? How does increasing/ descresing point summaries affect the final model performance?
- Why were other models not tested on the case studies? A lot seems to be riding on TusoAI's new findings, but other models were not given a chance.
- How is the iterative refinement done for "The initialization agent Ainit drafts 5 candidate solution descriptions from T and iteratively refines them using each paper summary in P"? It is not clear and appears to be super important.
- How many hyperparameters are there total for this agent? It appears to have a lot, and I would be curious how those hyperparameters affect the solutions. (ie. run time, integer values used in various places, utility scalars, probabilities, bug-fix attempts, etc..)
- Which hardware is used for all of this? 8 hours means a totally different thing on a laptop vs. a data center. Are all models tested on the same hardware? I see some models are only accessible on the web: how do you account for different compute capabilities?
- How many runs were used for ablation analysis (Table 3)? Was it just run once, or several times with mean score? If it was run once then it needs to be run more times, since LLMs are very stochastic and the results can vary significantly.
- Where is the conclusion/future work section? The paper seems to end abruptly.

## Feedback
- The code is a mess. There are import statements all over the place, comments are unstructured and lacking in many places, and there are no clear instructions on how to reproduce results. Please fix this so I can take a look at the code myself and run things locally.
- I_{diag} is not updated in Algorithm 1, it appears to remain as an empty list. Please consider removing or correct this.
- Line 265: Change "instruction by first uniformly draws" to "instruction by first uniformly drawing".
- The claims (line 297, first paragraph in section 4.1) that TusoAI provides "novel" and "computationally efficient" methods are not substantiated. Provide quantitative results for computational efficiency and a better empirical comparison of the results across all models. Simply stating the method TusoAI came up with does not mean it's novel.
- Please make it clear which model is presented in figure 2B in the caption. This specific sub-figure would benefit from a better description in general.
- Please provide motivation for why we care about generating diverse code. I would argue that if an agent can produce better methods, then it doesn't matter how diverse the methods produced along the way are. This is a central point studied in the paper that needs to be substantiated. In fact, producing lots of diverse models is at odds with the 8 hour time limit, since I can also imagine a model would benefit from trying to iterate on a model that works decently well instead of making smaller changes to a larger amount of models in the same period of time.
- Line 386, please correct: "Table ??".
- Line 430: "This improvement reflects TusoAI’s ability...", is not really a fair comparison. TusoAI started from the work the original authors had already done. Please correct this sentence to show how TusoAI expands upon the original work in a short amount of time, instead of making it seem it tested 167 unique versions _from scratch_.
  - Same comment for line 464 for the second case study.
- The text in the figures are tiny. Please make the text larger so it's easier to read instead of having to zoom in 300% to make out what it says.
- Line 1244: "It's samples" change to "Its samples".

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
The paper presents TusoAI, a method that hand designs a set of steps for automated model discovery using LLMs. The authors compare their method against baseline methods on set of task and find that their approach achieves good performance with high diversity in the suggested solutions.

### Strengths
- the problem of automated computational model design is important 
- ablation studies are provided to reveal some insight into the importance of the methods components
- strong results are achieved on a set of tasks

### Weaknesses
- the paper lacks any discussion or conclusion
- the Bayesian update step is unclear and not described
- the method seems overall very heuristic. A (theoretic) justification of the algorithm would be of great value.
- additionally the algorithm is rather complex and the explanation are difficult to follow at times
- figure are sometimes unclear and would need re-working

### Questions
- what does your acronym TusoAI stand for?
- Table 1 and 2: How is performance defined in these tasks?
- Is the comparison in the experimental section fair in terms of computational load? How long are baseline methods run compared to the 8h for TusoAI?
- Figure 2, 3 and 4: font size is too small and hard to read.
- line 323: How were the text embeddings computed?
- Figure 2B: Does this "optimization trajectory" look similar for all other tasks? Are there examples where the algorithm converges fast and then does not find a better solution?
- In the methods there is no explanation of how the Bayesian update is performed. Can you explain where and how exactly a Bayesian update is performed and what the prior and posteriors involved are?
- Table 3: Do the ablations only disable one of the parts (i.e. "No Bayesian" only disables the Bayesian part but keeps all other parts)?
- line 380: How do you define the mean time to optimize? What is the criterion that optimization is achieved?
- line 385: "Results are shown in Table ??"
- Figure 3 B/C: What does the circles show? How should we read and interpret those figures?
- Figure 4:
  - what does the distance threshold mean?
  - How should read and interpret figure 4C?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper develops TusoAI, an agentic AI system that, given a scientific task description and an evaluation function, autonomously develops and optimizes computational methods for the application. TusoAI is tested across several single-cell tasks and some general deep learning tasks.

### Strengths
1) Mixing Bayesian optimization in the loop of an agentic AI system is interesting. I wish the paper had substantiated whether this is the first agent of this type or whether the idea is inspired by existing agents. 

2) The results show that some of the methods constructed by TusoAI are novel rather than simple re-implementation of existing approaches or calls to standard packages.

3) The agentic system is tested in many scenarios, from final performance to behavior over time.

4) Single-cell results show promising biological results.

### Weaknesses
1) The writing can be improved. For example, the Abstract is excessively long. Instead of listing all results, please summarize the key points and condense the Abstract's opening.

2) While the evaluations involve tasks beyond a single cell, it seems like TusoAI's performance is more pronounced in the single cell case, which might suggest the paper could have made that the central focus rather than a general-purpose multi-agent AI.

3) The code is not available for a full review of the paper.

4) Narrow scope to problems that can be solved using off-the-shelf ML models and strategies rather than those that need in-depth model development (e.g., new deep learning models rather than say fine-tuning Restnet). 

5) As discussed in the *LLM-based general learning agents* section, there are already agents developed for ML engineering. This work differentiates itself from those by focusing on scientific domains, but some baselines are still missing (R&D, DS, etc).

6) Limitation in scale: only 10 papers retrieved from Semantic Scholar.

7) I have some concerns about model selection and evaluation, which I have asked in the box below.

### Questions
1) The key motivation behind TusoAI is to develop new computational solutions to existing problems in science.  Do the results show a new computational method developed beyond your expectations? One that is far beyond the reach of human ML engineers?

2) Is there an underlying optimization problem that the Bayesian framework is solving? 

3) What is the convergence criterion of the system? Figure 2A and 2B suggest that TusoAI never really converges, as code diversity oscillates and performance dips. Is this expected or desired? How did you pick when the algorithm should be terminated? Are there any prompting guardrails that avoid oscillations?

4)  Does increasing the number of papers retrieved from Semantics Scholar (from 10) improve the results?

5) Can you elaborate on how the paper ensures the model selection has been done in an apples-to-apples way across the datasets/baselines? Is model evaluation also handled by TusoAI or externally?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TusoAI, an agentic AI framework that automatically develops and optimizes computational methods for scientific tasks. Given task descriptions and evaluation functions, it can generate new algorithms rather than just run existing analysis pipelines. The main idea is to structure domain knowledge into a “knowledge tree,” combine it with Bayesian-guided iterative optimization and diagnostic feedback, and iteratively generate, implement, and evaluate candidate solutions. On 11 benchmarks, TusoAI achieved a higher average rank compared to expert-designed methods and existing AI agents.

### Strengths
- Clear motivation and problem framing. The paper focuses on creating methods rather than just executing pipelines. It specifies the objective and agent loop, including the task description and the evaluator $h(\cdot)$, with cold/warm start.
- Breadth of benchmarks with application case study. Empirical scope spans 11 tasks across two domains (six single-cell, five scientific deep learning).
- The warm-start improvements on scDRS and pgBoost are interesting. It is interesting to consider concrete empirical deltas and biological findings (e.g., new T-cell disease associations; rs138917529→GCK link).

### Weaknesses
- Too many symbols in Algorithm 1. If the authors want to introduce these annotations, they should include an appendix table.
- Baseline coverage is insufficient, mainly including general-purpose agents (AIDE, Biomni, ChatGPT-Agent) and simple expert or AutoML models. It lacks domain-specific, publication-level baselines for each scientific task. Including parts of the leaderboard is recommended.
- Ablation and diagnosis analyses lack depth. The ablation study in Table 3 mixes tasks from unrelated domains. It omits per-task variance or statistical testing, making it impossible to isolate the contribution of each component (knowledge tree, Bayesian updates, diagnosis). 
- No detailed analysis of failure cases or performance drops is provided, so readers cannot understand when the system fails or regresses.
- Case studies serve as illustrations but lack control. The genetic applications (scDRS, pgBoost) show improvements, yet they lack independent validation or error analysis.

### Questions
- Can the authors clearly specify what metric is used in each column of Tables 1 and 2, and how the “Avg” and “Avg rank” values are computed across heterogeneous tasks? If normalization was applied (e.g., 1 – MSE, scaled decomposition R²), please make this explicit in the main text or table captions.
- How do the qualitative examples, such as NMF with dropout/Poisson modeling and Satellite ensembling, support the narrative of the methods being "distinct" or "custom"?
- Is using text similarity (Fig. 2A) a sufficient proxy for diversity, or should a more principled novelty metric be employed in the evaluation?
- This work’s broad, application-driven scope and descriptive evaluation would be better suited for an applied-science journal rather than a methodology-focused venue like ICLR.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TusoAI, an LLM-based agentic system for automatically developing and optimizing computational methods for scientific applications. The system structures domain knowledge into a hierarchical knowledge tree, uses Bayesian updates to sample optimization strategies across categories, and incorporates diagnostic feedback during iterative refinement. The authors evaluate TusoAI on 11 scientific benchmarks (6 single-cell analysis tasks and 5 deep learning tasks), showing consistent improvements over baselines including AIDE, Biomni, and expert-designed methods. Two case studies in genetics demonstrate practical utility by improving existing methods (scDRS and pgBoost) and uncovering novel biological insights.

### Strengths
1. The paper evaluates TusoAI on 11 tasks spanning single-cell analysis and scientific deep learning, demonstrating broad applicability. The inclusion of both cold-start and warm-start settings, along with real-world case studies in genetics, strengthens the practical relevance.

2. The knowledge tree representation provides a structured way to organize domain knowledge, and the Bayesian update mechanism offers a principled approach to balancing exploration and exploitation. The diagnostic-based optimization is a nice addition that grounds the system in empirical data patterns.

3. TusoAI consistently outperforms strong baselines including AIDE and domain-specific expert methods. The ablation studies effectively demonstrate that each component (categories, Bayesian updates, diagnostics, domain knowledge) contributes meaningfully to overall performance.

### Weaknesses
1. The core algorithmic components are largely existing techniques (LLM agents, tree-based search, iterative optimization), and the main contribution is their combination for scientific method development. The concurrent work by Aygun et al. (2025) appears to address very similar problems, but the paper dismisses direct comparison due to unavailable code without sufficiently differentiating the approaches conceptually.

2. The 8-hour optimization budget is relatively short and may favor methods that converge quickly over those requiring longer refinement. The diversity analysis (Figure 2A) shows TusoAI explores more than AIDE, but it's unclear whether this translates to better generalization. The selection strategy (shortest code within 0.1% of best performance) seems ad-hoc and could inadvertently favor simpler but less robust solutions.

3. The paper doesn't discuss when TusoAI fails or performs poorly, nor does it address practical concerns like how to set appropriate evaluation functions for novel scientific problems, how sensitive the system is to task description quality, or how to validate generated methods when ground truth is unavailable. The computational cost analysis is limited to one brief mention ($0.37-$0.41 for case studies).

4. The case studies claim to discover "9 new associations" and "7 previously unreported links," but these are computational predictions that would require experimental validation to be considered true biological discoveries. The statistical methodology for declaring associations as "novel" versus "missed by previous methods" is not clearly described.

5. While the authors promise to release code upon publication, key details are missing: which specific LLM API versions were used, how were papers selected from Semantic Scholar (just citation count?), what are the exact prompts for different agents, and how stable are results across different random seeds beyond the reported 95% CIs?

### Questions
1. How does TusoAI handle the cold start problem for truly novel scientific domains? The knowledge tree construction relies on retrieving papers from Semantic Scholar based on task descriptions. For emerging research areas with limited literature, how does the system bootstrap domain knowledge? Have you tested TusoAI on tasks where relevant literature is scarce or the task description is deliberately vague?

2. Can you provide more details on the comparison with Aygun et al. (2025) and clarify the key technical differences? Beyond the unavailability of their code, what are the conceptual and methodological differences between your approach and theirs? This seems critical for establishing the novelty of your work.

### Soundness
2

### Presentation
3

### Contribution
3
