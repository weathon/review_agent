# HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations—not just a single correct answer—becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe—rather than a leaderboard—for methods that explicitly explore and cover admissible explanation spaces.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the ability of LLMs to infer multiple hypotheses when fitting the observations of under-determined problems. The paper introduces 3 toy problems with exact enumeration of hypothesis space, which enables systematic evaluation of LLM solutions' validity, uniqueness, and recovery. Empirically, the authors find that frontier models often find valid solutions, but suffer mode collapse as the space of possible solutions increases.

While the motivation and paper are well written, I am leaning towards rejection because (1) the empirical findings are not novel, (2) the artificial nature of the problems limit the conclusions that can be made, (3) many of the reasoning LLMs are acing the first two tasks at all difficulty levels, limiting its discriminatory power.

1) The main experimental conclusion of the paper: "mode collapse of solutions in larger solution spaces" has been reported before. See [Duan et al., NeurIPS 2025](https://arxiv.org/abs/2507.02083)
2) The three problems are artificially constructed such that the solution space is known a priori. This seems very restrictive to me and fundamentally limits the generalization power of conclusions made via this diagnostic. It also seems misleading to me that the authors have chosen domain-specific names like "DNA interaction" and "3D understanding", when the actual problems are extreme simplifications of these complex problems.
3) Could the authors perhaps explore more difficult versions of the first two problems in Table 1? It seems too easy at the moment for the reasoning models.

### Strengths
- good motivation, clear writing for the most part
- measuring the divergent creativity of hypotheses from an information theory perspective is interesting
- experiments are sound and thorough

### Weaknesses
- originality of experimental findings is limited
- artificial nature of problems limits conclusions
- diagnostic lacks discriminatory power

### Questions
- Can the authors explain how they calculated the entropy gain in section 3.3? Are they counting the number of times the LLM repeated each solution to estimate the empirical frequency of that solution? If so, doesn't this defeat the purpose of using the actual posterior probability of each solution given the observations? I thought that $P(h_i | O)$ is a property of the problem itself, not of the LLM.
- Can the authors provide the prompt used to obtain solutions from the LLMs? Is the ability of finding diverse hypotheses confounded by the model's ability to follow prompts? Is the prompt explicitly encouraging diverse mode exploration?

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
2

### Summary
This paper introduces HypoSpace, a diagnostic suite designed to evaluate LLM creativity as the ability to generate sets of valid, non-redundant hypotheses under scientific underdetermination. Instead of giving score to single correct answers, it proposes to measure how well a model explores and covers all admissible explanations consistent with the same observations by defining three complementary metrics, including Validity (VR), Uniqueness (NR), and Recovery (RR). As experiments, the authors apply them to three designed tasks: causal graph inference, 3D voxel reconstruction, and Boolean genetic interaction. Empirical experiments across frontier LLMs such as GPT-5 reveal that high validity but strong degradation in uniqueness and recovery as the hypothesis space enlarges, reflecting the (so-called) mode collapse in reasoning.

### Strengths
- Empirical results revealed some insight regarding the “mode collapse phenomenon” in reasoning models with the proposed VR/NR/RR that traditional accuracy-based metrics cannot capture.
- Proposed three relevant diagnostic tasks including causal graph inference, 3D voxel reconstruction, and Boolean genetic interaction, to study the capability of LLMs for sampling hypothesis. If these are open-sourced, it can help the analysis of LLMs in this regard for future research.

### Weaknesses
- The evaluation tasks are relatively toy and over-simplified. Being far from realistic scientific inference scenario can weaken the contribution of this study. One case study of real-world problem can help.
- The proposed indicators can serve as sufficient measure for simple hypothesis generation cases like the used three tasks, but may probably not be qualified for complex problem where scientific hypothesis can be tricky to evaluate with them (in my opinion).

### Questions
- Does the degradation in uniqueness and recovery measures is caused simply by the increase of “difficulty” of commonly studied reasoning problem? Can you explain what are the difference between increased hypothesis space size and increased difficulty in the context of this paper?
- When the |H_o| grows, does the corresponding change of sampling hyperparameters like temperature and top-p value helps improve the validity or uniqueness metrics? Since the problem scales up, I assume the model may need to re-balance the exploration somehow.
- Corresponding to one of the concerns above, can you also show some real-world or open-ended problems of generating hypothesis? This may make the proposed diagnostic framework, HypoSpace, more convincing and useful.
- What strategies do you think will ameliorate the collapse observed from the results? Via training-free prompting or proper finetuning?

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
Utilizing LLM's potential for scientific discovery is an exciting and challenging problem that has been a focus recently. However this work proposes a clean and effective way of identifying how good current models are in proposing supporting hypothesis for the observed data. I like the framework as it provides a holistic way of evaluating for this task, which goes beyond identifying the accuracy of the hypothesis which prior works optimise for. By evaluating validity, uniqueness and recovery, the framework identifies if models can act as strong samplers of all possible hypothesis which can be helpful for scientific discovery. Unlike most creativity or hypothesis-generation benchmarks that depend on human or LLM-as-judge scoring, HypoSpace uses deterministic validators and exactly enumerated admissible sets, removing subjectivity and enabling reproducible, model-agnostic evaluation

### Strengths
Overall I believe this work is strong, some of its strengths are:

1. I like the tasks this work targeted, they seem novel and widely different from typical applications past work evaluate on. 

2. This combination of ideas from psychology’s notion of divergent thinking with a concrete, enumerated approach to evaluating hypotheses feels genuinely fresh and thoughtfully put together.

3. Rigorous empirical evaluation of current frontier models is useful for understanding the current state of LLMs for this task

4. The modularity of the framework makes it directly applicable to other tasks, improving its potential generalization and usability in real world.

### Weaknesses
There are some aspects the work can improve upon:

1. While the paper quantifies exploration failure nicely, it doesn’t deeply analyze why models prefer certain hypotheses or how internal reasoning traces differ between models.

2. I am curious and confused about one thing: is it necessary for an agent to identify all possible solutions which justify the given observation data? I believe validating the number of potential hypothesis a model can come up with makes sense for certain tasks (more specifically when the subjectivity of the task is high), while identifying a smaller subset or getting the correct answer might be of higher importance in other cases.

### Questions
1. It would be interesting to see in how many cases the subset of hypothesis identified by the model were the most optimal solutions, or in how many cases the subset of defined hypothesis did not contain the top 3 or top 5 optimal solutions? In causal graph example, an optimal solution could potentially be given the nodes, which graphs require the smallest connections to justify the given observations?

2. I’m curious about why certain hypotheses are favored more frequently than others. It might be interesting to analyze which samples are easier for models to estimate and which are harder, and then compare their coverage. Such an analysis could guide targeted training to improve how LLMs explore hypothesis spaces. For instance, in the causal inference task, some graphs might only require adding a single edge to satisfy the interventional observations, while others may need multiple structural changes. If models consistently prefer these “easier” hypotheses, it would point to a deeper limitation in their ability to explore more complex or less accessible regions of the hypothesis space.



3. I am also curious if there is a way to identify if the model is actually equipped enough in identifying all possible hypothesis, or is it not good at this task. For example this is the first work I have seen where LLM is evaluated on 3D pixel reconstruction, is it possible that the model is just bad at this task, and therefore evaluating its ability to generate all possible hypothesis will be limited? It would be impactful to find simple cases where the model tends to have a high performance in getting the correct answer, but fails on uniqueness and recovery metrics.

I am happy to raise my scores, if the authors can provide some answers for the questions

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HypoSpace, a framework for evaluating large language models as set-valued hypothesis generators in underdetermined reasoning settings—where multiple valid explanations exist for the same evidence. Rather than focusing solely on accuracy, HypoSpace quantifies three dimensions of model performance: Validity, measuring the proportion of correct hypotheses; Uniqueness, capturing the diversity of generated solutions; and Recovery, reflecting how completely the model explores the admissible hypothesis space. The benchmark spans three structured domains—causal graph inference, 3D voxel reconstruction, and Boolean genetic interaction modeling.

### Strengths
The framework is methodically clean, deterministic, and reproducible. It formalizes a relatively underexplored dimension of LLM evaluation—output diversity—in a controlled setting. The domains span causal, spatial, and symbolic reasoning, and the inclusion of exact validators is elegant from a benchmarking standpoint. The analysis is thorough, and results are clearly visualized. From an engineering perspective, the pipeline for enumerating hypothesis spaces and evaluating them under canonicalization is well executed.

### Weaknesses
The paper’s core motivation is underdeveloped: it assumes that producing multiple distinct hypotheses is intrinsically valuable without explaining when or why that matters for reasoning or creativity. In most real reasoning contexts, the goal is not to enumerate all possible explanations, but to converge efficiently on the most plausible or useful one. Thus, the emphasis on “recovery” and “uniqueness” risks being arbitrary rather than insightful.
The evaluation tasks are artificial, limited to small, fully enumerable spaces that have little connection to real-world cognitive or scientific problems. High validity scores in the causal and 3D tasks indicate that the framework measures diversity in trivial domains, not meaningful reasoning skill.
The difficulty scaling is not adaptive; as task complexity grows, models degrade simply because they are not tuned for those domains, not because they lack creative capacity. This confounds interpretability of the NR/RR decline.
Finally, the claim of novelty is overstated: evaluating exploration in LLMs has been explored through uncertainty estimation, ensemble sampling, and information-theoretic diversity measures. HypoSpace provides a clean but limited instantiation of an old idea, not a new paradigm.

### Questions
Why is generating multiple hypotheses preferable or necessary compared to generating a single, well-reasoned explanation?

How do VR, NR, and RR translate to practical reasoning competence in real domains such as scientific discovery, design, or causal inference at scale?

If causal and voxel domains saturate easily, do these results truly measure reasoning or simply expose ceiling effects in trivial environments?

How would the authors calibrate difficulty relative to model capacity to avoid conflating lack of domain tuning with reasoning limits?

Could similar insights be achieved through simpler sampling-based diversity metrics rather than elaborate enumerated hypothesis spaces?

### Soundness
3

### Presentation
3

### Contribution
2
