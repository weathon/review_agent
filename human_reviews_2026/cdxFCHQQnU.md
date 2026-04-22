# LeGIT: LLM Guided Intervention Targeting for Online Causal Discovery

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 8, 2

## Abstract
A fundamental challenge in online causal discovery is designing effective experiments by selecting optimal intervention targets. Conventional numerical methods struggle in the early stages when limited interventional data is available, often yielding noisy or misleading selection guidance. In this work, we introduce the Large Language Model Guided Intervention Targeting (LeGIT), a novel collaborative framework that synergizes the vast world knowledge of LLMs with the precision of numerical algorithms. By analyzing the meta-information of the causal system, it proposes highly informative intervention targets, effectively bootstrapping the discovery process to augment existing numerical approaches, while retaining the convergence guarantees. Evaluated across four realistic benchmarks, LeGIT demonstrates significant improvements in performance and robustness over existing methods, and even surpasses humans. This work establishes that LLMs can play a pivotal role in experimental design, offering a scalable and cost-efficient strategy to accelerate causal and scientific discovery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a novel hybrid framework for causal structure learning that combines the vast prior world knowledge of LLMs with existing methods (referred to as numerical methods). The method consists of four stages: (i) Warmup Stage annd (ii) Bootstrapping Stage where LLMs are employed, following by a (iii) Re-Sampling Stage and (iv) Numerical-Method Stage** where the search is continued with “traditional” causal discovery methods. Eventually, the authors assess the proposed method on four datasets of the BN benchmark suite, and compare the performance to three online causal discovery methods, as well 3 heuristics and a human reference score.

### Strengths
- The idea to leverage the vast world-knowledge of LLMs as a prior for causal discovery methods is definitely an interesting idea that deserves more exploration.
- Clean Visualizations

### Weaknesses
- **Clarity:** The paper is somewhat hard to parse. Despite reading section 3.2 multiple times, I still find it hard to understand the proposed method in detail and cannot confidently say that I am 100% understanding what's going on.
- **Variance:** The variances in table 1 seem to be huge. When considering the standard deviation over 5 seeds, GIT and LeGIT seems to be overlapping in most cases, making it hard to make a conclusion about the effectiveness of the proposed approach.
- **Contamination:** While the author address the issue of contamination (lines 100 to 108), they refer to Long et al. (2023) as an evidence that prominent LLMs struggle to reconstruct causal graphs. However, the referred work tested GPT3 while the present work builds upon OpenAI's 4o. LLMs have come along way since GPT3, and it's very likely that results would look different with today's state of the art models, especially reasoning models. Hence, the effect of contamination cannot really be judged given the presented evidence. Instead of asking for interventional targets, it would be worthwhile to check which edges the LLMs are able to identify?
- **Theoretical Argument:** The theoretical argument is somewhat superfluous -- could you please provide more depth on what the theoretical argument provides? The entire theoretical novelty hinges on empirically verifying Assumption D.2. If the authors cannot show the LLM targets consistently satisfy this property, the whole theoretical arguments becomes a tautology with inherited proof from other work.

### Questions
- Could you please provided a clearer description on the algorithm?
- Have the authors also experimented to initialize the graph with LLM proposed edges instead of using the LLM to suggest informative intervention nodes? 
- Have the authors experimented with different prompts in the warmup stage? I suspect that the LLM's performance on discovering useful variables may largely vary between prompts. And the given prompt seems to be slightly ambigious -- so I would be curious how well the model performs with a more precise prompt.
- Could the authosr please run LLM baselines where the model directly predicts causal edges among given variables.
-  Line 86: "Relatively clearer" --> what do you mean with this?

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
The paper introduces LeGIT, an intervention-targeting method for experimental design in online causal discovery. LeGIT leverages the meta-knowledge embedded in large language models (LLMs) to identify which nodes should be selected for intervention. The LLM-enhanced framework can be integrated with any causal discovery algorithm that utilizes interventional data. In this setup, the LLM receives a prompt summarizing the metadata of the graph’s vertices (e.g., variable descriptions) and responds with a set of proposed interventional nodes based on its internal and contextual knowledge. The framework further incorporates bootstrap and resampling stages to enhance performance and is compared against alternative intervention-selection algorithms that do not exploit such meta-knowledge.

### Strengths
The topic addressed in the paper is highly engaging, as it pushes the boundaries of intervention targeting in causal discovery by integrating external meta-knowledge from large language models (LLMs). Exploring the applicability of LLMs to this problem offers an insightful avenue for assessing their reasoning abilities and the extent of their embedded common and expert knowledge. The idea of treating an LLM as an artificial expert in the absence of a human counterpart is particularly compelling, making the comparison between LLM-selected and human-selected intervention targets especially valuable.
I particularly appreciate the experiments presented in Figure 2, which provide intuition on how the set of intervention vertices may vary across methods, and Table 2, which—especially when compared to Table 1—illustrates how LeGIT’s performance is influenced by changes in the data budget. However, I would strongly encourage the inclusion of additional experiments analyzing how the proposed enhancements of LeGIT affect overall performance (see the discussion on soundness in the weaknesses section).

### Weaknesses
**Notes on Soundness:**

My most serious concern is that the comparison between the intervention selection algorithms and the LLM-enhanced algorithm is not entirely fair. This is because the LLM has access to additional metadata related to the studied graph, leveraging knowledge that the other algorithms were never designed to use (except for the input provided by human experts). In other words, the assumptions regarding the input data for the causal discovery problem have been extended. It is therefore quite expected that incorporating this extra knowledge—on top of the standard framework—would lead to better results. This is not to say that the impact of such an LLM-based enhancement is uninteresting to study; however, it is important to recognize that LeGIT addresses a different problem than, for example, GIT, AIT, or CBES. This distinction, however, is not reflected in the narrative presented by the authors. 

I would suggest that the authors focus on comparing their updated Online Causal Discovery framework with other LLM-guided causal discovery algorithms. For instance, the works of Jiralerspong et al. and Khatibi et al. rely on similar access to metadata from LLMs to uncover causal dependencies, and they use the same or similar graphs and metrics (other comparable algorithms are also summarized in Wan et al.). The experiments with GIT, AIT, and CBES serve more as a teaser or example of LeGIT’s potential, rather than as a fair comparison between methods that have access to the same data.

The authors could therefore also adjust their narrative to focus on understanding the impact of incorporating this additional knowledge. For example, which nodes are most affected? (Figure 2 is a nice example, although it might be more informative to compare against a simpler case where the ideal distribution can be directly derived. In the graphs from Figure 2, it is difficult to assess which intervention set is correct, or whether the interventions occur on edges that can be reversed within a MEC class—and hence whether the intervention is actually required to achieve identifiability.) Similarly, how do the different enhancements (bootstrapping, resampling) affect convergence or final performance? Does including information about the current graph in the prompt improve or deteriorate performance? How does performance change with the number of nodes proposed by the LLM—and can the LLM propose a ranking of nodes? These are just some examples of how the authors could build a stronger narrative around the ablation studies and deepen the understanding of their framework  while staying within the same online causal discovery framework.

**Notes on Experiments Setup:**

Throughout the paper, the authors use only four graphs of more or less similar size. This is, of course, a reasonable starting point, but I wonder why these specific graphs were selected. More importantly, I see no clear restriction preventing the extension of the evaluation to other networks (e.g., additional graphs from https://www.bnlearn.com/bnrepository/
Moreover, based on the description of LeGIT, it appears that the method can be used with any algorithm $A$ that satisfies the conditions for online causal discovery (i.e., potentially not even gradient-based). Did the authors attempt to use different base causal discovery algorithms besides ENCO? Even if not as a main experiment, such a demonstration could nicely illustrate the universality of the proposed approach.

**Notes on Contribution:**

I find the topic quite interesting, as it explores the integration of LLMs’ common meta-knowledge—akin to the bias present in human experts—into causal discovery. However, this perspective has also been explored in prior work (e.g. Jiralerspong et al., Khatibi et al., Wan et al.). While focusing on scoring the intervention potential of each node appears novel, the experimental design (see “Soundness” above) does not give me enough confidence to fully assess the strength of the contribution.

**Notes on presentation (Minor):**

The paper is generally well written, but I do think that some of the sections could use more attention. For instance, Preliminaries  do not explain all the notations in Algorithm 1 (e.g. what is $P(G)$), there is a discrepancy between $\phi$ in  Algorithm 1 (in “Output”) and in text (line 133). It is not explained what those parameters are and how $A$ updates them (in practice we also have functional parameters that are also updated by the algorithm A and can thus influence the structural ones - see Lippe et al.). In Algorithm 2, line 8, should it not be $D_{bootrstapped}[i-T_{warmup}]$?  

**References:**

Jiralerspong, Thomas, et al. "Efficient causal graph discovery using large language models." arXiv preprint arXiv:2402.01207 (2024).  
Khatibi, Elahe, et al. "Alcm: Autonomous llm-augmented causal discovery framework." arXiv preprint arXiv:2405.01744 (2024).  
Wan, Guangya, et al. "Large Language Models for Causal Discovery: Current Landscape and Future Directions." arXiv preprint arXiv:2402.11068 (2024).  
Lippe, Phillip, Taco Cohen, and Efstratios Gavves. "Efficient neural causal discovery without acyclicity constraints." arXiv preprint arXiv:2107.10483 (2021).

### Questions
The current algorithm primarily relies on meta-knowledge derived from the descriptions of the variables. I was wondering whether the authors also explored incorporating contextual information from the graph itself (e.g., details about the MEC class) or integrating principles from do-calculus or v-structure verification. Could such approaches work even in cases where no node descriptions are provided? In other words, could the LLM base its reasoning solely on causal discovery knowledge and the graph structure to refine the current solution and subsequently select the intervention nodes? In that way LeGiT could be used even on fully artificial graphs, or models when no previous bias/knowledge is available about the nodes. 

In Figure 4, it appears—especially for the Child and Fisram graphs—that LeGIT does not improve during the early acquisition steps and then sharply drops. Could this breaking point correspond to when the numerical scoring algorithms start to take effect?

In Algorithm 2, the LeGIT intervention targets are selected at the beginning of training. Did the authors examine whether incorporating them as the final interventions (or interleaving them with the numerically scored ones) provides any benefit?

I do generally believe it could be an interesting paper, but currently it suffers from its positioning (i.e. the algorithm works well, but it is either not fairly comparable to other approaches - beacuse of the additional knowledge - or other LLM-guided causal discovery approaches could be adapt to work within the online causal discovery framework to allow for a comparision with meta-data).

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a framework to incorporate LLMs into existing gradient-based intervention targeting methods. Specifically, the paper proposes using LLMs in a warmup phase that includes an initial guess for the causal graph based on prompting the LLM with the variable names in the dataset.  The paper performs ablations and shows that on 4 datasets, LeGIT, using GIT as underlying gradient-based intervention targeting method, outperforms vanilla GIT as well as a version of LeGIT where the LLM was replaced by human feedback.

### Strengths
Presentation of work is clear and results seem positive; the idea of using LLMs to resolve the issue of instability under certain initializations for gradient-based intervention targeting methods seems novel to me.

### Weaknesses
Experiments on only four standard datasets could be too few to draw statistically significant conclusions, especially if any optimizations (e.g. tuning prompts; tuning length of the warmup phase) have been performed using these datasets. It would be good to see robustness to cases where datasets are less well-known and LLMs may have difficulty coming up with a reasonable guess of the underlying structure.

### Questions
- Experiments have been performed iterating over 5 seeds. Are 5 seeds sufficient to obtain statistically significant results? 
- How was tuning the length of the warmup relative to the length of the experiments done? Are any plots showing performance under varying warmup-length? 
- Is there a possibility to experiment on more novel datasets where guessing the causal graph may be difficult for an LLM? It would be interesting to see to what extent LeGIT still outperforms or whether incorporating LLMs still helps in scenarios that are closer to "unsolved" cases, where there is much less background knowledge available to the LLM; as this would be relevant for real research problems.

### Soundness
3

### Presentation
3

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
The work explores the usage of LLMs for experimental design. The authors propose a framework and prompts for acquiring information about root causes in the network from LLMs. This information guides further experimentation. The approach is compared to several baselines on four real-world inspired problems using the ENCO algorithm to obtain structure posteriors.

### Strengths
1. The paper touches on an important problem of efficient experimental design.
2. The paper is easy to read.

### Weaknesses
1. The method description is unclear. The “Re-sampling stage” paragraph states that each LLM suggestion undergoes filtering (must “survive” two independent votes). Meanwhile, in Algorithm 2:
* In line 4, we intervene on all variables from the  warmup list
* In line 8, we intervene on all variables from the bootstrap-warmup list
* In line 12, we intervene on all variables from both datasets once again

Please clarify this discrepancy in descriptions.
Also, in the bootstrap stage, how is the intermediate causal discovery result leveraged?

2. The proposed method relies on LLMs identifying root causes. What is the theoretical motivation for including such a suggestion in the prompts? In general, selecting optimal experimental designs is a more complex task [Eberhardt]. 

3. The observed effectiveness of the method may relate to how the ENCO method works and may fail to generalize. Note that ENCO has no convergence guarantees when the number of interventions is less than d-1. The ENCO framework only guarantees the recovery of the correct graph when interventions were conducted on at least n-1 nodes. It would be beneficial for the applicability (and possibly performance) to use a method with stronger theoretical guarantees under partial intervention sets, for example, DCDI or DIBS.

4. The approach relies on the ability of LLMs to reason about the variables. This ability can be impaired in at least two cases: insufficient information about variables (when the descriptions are missing or are non-informative), lack of background knowledge. The discussion on these two limitations is missing from the paper. When applying this method to a new setting, how can I make sure both requirements are fulfilled and trust in the effectiveness of the method? The work also lacks discussion about the effectiveness of the proposed prompt and the consistency of LLMs' responses.

5. There are no experiments or analyses that evaluate the scalability and cost efficiency of the proposed method to support he claim “LLMs offer a scalable and cost-efficient approach to enhance experimental design” (line 485). Moreover, since LeGIT requires using a numerical intervention targeting method, the cost efficiency and scalability seem to be similar to existing methods.

**References**

[Eberhardt] Eberhardt, Almost Optimal Intervention Sets for Causal Discovery

[DCDI] Brouillard et al., Differentiable Causal Discovery from Interventional Data

[DIBS] Lorch et al., DiBS: Differentiable Bayesian Structure Learning

### Questions
1. What is the prompt for the boostraped-warmup intervention set acquisition?
2. Why are the results on statistical significance, in section 5.2, partial? How do the test results look for other considered graphs?
3. Line 468: “rapid interventions are required” - Can the Authors provide an example of a real-world problem where rapid experimentation is needed and is costly?
4. Line 93 typo: intervening -> intervention

### Soundness
1

### Presentation
3

### Contribution
1
