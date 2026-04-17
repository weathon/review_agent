# LLM Routing with Dueling Feedback

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
We study LLM routing, the problem of selecting the best model for each query while balancing user satisfaction, model expertise, and inference cost. We formulate routing as contextual dueling bandits, learning from pairwise preference feedback rather than absolute scores, thereby yielding label-efficient and dynamic adaptation. Building on this formulation, we introduce Category-Calibrated Fine-Tuning (CCFT), a representation-learning method that derives model embeddings from offline data using contrastive fine-tuning with categorical weighting. These embeddings enable the practical instantiation of Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB), a theoretically grounded posterior-sampling algorithm. We propose four variants of the categorical weighting that explicitly integrate model quality and cost, and we empirically evaluate the proposed methods on the RouterBench and MixInstruct datasets. Across both benchmarks, our methods achieve lower cumulative regret and faster convergence, with better robustness and performance-cost balance than strong baselines built with a general-purpose OpenAI embedding model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the LLM routing problem that is dynamically selecting the best-fit large language model for each query under constraints of user satisfaction, model expertise, and cost. The authors cast routing as a contextual dueling bandit task, where the system learns from pairwise preference feedback instead of absolute quality scores. They introduce a novel Category-Calibrated Fine-Tuning (CCFT) method to learn model-representative embeddings via contrastive fine-tuning with categorical weighting, encoding each model’s domain expertise. These embeddings enable an implementation of Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB), a Thompson-sampling algorithm adapted to preference feedback. The approach is evaluated on two benchmarks, RouterBench (a multi-task LLM routing dataset) and MixInstruct (an instruction-following benchmark). The experiments show that the proposed routing strategy achieves lower cumulative regret and faster convergence than baselines using generic embeddings, while maintaining better robustness and a favorable performance-vs-cost tradeoff. In summary, the paper’s contributions include formulating LLM routing as a dueling bandit, a new CCFT embedding strategy (with four variants integrating model quality and cost), and a theoretically grounded FGTS-based router that outperforms strong baselines on diverse tasks.

### Strengths
- Problem Formulation: The paper tackles LLM routing in a novel way by combining online adaptive learning with preference-based weak supervision. This joint treatment of adaptivity and pairwise feedback fills an important gap, as previous work had not simultaneously addressed both aspects. The use of dueling bandits (pairwise comparisons) for model selection is a creative and timely idea, reflecting how user feedback can be more practically obtained as comparisons rather than absolute ratings.

- Theoretical Foundation: The approach is built on a solid theoretical framework. It leverages Feel-Good Thompson Sampling (FGTS), which has known robust guarantees for bandits, and extends it to contextual dueling settings. The paper provides theoretical justification (Proposition 1) linking the learned embeddings to unbiased utility estimates, and clearly explains how binary preference feedback maps onto an underlying utility function of user satisfaction, expertise, and cost. 

- Effective Methodology (CCFT): The proposed Category-Calibrated Fine-Tuning strategy is a general yet powerful way to encode model specialization. By fine-tuning language model embeddings on representative queries per category, the router’s context features become highly informative of which model will perform well. This is the first demonstration of a trainable contextual dueling bandit for LLM routing. Technically, the paper integrates techniques like contrastive learning and softmax-weighted embeddings in an elegant manner, and the algorithmic solution (FGTS.CDB) is well-motivated.

- Empirical Results: The experiments are thorough and convincing. On two benchmarks (RouterBench and MixInstruct), the authors show that their approach consistently achieves lower cumulative regret and faster convergence than baselines, meaning it learns the optimal routing policy more efficiently. The methods are compared across multiple embedding models (open-source vs OpenAI) and across different weighting schemes, demonstrating the approach’s robustness.

- Practical Efficiency: The method is sample-efficient and adaptable. It requires only a small amount of offline data for fine-tuning – e.g., using merely 5 example queries per category to calibrate the embeddings, which is far less than what some prior approaches needed. This low data requirement makes the approach practical to deploy.

### Weaknesses
- Lack of Real-World User Validation: A limitation is that the feedback mechanism is only evaluated in simulation or with automated oracles, rather than with human users. For RouterBench, the “preferences” are presumably derived from known performance metrics, and for MixInstruct the pairwise labels are generated by prompting an LLM as an oracle. While this is a reasonable proxy, it remains uncertain how the routing algorithm would perform with actual user feedback, which can be noisy or inconsistent. In practice, obtaining pairwise comparisons may require showing multiple model outputs to users or running interleaved A/B tests, which the paper does not explore. Demonstrating the approach in a real user study or on an online platform would strengthen the claims of practical utility.

- Complexity and Weighting Scheme Efficacy: The proposed solution introduces several moving parts including fine-tuning embeddings, computing category-weighted representations with various “Perf cost” or “Excel” weighting schemes, and then running a specialized bandit algorithm. This complexity could hinder real-world deployment and is not thoroughly ablated for necessity. In particular, the categorical weighting variants sometimes show mixed results.

- Baseline Comparisons: While the paper compares to a strong baseline (using a general-purpose OpenAI embedding in the same FGTS bandit framework) and outperforms it, the scope of baselines is somewhat limited. There is no direct comparison to a conventional supervised routing approach or to other online bandit algorithms in the literature (e.g., UCB-based methods or the MixLLM approach). In fact, the authors only discuss MixLLM qualitatively in the appendix, noting differences in feedback type and data requirements, but do not provide a head-to-head empirical comparison (likely because the problem settings differ). Including or adapting a few alternative routing strategies (such as a static classifier trained on a portion of data, or a simpler bandit without the special embedding) would bolster the empirical evaluation. 

- Focus on Pairwise Feedback Only: The solution is tailored to preference feedback, which is a reasonable design given the motivation, but it does not incorporate pointwise feedback (individual ratings or correctness signals) or hybrid approaches. In real applications, one might have access to occasional explicit ratings or other supervision in addition to pairwise comparisons. The authors acknowledge in discussion that FGTS.CDB is designed for pairwise data and conjecture it could be extended to pointwise signals, but they do not implement or test such extensions. This is a missed opportunity to demonstrate flexibility.

### Questions
- Collecting Pairwise Feedback in Practice: How do you envision obtaining the pairwise preference feedback from real users in an online setting? The paper argues that binary comparisons are more user-friendly than absolute ratings, but implementing this in a one-shot routing scenario is non-trivial as users typically only see one model’s answer. Would the system occasionally present two answers for the same query and ask the user to choose (akin to an A/B test)? If so, how would you manage the added latency and cost? 

- “Excel” Weighting Strategy Insight: The results indicate that the more complex weighting schemes (e.g., Excel perf cost) did not universally outperform simpler ones, and in some cases weighting fewer top categories yielded better outcomes. Could you elaborate on why that might be? For instance, does the “Excel” thresholding effectively filter out noise, or could it be discarding useful signal from less prominent categories? What does this imply about the quality of the metadata (performance/cost) being used? 

- Integration of Pointwise Feedback: Do you have plans to extend or adapt your approach to also utilize pointwise feedback signals? Since you conjecture that FGTS.CDB could be modified for pointwise rewards, it would be interesting to hear how this might be achieved.

### Soundness
4

### Presentation
3

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
This paper studies LLM routing problem and formulate this problem as contextual dueling bandit learning from pairwise preference feedback. To leverage dueling bandit algorithm, this work also introduces category-calibrated fine-tuning to derive model embedding. Empirical results are presented with four different types of the category weighting with the state-of-the-art feel-good Thompson sampling for contextual dueling bandit algorithm. Across benchmarks, the proposed methods can achieve lower cumulative regret over generic embedding models.

### Strengths
1. This paper studies an interesting and practical problem: how to route user queries to the best LLM. I found this problem has great potential and influence in the future. Also it is interesting to formulate this problem based on dueling bandit to leverage sequential learning and pairwise preference.
2. It is new for me to use dueling bandit for this problem setting.
3. The empirical results are complete and prove that the proposed methods can beat baselines.

I am not familiar with this line of LLM routing works so I may refer to other reviewers' opinions and the rebuttal to adjust my rating accordingly.

### Weaknesses
1. I feel the presentation can still be improved, especially Section 4.2. It is still not very clear to me how did your fine-tuning work with the model embedding $a_k$ you deduced there. It seems that you have already deduced the model embedding formulation, and then how did you fine-tune the small embedding models with contrastive learning? The training datasets are not very clear to me. The last paragraph of page 5 (line 261-266) is also not very clear to readers, and it is a bit hard to understand.

2. After reading some related works before reviewing this work, it seems that there is a line of methods considering model/LLM embedding along with the query embedding for the LLM routing problem ([R1, R2]). And I feel the methodology is kinda similar with this work: cluster the user queries and take the (weighted) average performance for each LLM. It is better to highlight the differences here.

3. It might be better to include some modern baselines (e.g. Universal routing, CARROT) to showcase the superiority of the proposed methods. Although most of them consider offline training framework instead of online learning, you can still compare your method with those baselines in a testing datasets after convergence. 

R1: Universal Model Routing for Efficient LLM Inference, Jitkrittum et al. 
R2: One Head, Many Models: Cross-Attention Routing for Cost-Aware LLM Selection, Pulishetty et al.

### Questions
Why did you choose to use Thompson sampling for the bandit algorithm part? I think UCB is also a popular and state-of-the-art contextual bandit framework.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CCFT for encoding LLM expertise and applies FGTS.CDB, based on CCFT-derived LLM embeddings, for online LLM routing under pairwise preference feedback.

### Strengths
1.	This work first studied LLM routing under online pairwise preference feedback, which meets the needs of practical applications.
2.	The proposed method has been empirically proven to improve cumulative regret.
3.	The writing is basically clear.

### Weaknesses
1.	Since prior works (Feng et al. 2025, Zhuang et al. 2025) have proposed techniques to strengthen users' and models' semantic information, it would be better to discuss why these previous approaches cannot be directly applied or what advantages the proposed CCFT offers.
2.	The experiments can be further strengthened. Apart from cumulative regret, the paper should include more intuitive evaluation metrics (e.g, accuracy on different tasks) to demonstrate the effectiveness of the proposed method. The work can consider two more baselines for ablation study, namely, OpenAItext with categorical weighting, e5b fine-tuning but without categorical weighting to construct model embedding. The standard derivation is better to report for understanding the model performance stability.


Minor   
1.	Not clear how to compute regret. Line 795-798 is hard to follow.  
2.	It is hard to tell which one is better in Figure 5. It may provide 2D t-SNE.  
3.	In A.2, $m$ should be $M$.  
4.	Line 256 should be three variants.

### Questions
1.	How is preference feedback collected in the experiments? Lines 353 and 433 mention it, but the process is still unclear.
2.	Confusion about the usage of pairwise comparison labels mentioned in Line 446-447.
3.	Confusion about how to calculate model embedding for the MixInstruct experiment when there is no explicit category label. From my understanding, Eq. (6) is still based on the label information as mentioned in Line 263-264.
4.	What distribution is considered for $p_0$ in Algorithm 1?
5.	How many queries pairs required for fine-tuning text embedding models?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose FGTS.CDB to address the LLM routing task. They leverage the Contextual Dueling Bandit paradigm, employing methods from online learning, reinforcement learning and weak supervision to tackle the high costs associated with both data labeling and model inference. Through experiments on the RouterBench and MixInstruct datasets, the proposed model demonstrates lower cumulative regret, faster convergence, and robust generalization, outperforming OpenAI Text embedding model.

### Strengths
1. The method utilizes online learning to dynamically update the agent, enabling an adaptive balance in selecting among multiple models, which is a significant advantage over traditional static approaches.   
2. By relying on comparative judgments (i.e., pairwise preference feedback), the framework reduces the high cost of annotation and mitigates the problem of inconsistent evaluation criteria that often plagues absolute scoring.   
3. Through the use of context-based categorization and weighting (the CCFT method), the approach effectively refines the feature space, leading to more targeted and efficient model selections.

### Weaknesses
1. The online learning phase introduces significant cost and latency overhead, which seems to contradict the primary motivation of LLM routing—cost reduction. The dueling method mandates two LLM calls for each learning step, whereas a well-tuned cascading system might succeed with just a single call in most cases.   
2. The assumption that pairwise preference feedback can be synthetically generated via a BTL model is overly idealistic. Real-world human feedback is notoriously stochastic, non-stationary, and can exhibit non-transitivity (e.g., a user might prefer A>B and B>C, but also C>A), which violates the BTL model's assumptions. Consequently, the smooth convergence of the regret curves may be an artifact of this idealized simulation.   
3. The CCFT method primarily relies on a predefined set of categories. When dealing with cross-domain or novel query types not seen during the offline phase, this rigid structure could lead to suboptimal routing decisions, as the a priori categorization may not generalize well and could mask essential information.   
4. The red line in Fig. 1, which is meant to represent a successful learning curve, is not identified in the legend, hindering clarity.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
