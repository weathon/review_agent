# Dirichlet-Prior Shaping: Guiding Expert Specialization in Upcycled MoEs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Upcycling pre-trained dense models into sparse Mixture-of-Experts (MoEs) efficiently increases model capacity but often suffers from poor expert specialization due to naive weight replication. Our analysis reveals that upcycled MoEs, even with conventional regularization, exhibit low-confidence, weakly differentiated routing, hindering performance. We introduce Dirichlet-Prior Shaping Loss (DPSL), a novel router regularization technique that directly shapes routing probability distributions by matching expert assignments to a target Dirichlet prior. DPSL offers fine-grained control over expert balance and specialization, and enables encoding of inductive biases such as encouraging experts to focus on specific modalities or tasks, without requiring manual intervention; notably, DPSL is a general tool applicable to any module that outputs categorical probability distributions, extending its utility beyond MoE training. Experiments on upcycled MoE vision-language models (with Qwen2, Phi3, Llama3.2 LLM backbones) show DPSL consistently outperforms upcycling strategies and regularization techniques across standard vision-language benchmarks, addressing the critical issue of poor specialization and fostering more adaptive, higher-performing models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a new regularization loss for Upcycled MoE training to address the issue of uniform routing weights. Specifically, the DPSL regularizes the distance between the empirical CDF and a target Beta CDF for each expert category.

### Strengths
1. The motivation is clear, and I believe that the distribution of routing probabilities is a critical issue in upcycled MoE models.  
2. The proposed method aligns well with this motivation, and performance improvements are observed in several scenarios.

### Weaknesses
1. It is unclear why shaping the routing probability distribution to follow a Dirichlet distribution is the optimal choice. Could the authors provide stronger theoretical or empirical justification for this design decision? Additionally, the reported performance gains appear marginal in most cases, which raises questions about the practical significance of the proposed approach.

2. According to Table 8, DPSL achieves better load balancing than the standard load-balancing loss. Could the authors clarify the underlying reason for this improvement? Furthermore, would this advantage persist in Modality-Specific and Task-Specific settings? For a more comprehensive evaluation, it would also be helpful to report training and inference times.

3. While the hyperparameter $\alpha$ has a substantial impact on the logit distribution, Tables 4 and 5 suggest that identifying an optimal setting is non-trivial. This may limit the practical applicability of the method in real-world scenarios where extensive tuning is often infeasible.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of poor expert specialization in Mixture-of-Experts (MoE) models created by upcycling pre-trained dense models. The authors argue that naive weight replication leads to homogeneous experts and low-confidence routing, which conventional regularization methods like load-balancing or z-loss fail to adequately resolve. To this end, they introduce Dirichlet-Prior Shaping Loss (DPSL), a novel regularization technique that directly shapes the router's output probability distribution. DPSL matches the empirical cumulative distribution function (CDF) of expert assignment probabilities to a target Beta distribution, which is the marginal of a chosen Dirichlet prior. This approach allows for fine-grained control over expert utilization and specialization. The authors demonstrate that symmetric priors can enforce confident and balanced routing, while asymmetric priors can be used to instill inductive biases, such as encouraging experts to specialize in specific modalities (e.g., vision vs. language) in Vision-Language Models (VLMs). Through extensive experiments on upcycled VLM MoEs with various backbones (Qwen2, Phi3, Llama3.2), the paper shows that DPSL consistently outperforms existing upcycling and regularization techniques on a suite of standard vision-language benchmarks.

### Strengths
- DPSL is a clean approach that provides fine-grained control over router behavior. It moves beyond simple heuristics like load balancing and allows practitioners to instill complex and desirable statistical properties into the routing mechanism.
- The authors have conducted an extensive set of experiments across three different modern LLM backbones, two MoE configurations, and six benchmarks.
- The analysis of router output distributions (e.g., Figure 3 and Appendix C.5) provides valuable qualitative insight into why DPSL works better than alternatives, by encouraging more confident and diverse routing assignments.

### Weaknesses
- The paper focuses exclusively on upcycled VLMs. While the authors claim DPSL is a general tool, they provide no evidence of its efficacy for training MoEs from scratch or in other domains like language-only models. Demonstrating its utility beyond the upcycling VLM setting would significantly strengthen the claims of generality.

- The paper reports a 10-15% computational overhead during training, which is not negligible. The mitigation strategy of only applying it during a warm-up phase is reasonable, but a more detailed analysis of the impact on total training time would be welcome. Furthermore, DPSL introduces new hyperparameters (α and λ), and the ablation study shows that different model backbones prefer different settings (e.g., α=0.75 for Llama3.2 vs. α=1.0 for Qwen2). This suggests that some degree of architecture-specific tuning may be required, which could be a practical drawback.

### Questions
- The lack of multiple seeds is a concern. Could you provide results for at least one model configuration (e.g., Llama3.2-1B 2in4) across 3-4 seeds for your method and the top-performing baselines (e.g., DeepSeek balancing, Drop-Upcycling)?
- The results in Table 2 show that explicitly guiding specialization toward pre-defined task subsets (both with BTX and DPSL) underperforms the general-purpose symmetric DPSL. The paper speculates this is due to "nonoptimal data subsets." An alternative hypothesis is that enforcing hard, fine-grained specialization is simply the wrong inductive bias for these multi-faceted VLM tasks. Could you comment on this alternative interpretation? Does this result suggest a potential limitation of highly specialized expert priors?
- Could you discuss the expected behavior and potential challenges of applying DPSL to training MoEs from scratch? In that setting, the router and experts co-evolve from a random initialization. Would you expect DPSL to be more or less effective compared to its application in the upcycling setting where the router's main job is to break symmetry?

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
3

### Summary
The article proposes a novel router regularization technique, Dirichlet-Prior Shaping Loss (DPSL), to address the issue of insufficient specialization among experts in sparse mixture-of-experts (MoE) models. The paper analyzes the routing problems of low confidence and weak discrimination in conventional upcycled MoEs. To improve this, the authors propose matching the routing probability distribution with the target Dirichlet prior, flexibly controlling the balance and specialization level of experts.

### Strengths
- The method allows flexible control over the trade-off between expert balance and specialization by aligning router outputs with a tunable Dirichlet prior.

- Experimental results demonstrate its strong performance.

- The presentation is clear and easy to understand.

### Weaknesses
- The DPSL method shows good performance on various benchmarks but relies on selecting the appropriate Dirichlet prior (e.g., α=0.75 being more effective for Llama3.2-1B). This dependence may increase the hyperparameter tuning effort during training. It is recommended to further analyze the parameter selection process to help readers understand the rationale behind the choices.

- The authors mention in the abstract that DPSL is also a general-purpose tool. It is recommended to further elaborate on the potential applications of DPSL in other areas in the main text to demonstrate its broad applicability.

### Questions
- Consistency of terminology, it is recommended to unify the use of either "asymmetric priors" or "non-symmetric priors" to ensure consistency and accuracy in terminology.

- In Table 1, there is an inaccuracy in the labeling of the second-best data; please correct it.

### Soundness
3

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
4

### Summary
This paper addresses the challenge of poor expert specialization with upcycled MoE models, where pre-trained dense models are converted to (sparse) MoEs. The authors propose the use of Dirichlet priors in order to regularize router probability distributions, matching expert assignments to target Dirichlet priors. The method shows good results in providing fine-grained control over expert balance/specialization, with experiments on popular vision language models that demonstrate improvements over baseline approaches.

### Strengths
- the challenge adressed is well defined, and explained intuitively through visualizations

- the use of Dirichlet priors to shape routing distributions is well motivated 

- the framework can be more general than the presented application of MoE routing (e.g. in cases where modules output categorical probability distributions). Symmetric and asymmetric priors are both integrated in the framework

- evaluation spans multiple backbones and MoE configurations, indicating broad applicability

- ablation studies support the strategies employed, and provide insights into the method's behavior

### Weaknesses
- While the application to MoE routing is novel, the methodological innovation is incremental. The paper could benefit from discussing fundamental differences from related work.

- while improvements are consistent, absolute improvements are often small (e.g., even less than 1% in some cases), which makes the improvement questionable at times given the additional computational overhead

- although the framework can be more generally applicable, this is not evidenced in the paper

- The paper could benefit from more in depth analysis; e.g. specialization evidence (Sec 3.6), it is not clearly demonstrated that experts actually learn different functions, and there is no analysis of what each expert specializes in. 

-In Table 2, modality-specific experiments are shown that demonstrate task-specific specialization that may sometimes underperform symmetric priors which could be seen as contradicting the specialization narrative.

### Questions
please see above. Additionally,
- more evidence for expert specialization - are experts really learning different functions? (E.g. via activations)
- how does this approach compare to simpler approaches, e.g. entropy regularization
- When should asymmetric priors be used? Given task-specific priors underperform e.g. in Table 2
- How stable is the method with respect to parametrization? can some settings lead to collapse?
- Scaling/computational overhead when increasing number of experts?
- Make sure to describe well how Avg score in Table 1 is computed

### Soundness
3

### Presentation
3

### Contribution
3
