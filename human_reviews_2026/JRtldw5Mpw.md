# Sparse Models, Sparse Safety: Unsafe Routes in Mixture-of-Experts LLMs

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
By introducing routers to selectively activate experts in Transformer layers, the mixture-of-experts (MoE) architecture significantly reduces computational costs in large language models (LLMs) while maintaining competitive performance, especially for models with massive parameters. 
However, prior work has largely focused on utility and efficiency, overlooking the safety risks associated with this sparse architecture.
In this work, we show that the safety of MoE LLMs is as sparse as their architecture by discovering $\text{\emph{unsafe routes}}$: routing configurations that, once activated, convert safe outputs into harmful ones.
Specifically, we first introduce the $\underline{\text{Ro}}$uter $\underline{\text{Sa}}$fety $\underline{\text{i}}$mportance $\underline{\text{s}}$core ($\textbf{RoSais}$) to quantify the safety criticality of each layer's router. 
Manipulation of only the high-RoSais router(s) can flip the default route into an unsafe one.
For instance, on JailbreakBench, masking 5 routers in DeepSeek-V2-Lite increases attack success rate (ASR) by over 4$\times$ to 0.79, highlighting an inherent risk that router manipulation may naturally occur in MoE LLMs.
We further propose a $\underline{\text{F}}$ine-grained token-layer-wise $\underline{\text{S}}$tochastic $\underline{\text{O}}$ptimization framework to discover more concrete $\underline{\text{U}}$nsafe $\underline{\text{R}}$outes ($\textbf{F-SOUR}$), which explicitly considers the sequentiality and dynamics of input tokens.
Across four representative MoE LLM families, F-SOUR achieves an average ASR of $\sim$0.90.
Finally, we outline defensive perspectives, including safety-aware route disabling and router training, as promising directions to safeguard MoE LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the safety of MoE LLMs and discovers unsafe routes within the sparse architecture. The authors first propose RoSais to quantify the safety importance of layers across layers. This is achieved by randomly masking routing decision and observing the model output changes. Then, the authors find that manipulating the safety-critical routers, the ones with higher RoSais, cause the models to generate unsafe content. Furthermore, the authors propose a more effective attack framework F-SOUR to achieve an average ASR of 0.9, considering the interaction between shallow and deep routers after intervention. Defensive perspectives are also discussed.

### Strengths
1. **Good research question**: The paper addresses an important and underexplored problem — the safety of routing mechanisms in MoE large language models. While prior work has primarily focused on improving utility and efficiency, this study fills a crucial gap by systematically examining router safety.
2. **Intuitive approach to find safety-critical routers**: The intuition behind identifying safety-critical routers is clear and insightful: the importance can be revealed by observing how the model’s output changes when its routing decisions are manipulated. 
3. **Extensive experiments**: The authors evaluate four models, compare against four baseline approaches, and propose two methods (RoSais-based unsafe route discovery and F-SOUR), complemented by a case study on the defense mechanism. This strengthens the validity of the findings.

### Weaknesses
1. The main concern lies in the trivial nature of the central finding regarding safety-critical routers. Since routers are designed to assign expert FFNs based on the characteristics of the input task, it is expected that risky or harmful inputs would naturally be routed to specific experts. Thus, identifying these “safety-critical routers” may reflect an inherent property of MoE routing rather than a novel safety insight.
2. RoSais is based on the model's next-token distribution given a harmful question (line 197). However, prior work [1] suggests that next-token distributions are insufficient as reliable indicators of model safety. This raises concerns about whether RoSais accurately captures safety-relevant behavior and how this limitation may affect its reliability.
3. The approach to find and leverage safety-critical routers depends on a random sampling mechanism (line 186, 248). However, the paper does not provide details about the computational cost of this procedure, which could limit its practicality for large-scale MoE models.

[1] Safety alignment should be made more than just a few tokens deep

### Questions
1. Do the identified safety-critical routers generalize across different harmful questions or categories of unsafe content? For example, if a router is found critical for one harmful prompt (Q1), does it remain critical for another (Q2)?
2. Given that prior studies [1] suggest next-token distributions may not fully reflect model safety, how do the authors justify using this signal as the basis for RoSais? Would incorporating multi-token contexts improve robustness?
3. What is the computational overhead of the random sampling approach (line 186, 248) used to identify and leverage safety-critical routers? Could this process scale to larger MoE models?

[1] Safety alignment should be made more than just a few tokens deep

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper reveals that the safety of Mixture-of-Experts (MoE) large language models is as sparse as their architecture; manipulating only a few critical routers can convert safe outputs into harmful ones. The paper introduces the Router Safety Importance Score (RoSais) and a fine-grained optimization method (F-SOUR) to identify unsafe routing configurations, achieving high attack success rates and highlighting the need for safety-aware router defenses.

### Strengths
1. The proposed framework enables the discovery of unsafe routing paths with token-level precision, highlighting a sophisticated methodological design.

2. Through controlled experiments across multiple MoE LLM families, the paper convincingly demonstrates that manipulating only a few routers can dramatically increase attack success rates, providing strong empirical validation of the threat.

### Weaknesses
1. Incremental contribution. The contribution appears limited and incremental, given concurrent efforts exploring similar safety issues in MoE LLMs. Previous works [1,2] have already shown that altering routers can induce harmful outputs. This paper primarily introduces a router scoring mechanism similar to a filtering approach, which may not represent a substantial conceptual advance.

2. Lack of comparison with recent baselines. The paper omits comparisons with concurrent works [1,2], both of which analyze router-related vulnerabilities and jailbreak effects in MoE LLMs. Without such baselines, it is difficult to assess the novelty and relative effectiveness of the proposed method.

3. Unclear effectiveness of the RoSais score. The improvement reported in Table 1 may stem from random masking rather than genuine routing manipulation. The paper lacks a solid ablation or sensitivity analysis to isolate the effect of RoSais.

[1] Mohsen Fayyaz, Steering MoE LLMs via Expert (De)Activation.
[2] Zhenglin Lai, SAFEx: Analyzing Vulnerabilities of MoE-based LLMs via Stable Safety-critical Expert Identification.

### Questions
1. When a specific router is modified, subsequent routers may also change their activation patterns. How does the paper ensure that the proposed metric accurately quantifies safety criticality without being confounded by downstream routing effects?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates inherent sparse safety in sparse Mixture-of-Experts (MoE) architecture for large language models (LLMs). It introduces the Router Safety importance score (RoSais) to identify safety-critical routers within the model and demonstrates that manipulating a small number of these sparsely distributed routers can drastically increase the rate of unsafe outputs. The authors further propose F-SOUR, a fine-grained, token- and layer-wise optimization framework to discover concrete unsafe routing configurations. Experiments across four recent MoE LLMs show that simple or well-optimized manipulations of expert routing can yield attack success rates near 90% or higher, even in safety-aligned LLMs, revealing significant novel attack surfaces in MoE architectures. Possible defensive strategies and an extensive analysis are provided. 

However, both RoSais and F-SOUR require full access to internal routing scores, but the persistent gap between academic white-box settings and real-world MoE deployments (often closed) is not deeply interrogated. This work lacks evaluation on the impact of routing manipulations (attack or defense) on the model’s general utility.

Overall, this work provides an interesting insight into the safety sparsity of MoE from the perspective of routers. But the attacks lack realistic applicability, and the tradeoff between general utility is unknown.

### Strengths
1. This paper highlights the inherent sparse safety in the sparse Mixture-of-Experts (MoE) architecture for large language models (LLMs). 
2. It introduces the Router Safety importance score (RoSais) to identify safety-critical routers within the model and demonstrates that manipulating a small number of these sparsely distributed routers can drastically increase the rate of unsafe outputs. 
3. It also proposes F-SOUR, a fine-grained, token- and layer-wise optimization framework to discover concrete unsafe routing configurations. Experiments across four recent MoE LLMs show that simple or well-optimized manipulations of expert routing can yield attack success rates near 90%, even in safety-aligned LLMs. 
4. This work lists possible defensive strategies and an extensive analysis.

### Weaknesses
1. A key limitation of this work is the lack of evaluation on the impact of routing manipulations on the model’s general utility. While the paper demonstrates significant increases in attack success rate (ASR) under RoSais-guided or F-SOUR-based routing interventions, it does not assess whether these modifications degrade performance on benign inputs or standard NLP tasks. Similarly, the defense strategy in Appendix D disables safety-critical experts without reporting any utility-preserving analysis, leaving open the question of whether safeguarding against unsafe routes comes at the cost of reduced model capability. 
2. The proposed attack methods assume white-box access to internal routing scores, limiting their applicability to real-world black-box API settings.

### Questions
1. When disabling high-RoSais routers or experts for defense, what is the impact on utility benchmarks (e.g., standard language or task performance)? Could the authors provide data or an analysis?
2. The dataset-level RoSais are averaged, but are unsafe routes transferable between very different datasets, models, or domain-adapted variants?

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
4

### Summary
The paper analyzes safety risks associated with the Mixture-of-Experts (MoE) transformer architecture. Specifically, the authors look into the possibility of changing the routing configurations in order to specifically activate experts from the MoE model that are more likely to produce harmful responses to malicious questions. They show that they can successfully identify these vulnerable routings by computing the router's safety score (RoSais). Changing the route to an unsafe one either by naively masking safety-critical routes or by using a stronger optimization framework (F-SOUR) manages to make the model output harmful content in most cases (ASR = 0.79 and 0.90 respectively).

### Strengths
The paper is well structured and easy to read. The theoretical background and the methods are well-explained, with enough details so that a reader without a solid background in MoE architectures or LLM safety can still understand the points made in the paper.

The discovery that deliberately changing the routing configurations of MoE models can result in harmful behaviour is interesting. The formulation of sparse safety is insightful and original.

The experiments are well organized and the results are presented and interpreted in insightful ways.

### Weaknesses
**Unclear Threat Model and Limited Practical Motivation** 

The paper identifies a structural vulnerability in MoE architectures, but the real-world applicability of the attack scenario is underexplored. The threat model assumes that an adversary can manipulate routing configurations during inference. However, in practice, such access is typically restricted to the model owner or deployer. If an attacker can modify routing, they likely have control over other model components (weights, safety filters, etc.), making routing manipulation less of a uniquely exploitable vector.

**Overcomplexity of F-SOUR Relative to Its Gains**  

While F-SOUR is methodologically interesting, it only produces marginal ASR gains over the simpler RoSais-based attack. The algorithm’s stochastic token-layer optimization adds significant complexity and computational cost without clear evidence that it uncovers qualitatively different unsafe routes than simpler manipulations.

**Unjustified Emphasis on Edge/IoT Scenarios** 

The authors claim that the safety concerns are especially elevated for edge/IoT devices, without any further justifications for this claim. In fact, these devices are typically black-box environments where adversaries cannot directly modify routing decisions. The paper’s argument would be more convincing if it included a concrete example of how routing decisions could be altered in these situations.

**Defense Section Underdeveloped** 

While the paper presents a strong and novel attack-side analysis of MoE vulnerabilities, the discussion of defenses feels somewhat incomplete and conceptually shallow compared to the technical rigor of the earlier sections. The proposed countermeasures, route disabling and safety-aware router training, are briefly outlined but lack quantitative evaluation or theoretical grounding. It remains unclear how practical these approaches are under realistic constraints: for instance, how much utility or efficiency would be lost by permanently disabling high-RoSais experts. I know that expanding the defense section would require significant experimental overhead and therefore I don’t expect these experiments to be completed during the rebuttal, but further results in this direction would strengthen the paper.

**Minor Experimental Irregularities (TAP/PAIR Baselines)** 

In some tables, the ASR of known jailbreak attacks (PAIR, TAP) is lower than the original baseline, which suggests potential misconfiguration of attack parameters or mismatched evaluation settings. While this does not undermine the core claims, it introduces small doubts about the exact comparative strength of F-SOUR.

### Questions
1. Could the authors expand on the attacker threat model? I am interested in both the general case and the edge/IoT case.

2. I would like to see some comparisons between base models and established defences (e.g. adversarial finetuning, llama-guard, activation steering). It would be interesting to see if routing attacks can bypass these defences, if adversarial finetuning can balance the RoSais scores between routers or if already safe experts are trained to become even safer. I know that these experiments would require significant time and resources and I don’t expect complete results during the rebuttal, but I would like to see at least some insights in this direction.

### Soundness
3

### Presentation
3

### Contribution
2
