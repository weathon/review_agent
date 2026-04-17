# Iterative Multi-Objective Policy Optimization for Antibody Sequence Design

- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Antibodies are among the most important medicines in use today, yet their development is constrained by costly and labor-intensive affinity maturation. Computational antibody design offers a scalable alternative, but faces two central challenges: the lack of reliable affinity labels and the need to balance binding affinity with structural fidelity (self-consistency of refolded sequences to the original backbone). In this work, we formulate antibody sequence design as a multi-objective optimization problem and develop an iterative policy optimization framework tailored to this setting. To approximate experimental binding affinity, we construct a surrogate reward by regressing wet-lab $\Delta \Delta G$ measurements against Rosetta-derived interface metrics, including shape complementarity, buried surface area, and interfacial hydrogen bonds. To preserve structural fidelity, we introduce self-consistency RMSD as a complementary objective. Our method performs iterative training with a regression loss derived from the KL-regularized policy optimization objective, enabling stable on-policy learning under expensive structural evaluations and progressively guiding the policy toward Pareto-efficient trade-offs between binding affinity and structural fidelity. Across diverse antigen targets, this approach yields antibody sequences that achieve improved binding affinity while maintaining structural consistency, advancing computational antibody design toward practical therapeutic application.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce an iterative policy optimisation for structure-conditioned antibody sequence design. They reformulate the design as a multi-objective problem with the goal of identifying solutions that are Pareto optimal with respect to binding affinity and structural fidelity. They explore fixed-weight combination and adaptive-weight conditioning between the two trade-offs. For binding affinity they propose a surrogate reward by regressing DDG measurements against Rosetta-derived metrics (using a linear regression). For structural fidelity they use self-consistency RMSD, i.e self-consistency between initial backbone and refolded structure. To address the computational expensive steps of structure refolding and Rosetta scoring as well as scarce labels, they developed an iterative policy optimisation framework (which they also theoretically derive) resulting in a 'resample-re-estimate-update' loop that ensures successive policies are close to each other. Results are experimentally evaluated across four representative antigen targets and compared with a variety of baselines across various metrics, which shows that their approach is superior across all tasks. Next to that they evaluate the Pareto front built by the two alternative strategies.

### Strengths
- clarity of writing
- original solution of iterative policy optimisation (and mathematical derivations thereof)
- approximation of binding strength by regressing DDG measurements against Rosetta-derived metrics 
- comprehensive comparison to various baseline methods (including sequence only & structure-conditioned methods)
- extensive appendix with additional proofs and results

### Weaknesses
- multi-objective guidance (and Pareto front analogy) for generative antibody design is not completely novel 
- distribution results (Fig 2,3) are less convincing (across all metrics) and slightly over-interpreted
- Pareto front results (Fig 4) could have been interpreted more in detail (other metrics, visualisations)

### Questions
Please address the points mentioned as weak points; first two textual, third one with some additional figures/tables.

### Soundness
3

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
The paper tackles in silico antibody affinity maturation as a multi-objective RL problem. 
This work (AbMPO) poses antibody sequence design as a multi-objective RL task, optimizing for binding affinity (via a learned surrogate of experimental $\Delta \Delta G$) and structural fidelity (scaffold RMSD).
The authors fine-tune a structure-conditioned sequence policy through iterative, KL-regularized advantage matching (regression) updates rather than direct PPO.
In simulation studies across diverse antigens, the method produces sequences with improved predicted binding while preserving structural consistency.

### Strengths
1. Multi-objective design: Weight-conditioned policy yields continuous Pareto-optimal trade-offs

2. Strong improvements across multiple antigens and evaluation metrics. Includes SOTA baselines (DiffAb, AbDPO, Ab-Gen) for fair comparison

3. Integrates experimental data and structural biology knowledge into the reward.

4. good theoretical supports. I skimmed through proofs. didn't check line-by-line, but looks correct. The results are neither vacuous nor irrelevant to the core method. The authors clearly distinguished prior results from their own contributions (as in props.) This level of detail strongly reinforces their credibility. Therefore, even without code, I tend to trust their numerical results, which appear as solid and reliable as their theoretical results.

### Weaknesses
This paper is quite technical. I tend to accept this paper. Below are points for improvements. Willing to change my mind during rebuttal, if below weaknesses get addressed adequately. Feel free to correct me if I am wrong (please in the most precise, concise and reasonable way.)

0. [minor] Relies on a strong pretrained $\pi_{\text{ref}}$. From‑scratch claims remain unclear.

1. [major] Novelty: Method is largely a combination of known techniques (KL-regularized policy updates + multi-objective conditioning). The contribution is incremental rather than fundamentally new.

2. [major] Reproducibility: Results lack statistical analysis (no multiple runs or error bars in most tables). So the significance of improvements is not quantified. No code provided but the descriptions are very detailed.

3. [minor] Computational cost: The pipeline relies on expensive structure evaluations (Rosetta and refolding for thousands of sequences). They emphasize “large‑batch offline sampling” and per‑sequence Rosetta evaluation. The method section shows offline soft‑value estimation and iterative on‑policy refinement.
 Please at least document the computational cost (e.g. hours per round, hardware used) to set expectations for practical use.

4. [minor] Scope & Scalability & Generality: 
    - Clarify the method’s applicability and limits. 
    - The approach is trained per-antigen. It does not demonstrate generalization to new targets without retraining.
    - High compute is implied as stated above. 

5. [major] Validation:  
    - Statistical Validity: Provide some measure of result variability or significance. For example, run the policy optimization multiple times (or use bootstrapping) to show that improvements over baselines are consistently significant and not due to lucky initialization or sampling.
    - [very minor yet I still flag it here] Evaluation is entirely in silico (Rosetta metrics and learned predictors). No experimental or in vitro validation is presented to confirm real-world impact.

6. Experiments:
    - No ablation comparing the surrogate to raw Rosetta energy.
    - scRMSD is sometimes worse than AbDPO. need failure modes analysis

### Questions
see above

LLM disclaimer: I used LLM to polish language and to understand some cited refs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present AbMPO for antibody sequence design, and propose an iterative policy optimization algorithm, a surrogate affinity reward, and an adaptive weight-conditioning for multiple objectives. AbMPO outperforms existing baselines across multiple antigens.

### Strengths
- The problem is well-formulated, and the proposed method clearly addresses the challenges that the authors present for antibody design. The paper is clearly-written and easy to follow.
- The proposed method is intuitive and aptly demonstrates how ideas in policy optimization can be combined to improve performance for antibody design. AbMPO improves performance compared to a diverse set of baselines, which include both sequence and structure-based models. The benchmarks also include many antigen targets, so their conclusions appear to be robust.
- The proposed surrogate model seems to be very effective, and the ablations included in the paper demonstrate that combining the three metrics work better than directly optimizing for each metric individually. This could have interesting implications about how binding affinity should be optimized.

### Weaknesses
The paper has limited novelty and does not propose substantial machine learning ideas. Here's my understanding of the main contributions, and why I don't believe they constitute meaningful innovations. 
  - Iterative policy algorithm: This seems very similar to [1] from 2019, but this paper is not discussed. Could you elaborate on the differences between your proposed method and this paper? From my understanding, it seems like they are both iterative methods which upweight based on the observed advantage.  
[1] Peng et al, Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning
  - Surrogate reward model: Although this surrogate is well-motivated, the model itself is an extremely simple linear regression of three metrics. 
  - Multi-objective optimization: The RBF kernel expansion is interesting, and this seems like a clever way to have a single model represent the multi-objective problem. However, there are very few ablations which compare the fixed-weight scalarization with adaptive weight-conditioning, or other MO methods proposed in prior works.

### Questions
- Could you elaborate on the advantages of adaptive weight-conditioning for MOO? How does your approach of adaptive weight-conditioning compare to other MOO methods?

### Soundness
3

### Presentation
3

### Contribution
2
