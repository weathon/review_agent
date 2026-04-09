# MOST RELEVANT HUMAN REVIEWS FOR "THE ART OF SCALING RL COMPUTE FOR LLMS"

## EXECUTIVE SUMMARY

Found **4 Tier-1 papers** (highest relevance) and **4 Tier-2 papers** (supporting evidence) with direct applicability to SCALERL's large-scale RL empirical study. These reviews identify critical weakness patterns that recur across large-scale machine learning research.

---

## TIER 1 - DIRECTLY APPLICABLE (Highest Relevance)

### 1. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/m29SV0n6DO.md`
**Paper:** Generative Pre-training on Videos with Large-Scale Empirical Study (1T tokens)
**Relevance:** Large-scale empirical training study with design choice ablations

#### Key Weakness Patterns:

1. **Massive scale doesn't guarantee insights**
   - *Quote:* "While there are a lot of experiments, the takeaway and why is not really answered"
   - *Issue:* Lack of mechanistic explanation for observed scaling behavior
   - *Applies to SCALERL:* Must explain WHY sigmoid curves fit better, not just that they do

2. **Design choices scrape the surface**
   - *Quote:* "The study only scratches the surface" (Reviewer 2 on tokenizer choices)
   - *Issue:* Design choices examined superficially without deep analysis
   - *Applies to SCALERL:* GRPO vs DAPO vs CISPO comparisons need deeper mechanistic analysis

3. **Scaling behavior can be trivial**
   - *Quote:* "Isn't the scaling behavior somewhat trivial? We know Llama already exhibits these scaling behaviors"
   - *Issue:* Just because a larger model with same architecture scales similarly isn't novel
   - *Applies to SCALERL:* Are sigmoid scaling laws for RL just expected behavior from LLMs?

4. **Missing ablations and justifications**
   - *Quote:* "For results in Table 6, why is Mamba so much worse than Toto? Is comparison to GPT2 fair?"
   - *Issue:* Results presented without explaining why discrepancies exist
   - *Applies to SCALERL:* Why do some design choices work better on 8B vs 17B models?

5. **Generalization scope overstated**
   - *Quote:* "Claiming generality might be overstated given limited evaluation" (Reviewer 1)
   - *Issue:* Limited to in-distribution tasks, no out-of-distribution evaluation
   - *Applies to SCALERL:* Trained on math tasks—do scaling laws generalize to other domains?

6. **Missing ablation of key claims**
   - *Quote:* "Where is the part showing relative positional embeddings are better? This is mentioned but not ablated"
   - *Issue:* Claims made without corresponding ablation studies
   - *Applies to SCALERL:* Are all claimed design choices actually ablated?

---

### 2. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/R1hIXdST22.md`
**Paper:** MR.Q: Model-Free Deep RL Algorithm with Model-Based Representations
**Relevance:** RL algorithm with empirical evaluation across benchmarks and ablations

#### Key Weakness Patterns:

1. **Hyperparameter fairness in baseline comparisons (CRITICAL)**
   - *Quote:* "The paper provides no detail regarding hyperparameter settings of baseline algorithms, nor how or if they were tuned. This is a critical weakness as the possibility of untuned baselines undermines the claimed performance improvements."
   - *Issue:* Can't trust performance improvements if baselines aren't equally tuned
   - *Applies to SCALERL:* Were GRPO/DAPO/CISPO baselines properly tuned with same compute budget?

2. **Statistical significance lacking in ablations**
   - *Quote:* "The ablation study fails to provide the statistical significance of the results and lacks analysis"
   - *Issue:* Small performance differences without significance testing are meaningless
   - *Applies to SCALERL:* Are sigmoid curve improvements statistically significant?

3. **Design choice justification incomplete**
   - *Quote:* "Ablations show minimal performance gains. Further analysis of why each component matters would strengthen contribution"
   - *Issue:* Need to explain not just that components help, but WHY
   - *Applies to SCALERL:* Why does off-policy async training help specifically for 8B/17B models?

4. **Limited evaluation scope**
   - *Quote:* "Authors claim generality but only tested on Gym, DMC, and Atari (classic but homogeneous environments)"
   - *Issue:* Limited domain diversity undermines generality claims
   - *Applies to SCALERL:* Only evaluated on math tasks—generalization to reasoning/coding unclear

5. **Missing key baselines**
   - *Quote:* "Some key algorithms missing, most notably PPO, which is the closest thing to a 'general-purpose' RL algorithm"
   - *Issue:* Can't claim superiority without comparing to standard baselines
   - *Applies to SCALERL:* How does SCALERL compare to standard RLHF baselines?

6. **Model size unfairly compared**
   - *Quote:* "Paper seems to maintain model sizes from original implementations. Would be more convincing to normalize parameter count and show MR.Q achieves superior performance at same size"
   - *Issue:* Different model sizes make performance comparisons meaningless
   - *Applies to SCALERL:* Fair comparison between 8B and 17B requires same tokenizer/vocab?

7. **Theory-practice disconnect**
   - *Quote:* "Theory is based on highly constrained linear assumptions and provides little justification for proposed method"
   - *Issue:* Empirical components not theoretically justified
   - *Applies to SCALERL:* Theory behind sigmoid fitting for RL is under-developed

---

### 3. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/P7f55HQtV8.md`
**Paper:** QuaDiM - Diffusion Model for Quantum State Property Estimation (Extrapolation Claims)
**Relevance:** Paper claiming extrapolation/prediction from limited training data

#### Key Weakness Patterns:

1. **Extrapolation validity not convincingly demonstrated**
   - *Quote:* "Does QuaDiM extrapolate to unseen system parameter x distribution, likely with disjoint support? E.g., trained on [0, 0.8], tested on [0.9, 1]?"
   - *Issue:* Extrapolation tested only on narrow scenarios similar to training
   - *Applies to SCALERL:* Do sigmoid curves extrapolate beyond training compute range? Tested on [1e5, 4e5] but claim holds for 1e6?

2. **Experimental design weak for extrapolation claims**
   - *Quote:* "Experiment design relatively weak. Would be interesting to consider more complicated property estimation problems"
   - *Issue:* Simple test cases don't validate extrapolation capability
   - *Applies to SCALERL:* Are math tasks representative enough to extrapolate scaling laws?

3. **Setup not clearly presented**
   - *Quote:* "Generation protocol of system parameters not described. Training/test splitting process unclear. Hard to understand under what level QuaDiM extrapolates"
   - *Issue:* Can't validate extrapolation claims without clear setup
   - *Applies to SCALERL:* How were curve fits generated? Cross-validation? Out-of-distribution test set?

4. **Benchmarks not representative**
   - *Quote:* "Benchmarks not representative enough. Related works not discussed or compared"
   - *Issue:* Limited baselines make strong claims questionable
   - *Applies to SCALERL:* How do sigmoid curves compare to power-law fits from prior work?

5. **Limited mechanistic understanding**
   - *Quote:* "More discussion needed on why diffusion models are necessarily better [than autoregressive]"
   - *Issue:* Lacks principled explanation for method choice
   - *Applies to SCALERL:* Why sigmoid curves fit better than exponential/power-law? What's the underlying mechanism?

---

### 4. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/cu2CT2VAvs.md`
**Paper:** State Collapse in RNN-based Language Models (Length Extrapolation)
**Relevance:** Large-scale training study with multiple design choice ablations

#### Key Weakness Patterns:

1. **Definitions and concepts vague/circular**
   - *Quote:* "Lack of formal, clear definition of State Collapse. Descriptions are vague, making it difficult to assess analysis"
   - *Issue:* Central concepts undefined rigorously
   - *Applies to SCALERL:* Is SCALERL recipe clearly defined or are design choices vague?

2. **Design choices only scratch surface**
   - *Quote:* "Authors claim to study tokenizers in detail, however the study only scratches the surface"
   - *Issue:* Design choices examined without sufficient depth
   - *Applies to SCALERL:* Are RL design choices (GRPO vs DAPO) deeply understood or superficially compared?

3. **Generalizability questionable**
   - *Quote:* "Methods evaluated on single model and dataset only. Limiting generalizability of findings"
   - *Issue:* Results may not transfer to other domains/architectures
   - *Applies to SCALERL:* Do scaling laws transfer to models other than 8B/17B? Different training data?

4. **Limited evaluation despite scale**
   - *Quote:* "Despite massive compute, evaluation limited to narrow set of tasks without testing transfer learning or out-of-distribution tasks"
   - *Issue:* Large experiments don't guarantee broad applicability
   - *Applies to SCALERL:* 400k GPU hours but only on math tasks—what about other domains?

5. **Ablation completeness questioned**
   - *Quote:* "Additional comparisons between methods would enhance paper's practical value"
   - *Issue:* Not all relevant comparisons made despite large scale
   - *Applies to SCALERL:* All relevant RL algorithm variants compared?

---

## TIER 2 - SUPPORTING EVIDENCE (Good References)

### 5. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/5IkDAfabuo.md`
**Paper:** Prioritized Generative Replay (PGR) for RL with Scaling Experiments
**Relevant Weakness Themes:** Inconsistent scaling trends, robustness of design choices

**Key Pattern:** Increasing synthetic data ratio benefits methods inconsistently
- *Quote:* "Trend is inconsistent and lacks clear intuition. Why does PGR scale better at r=0.75 than at 0.875?"
- **Applies to SCALERL:** Sigmoid curve fits may hide inconsistent scaling at different compute scales

---

### 6. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/cojJ2s1e35.md`
**Paper:** Training Stability Analysis
**Relevant Weakness Themes:** Statistical significance lacking, mathematical justifications inadequate

**Key Pattern:** Results presented without statistical rigor
- **Applies to SCALERL:** Are scaling law improvements within confidence intervals?

---

### 7. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/UUwrBhhsxT.md`
**Paper:** Large-Scale Training Study with Multiple Ablations
**Relevant Weakness Themes:** Limited evaluation scope despite scale, overstated generality claims

**Key Pattern:**
- *Quote:* "Evaluation conducted entirely in simulated controlled environments with no real-world empirical study"
- **Applies to SCALERL:** Tested only on math tasks—what about real RL applications?

---

### 8. FILE: `/home/wg25r/review_agent/iclr2025_data/human_reviews/gInIbukM0R.md`
**Paper:** Training Study with Design Choice Ablations
**Relevant Weakness Themes:** Limited baseline comparisons, missing analysis of design motivation

**Key Pattern:** Design choices not well justified beyond numerical results
- **Applies to SCALERL:** Why is off-policy async better than alternatives?

---

## CRITICAL WEAKNESS PATTERNS APPLICABLE TO SCALERL

### 1. SCALE ≠ INSIGHT
**Pattern:** Large-scale studies can present many experiments without providing mechanistic understanding

- **For SCALERL:** 400k GPU hours must be backed by understanding WHY sigmoid curves work, not just showing they fit
- **Evidence:** m29SV0n6DO, cu2CT2VAvs

### 2. EXTRAPOLATION CLAIMS ARE HIGH-RISK
**Pattern:** Extrapolation beyond training distribution requires careful validation

- **For SCALERL:** Sigmoid curve predictions beyond observed compute range need rigorous validation
- **Evidence:** P7f55HQtV8 ("Does it extrapolate beyond training distribution?")

### 3. DESIGN CHOICES NEED DEEP JUSTIFICATION
**Pattern:** Showing ablations exist doesn't mean design choices are well-understood

- **For SCALERL:** GRPO vs DAPO vs CISPO must have mechanistic explanations, not just performance numbers
- **Evidence:** m29SV0n6DO, R1hIXdST22

### 4. BASELINE FAIRNESS IS CRITICAL
**Pattern:** Hyperparameter tuning and model size fairness determine credibility of comparisons

- **For SCALERL:** Must ensure all RL algorithms (GRPO/DAPO/CISPO) equally tuned with same compute
- **Evidence:** R1hIXdST22 (critical weakness about untuned baselines)

### 5. STATISTICAL RIGOR REQUIRED
**Pattern:** Performance improvements without significance testing are meaningless

- **For SCALERL:** Need confidence intervals around sigmoid curve fits
- **Evidence:** R1hIXdST22, cojJ2s1e35

### 6. GENERALIZATION SCOPE MUST BE EXPLICIT
**Pattern:** Limited evaluation domain implies limited generality

- **For SCALERL:** Scaling laws trained on math tasks may not generalize to coding/reasoning
- **Evidence:** m29SV0n6DO, R1hIXdST22, UUwrBhhsxT

### 7. VAGUENESS IN DEFINITIONS IS DANGEROUS
**Pattern:** Undefined concepts lead to circular reasoning

- **For SCALERL:** Recipe components must be precisely defined
- **Evidence:** cu2CT2VAvs (vague definition of "state collapse")

---

## USAGE RECOMMENDATIONS FOR EVALUATING SCALERL

### Use as PRIMARY REFERENCES for:
1. Checking if sigmoid curve improvements are statistically significant
2. Validating extrapolation claims beyond observed compute range
3. Assessing whether 400k GPU hours yields mechanistic insights
4. Evaluating fairness of baseline algorithm comparisons
5. Determining if design choices are well-justified

### Most Critical Questions (from Tier-1 Reviews):
1. **Were GRPO/DAPO/CISPO baseline algorithms equally tuned?** (R1hIXdST22)
2. **What is the mechanistic reason sigmoid curves fit RL scaling?** (m29SV0n6DO, P7f55HQtV8)
3. **How were curve fits generated? Cross-validation used?** (P7f55HQtV8)
4. **Do scaling laws generalize beyond math tasks?** (m29SV0n6DO, R1hIXdST22)
5. **Are improvement margins statistically significant?** (R1hIXdST22, cojJ2s1e35)

---

## FILE REFERENCE GUIDE

**Tier 1 Files (Direct Applicability):**
- m29SV0n6DO.md - Large-scale video pre-training with ablations (1T tokens)
- R1hIXdST22.md - RL algorithm with empirical benchmarks and fairness issues
- P7f55HQtV8.md - Extrapolation claims and validation methodology
- cu2CT2VAvs.md - Large-scale study with design choice analysis and vague definitions

**Tier 2 Files (Supporting Evidence):**
- 5IkDAfabuo.md - Inconsistent scaling trends
- cojJ2s1e35.md - Statistical significance in results
- UUwrBhhsxT.md - Evaluation scope and generalization
- gInIbukM0R.md - Baseline comparisons and design justification
