# SafeProtein: Red-Teaming Framework and Benchmark for Protein Foundation Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Proteins play crucial roles in almost all biological processes. The advancement of deep learning has greatly accelerated the development of protein foundation models, leading to significant successes in protein understanding and design. However, the lack of systematic red-teaming for these models has raised serious concerns about their potential misuse, such as generating proteins with biological safety risks. This paper introduces **SafeProtein**, the first red-teaming framework designed for protein foundation models to the best of our knowledge. SafeProtein combines multimodal prompt engineering and heuristic beam search to systematically design red-teaming methods and conduct tests on protein foundation models. We also curated **SafeProtein-Bench**, which includes a manually constructed red-teaming benchmark dataset and a comprehensive evaluation protocol. SafeProtein achieved continuous jailbreaks on state-of-the-art protein foundation models (up to 70% attack success rate for ESM3), revealing potential biological safety risks in current protein foundation models and providing insights for the development of robust security protection technologies for frontier models. The codes will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SafeProtein, a red-teaming framework, and an accompanying benchmark SafeProtein-Bench, designed to evaluate the biosecurity risks of protein foundation models. Experiments show that under multiple masking and generation strategies, ESM3 and DPLM2 can be induced to recover potentially harmful proteins.

### Strengths
- This paper focuses on the critical issue at the intersection of AI safety and biosecurity, providing a systematic framework and benchmark for this problem.
- The authors propose a variety of generation strategies, including multiple masking strategies and five generation paradigms (including Foldseek structural prompts and score-guided beam/diffusion guidance).

### Weaknesses
Major
- The authors used ESMFold to predict structures of generated sequences, which is fast but not very accurate, and ESM-2 is used as the backbone network in ESMFold. Since the training sets of ESM2 and ESM-3 may have an overlap, the authors should further validate this potential bias.
- The highest success rates reported (e.g., for Strategies 2, 4, and 5 in Figure 3 and Table 4) all rely on providing the "Native Backbone Structure" as a prompt. This is an extremely strong assumption that weakens the "jailbreak" claim. 

Minor:
- Fig 1A: Maked -> Masked

### Questions
- What is the pLLDT distribution of the predicted structures? Low-quality structures should be rejected.

### Soundness
3

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
The paper proposes SafeProtein, a systematic red-teaming framework for protein foundation models, together with a benchmark, SafeProtein-Bench. The core task is masked recovery on real harmful proteins (toxins and viral proteins): the authors mask sequence token, then probe whether models can reconstruct both sequence and structure under five generation conditions; and a score-guided diffusion scheme. Evaluation is intentionally conservative: joint thresholds on sequence identity and structural RMSD (tighter cutoffs at lower mask ratios) to reduce false positives. Experiments on ESM3-open and DPLM2-650M show notably high “jailbreak” success, especially for ESM3 under score-guided decoding. The authors argue this indicates latent dual-use/biosecurity risks in current Protein FMs. They also acknowledge limits (no wet-lab validation; larger closed models not tested) and sketch future alignment/mitigation directions. This is an interesting approach, but my major concern is about whether the conclusion is solid enough.

### Strengths
- The work targets a real governance gap: standardized, model-facing tests for dual-use risks in Protein LMs. Translating LLM red-teaming into a protein co-design setting, and operationalizing “jailbreak” as conservation-aware masked recovery, provides a concrete, reproducible way to probe risk.

- The framework covers multimodal prompting (sequence + structure), multiple decoding strategies (including heuristic beam and diffusion guidance), and a clear evaluation protocol. The components form a closed loop that other groups could adopt or extend.

- The dataset focuses on experimentally resolved harmful proteins (toxins/regulated viruses), enforces length filters, and ships conservation profiles—making the masking task biologically meaningful (i.e., hitting likely functional residues rather than random spans).


- Results are broken down by model, masking strategy, mask ratio, and generation strategy; the added Strategy 4/5 experiments for ESM3 make the security story clearer (structure prompts and score-guided decoding substantially raise success rates).
- The overall presentation of the paper is excellent.

### Weaknesses
- The main claim relies on joint identity + RMSD as a proxy for “harmful capability.” That’s a reasonable first pass, but structural proximity does not guarantee functional equivalence. Without even lightweight functional proxies, risk remains “potential.” Adding small-scale functional surrogates (active-site recovery accuracy; catalytic residue recovery; binding/docking scores; interface geometry preservation) would strengthen the causal link from “recovery” to “operational risk.”


- For scale reasons the paper uses ESMfold to predict structures of generated sequences. If RMSD/ptm disagreements exist versus AF3 (or similar models), some success/failure calls and even the relative ordering of strategies might change at the margin. As I know, protein structure prediction models at current stage have significant variances(like Proteinix, AF3, ESMFold).


- Score-guided decoding uses sequence identity with a ptm penalty, which is close to the evaluation signal. The consistently high success rates across masks/ratios might partly reflect alignment between the guidance objective and the benchmark metric. A sensitivity study replacing the guidance score with more “functional” criteria (e.g., pocket geometry, interface complementarity, docking energy, active-site recovery) would clarify whether the effect persists under metric shifts.


- The Foldseek pipeline filters candidates by UniProt annotations to avoid harmful templates, but annotation coverage is imperfect. Structural near-neighbors could still encode homologous functional scaffolds. More detail on blacklists/thresholds/human review, plus error analysis of false admits/false rejects, would help assess whether Strategy 3 is genuinely “benign-guided” rather than “near-homolog-guided.”



- The paper notes protein sequences are not human-readable; consequently, “success” is defined operationally via recovery metrics. A short appendix that explicitly maps red-team success to real-world misuse preconditions would limit over-interpretation and guide policymakers.

### Questions
Could the authors check a representative subset and report agreement with ESMfold on RMSD/ptm (e.g., mean absolute differences, rank correlations for success/failure)? If discrepancies are sizable, how do conclusions change (e.g., which strategies look most risky)?


Any plans for evaluations on larger ESM3 variants? If direct access is infeasible, could the authors demonstrate transferability: learn attack strategies on open models and apply them, via constrained prompts/decoding knobs, to closed APIs?

### Soundness
2

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
3

### Summary
The submission introduces ​​SafeProtein​​, the first systematic red-teaming framework designed to evaluate and expose potential biosafety risks in protein foundation models (e.g., ESM3, DPLM2). The authors also present ​​SafeProtein-Bench​​, a curated benchmark dataset of harmful proteins (toxins, viral proteins) alongside a comprehensive evaluation protocol. By combining multimodal prompt engineering and heuristic beam search, SafeProtein successfully jailbreaks the protein foundation models, showing that protein foundation models present potential biosafety risks.

### Strengths
- **Novel Problem Formulation:** This submission introduces a new research direction by systematically studying the red-teaming of protein foundation models, a topic that has not been previously explored in the literature.

- **Comprehensive Methodological Exploration:** Given the absence of prior work, the authors propose multiple red-teaming strategies and conduct extensive experiments to compare their effectiveness, providing valuable insights into the relative strengths of different approaches.

- **Well-Constructed Benchmark:** The paper presents SafeProtein-Bench, a carefully curated benchmark for evaluating the dual-use potential of protein language models. The benchmark includes a manually verified dataset of 429 harmful proteins with experimentally determined structures, along with detailed dataset construction procedures and a rigorous evaluation protocol.

### Weaknesses
- **Limited Conceptual Foundation**: The fundamental premise of applying LLM-style "jailbreaking" to protein models appears to have limited scientific value. The work primarily combines concepts from two distinct domains without demonstrating significant depth in machine learning methodology or practical applicability. The approach lacks substantive technical innovation beyond this conceptual fusion.

- **Narrow Model Evaluation**: The study's scope is constrained to only two protein foundation models (ESM3 and DPLM2), with no comparative analysis of their respective results. This limited selection prevents meaningful generalizations about the broader landscape of protein models and their vulnerabilities.

- **Restricted Benchmark Scope**: As a benchmark, SafeProtein-Bench suffers from limited coverage, focusing exclusively on diffusion-based language models while neglecting other important architectural paradigms such as autoregressive models for protein sequences or diffusion approaches for backbone structure generation.

- **Insufficient Failure Analysis**: The paper lacks thorough investigation of failure cases, missing opportunities to provide insights into why certain attack strategies succeed while others fail. This omission limits the work's utility for developing effective defensive strategies against potential misuse.

### Questions
See the weakness above. 

The paper's fundamental premise raises an important question: **Why study the safety of protein foundation models, and is there significant evidence for its necessity?**

The work would benefit from stronger justification connecting the demonstrated adversarial vulnerabilities to plausible threat scenarios that justify the research investment.

### Soundness
2

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
3

### Summary
This work studies **red-teaming for protein language models (PLMs)** — a scenario in which an adversarial user attempts to use PLMs to discover harmful protein sequences. The authors construct a dataset of harmful proteins (toxins and viruses) and investigate whether a **generative model** can recover these harmful sequences when they are masked.

### Strengths
- Given the task of recovering sequences from curated databases, the authors do a good job of considering multiple sampling techniques.  
- The topic of **safety** and **dual-use** concerns in generative models for biology is both timely and important.

### Weaknesses
- I am not entirely convinced of the **relevance** of this specific type of red-teaming for producing harmful proteins. The idea that harmful proteins could be generated through *zero-shot prompting* of PLMs seems unrealistic, to the best of my understanding.  
  Can the authors justify why they believe harmful proteins would realistically be generated this way?  
- The paper does not sufficiently discuss prior research on the **generation of harmful proteins**, which is central to the stated goal of this work.  
  It would strengthen the paper if the authors devoted a substantial portion of the introduction — or even an entire section — to known or plausible **use cases of foundation models for harmful purposes**, and then clearly justified how their evaluation setup directly relates to these use cases.

### Questions
1. Can the authors provide a **comprehensive literature review** on potential use cases of foundation models for designing harmful proteins?  
2. Can the authors clearly explain the **role of foundation models** in these use cases and how their proposed evaluation helps **mitigate such risks**?  
3. Are the proteins in the curated harmful-protein database **part of the ESM-3 training set**?  
   If so, how does that affect the validity of the results?  
   More broadly, for the task of discovering **new viruses or toxins**, shouldn’t we primarily care about sequences **outside** the training set?

### Soundness
2

### Presentation
2

### Contribution
2
