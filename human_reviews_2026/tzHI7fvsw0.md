# MOLLEO+: Towards Optimized Use of LLMs for Drug Discovery

- Decision: Reject
- Scores: 4, 0, 4, 4

## Abstract
Large language models (LLMs) have recently emerged as a promising tool for small-molecule generation in drug discovery. One notable recent state-of-the-art work in this field is MOLLEO, which combines an evolutionary algorithm with an LLM that acts as the operator for making crossovers and mutations on the ligand population. MOLLEO demonstrates strong results on optimizing molecular docking scores, but several aspects of their model are not well suited to real-world drug discovery. We introduce MOLLEO+, an optimized LLM workflow for de novo molecule generation.  First, we replace docking with the recently released biomolecular foundation model Boltz-2 as an oracle, which improves the  predicted binding affinity of generated molecules using gold-standard molecular dynamics by over 100\%. Second, we incorporate knowledge of existing ligands, which is present in most practical drug discovery scenarios, using ligands from BindingDB instead of ZINC 250k as the starting population for the genetic algorithm. Third, we propose a fine-tuning strategy to better modify existing ligands towards higher activity. We demonstrate the superiority of MOLLEO+ on the receptor tyrosine kinase c-MET and the BRD4 protein, yielding an improvement over state-of-the-art baselines by up to 20\% for Boltz-2 binding affinity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed MOLLEO+, an optimized LLM workflow that improves MOLLEO for de novo molecule generation. Specifically, they replace docking with the recently released biomolecular foundation model Boltz-2 as an oracle which can improve the predicted binding affinity. Additionally, they incorporate knowledge of existing ligands and propose a fine-tuning strategy to better modify existing ligands towards higher activity. Those components together demonstrate the superiority of MOLLEO+ on the receptor tyrosine kinase c-MET and the BRD4 protein.

### Strengths
- Replace the docking-based fitness evaluator with Boltz-2, which increases the mean Absolute Binding Free Energy of generated molecules by over 100%.
- Utilize a starting population of ligands based on BindingDB, which increases mean predicted binding affinity of generated compounds by up to 15%
- Propose a novel post-training framework to fine-tune LLMs for lead optimization tasks using a semi-synthetic dataset, which can improve  the quality of its generated molecules

### Weaknesses
- The experiments only evaluate two targets, c-MET and the BRD4. It’s unclear that if this method can be applicable to more broad targets.
- Using Boltz-2 to evaluate ABFE also introduces bias by the computational method itself.
- This method may be less effective for those targets that lack sufficient ligand data.
- Generated molecules have less novelty due to the use of strong known binders from BindingDB as the starting population

### Questions
Please refer to the weakness section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The authors propose MOLLEO+, an extension to the existing MOLLEO framework for molecular optimization via the use of large language models. The authors demonstrate that the use of a Boltz-2 oracle and an improved starting population leads to better generated molecules (in terms of targeted molecular properties). The authors also suggest a supervised fine-tuning framework for molecular editing, but show that the fine-tuned model still underperforms larger models. In my opinion, the work that the authors propose here falls within the way that the original MOLLEO was designed to be used. Critically, their work does not extend the capabilities of the original framework. While their supervised fine-tuning approach looks promising, it is underevaluated. Thus, this work is not worthy of acceptance to ICLR.

### Strengths
The authors show the intended use of MOLLEO in this work and the benefits that it can lead to. This is instructive for users who are curious about adopting MOLLEO in their molecular design efforts.

### Weaknesses
Naturally, it should be trivial that when using MOLLEO, one chooses an oracle that is supposed to correlate to the property that they care about (the authors choose ABFE). I don’t think it’s a surprising result that Boltz-2 gives better correlations for ABFE as it was trained to reproduce binding affinities. Since docking is constructed to be extremely approximate and is designed for speed in screening extremely large libraries, I also don’t think it’s surprising that using a docking oracle results in poor estimates for ABFE. In light of this, Section 4.1 appears to be a very limited benchmark of oracle performance confounded by MOLLEO’s evolutionary search, since MOLLEO + AutoDock is attempting to maximize properties (i.e docking scores) completely uncorrelated to their intended target. An independent benchmark on a set of molecules independent of MOLLEO’s structural generation would have been much more appropriate. 

The authors also report that the molecules have better Boltz-2 affinity scores because they use a better starting population correlated to their property of interest. In the original MOLLEO work, the authors have already evaluated something equivalent (best molecules in ZINC 250K) in Table 3. Again, the increased performance of MOLLEO+ versus the unmodified MOLLEO (Boltz-2) comes from aligning the starting population towards their intended goal, which seems to me as the normal way to use MOLLEO to begin with. MOLLEO was never designed to be strictly used only with ZINC 250K, and so it is confusing to me that the authors frame the correct usage of MOLLEO for their problem as an optimization of the original framework. 

The authors also do not report (or maybe I missed it somewhere) if the results of their runs for each method were the result of multiple runs with random seeds, so it’s hard for me to assess if the performance of their method is consistent. 

The supervised fine-tuning method shows promise -- it would indeed be desirable if the molecular edits by the LLM encapsulate the way that medicinal chemists would think. However, Tanimoto similarity is not a smooth metric, given that closely related structures (i.e changing functionalizations on a backbone/template) can also result in large differences in Tanimoto similarity. While it is also the cornerstone of structure-activity relationships, there’s also no guarantee that the same structural edits necessarily correlate with activity across different backbones or chemical environments. Without being able to see what these “ligand chains” look like, it’s hard to rationalize with physicochemical principles that the models are being fine-tuned on the correct concepts. The aggressiveness of the fine-tuning also is not evaluated, as one can imagine that in a strongly overfit case, all starting structures immediately collapse towards their respective cluster’s local maxima in the training data. In a real-world case, accessing a dataset that is large enough for a model to learn these edits relevant for their target protein may not exist as well, so it would have been interesting to see if there could be some level of general chemistry knowledge learned from a dataset not directly related to their optimization task.

### Questions
1) Were there multiple runs for each method with a random seed?
2) Was your BindingDB initial population sorted by experimental or Boltz-2 binding affinity?
3) What is an example of this ligand chain? Do you observe these chain edits being used over the course of the MOLLEO+ run? Do these chain edits actually result in a positive chain in molecular property? 
4) How do you support the claim that “GPT-4.1mini is known to be a stronger model overall”? Is there a relevant source?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present Molleo+, an iteration of the original Molleo method, in which the proxy for the binding energy oracle is replaced with Botz2, a different initial set of molecules is used, and an additional fine-tuning step is introduced on paired low- and high-affinity binders to better focus the model on optimizing affinity. The approach is evaluated on the task of identifying higher-affinity binders for two target proteins, c-MET and BRD4.

### Strengths
1.	The problem statement and main contributions are clearly articulated, and the manuscript is easy to follow. The task of improving binding affinity is important in the life sciences, and the proposed approach is well-suited to addressing it.
2.	Molleo with Boltz2 and Molleo+ demonstrate strong performance improvements across the reported metrics, in some cases making the Molleo lineage competitive with or superior to other baselines.
3.	The authors also provide useful insights into the factors driving these results, such as the correlation analysis presented in Figure 3

### Weaknesses
The central theme is that the evaluation requires greater robustness:

1.	Demonstrating meaningful benefit: More evidence is needed to show that the Molleo+ approach provides a genuine performance improvement (Table 2 and Table 3). First, the overrepresentation of high-affinity binders in the training data may be driving the observed results (see Question 1). Second, since binding affinity is evaluated using Boltz2, Molleo+ may effectively be distilling Boltz2 rather than making genuinely improved affinity predictions (see Question 2).

2.	Target-dependent performance: There is also a risk that the reported gains are partly due to the particularly poor Autodock binding scores for the selected targets (as illustrated in Figure 3). Consequently, the observed improvement may not generalize to other targets. (see Question 3)

Without the above the novelty would be limited to using higher quality data (via Boltz2) as input to the original Molleo.

### Questions
1.	Filtered Mean analysis (Table 3): The Filtered Mean is computed based on ligands with scores below a threshold of 0.5. What is the resulting average maximum similarity for this subset? If it exceeds 0.35 for c-MET and BRD4 (the value observed for Molleo and Molleo (Boltz2)), the authors could examine a subset of data with a mean similarity of 0.35. This would enable a fairer comparison with Molleo (Boltz2) and Molleo.

2.	Orthogonal evaluation of binding affinity: For Molleo+, it would strengthen the work to include an orthogonal method for estimating binding affinity. Specifically, could the evaluation using ABFE values from Table 1 also be performed for BRD4, and for Molleo+ across both targets?

3.	Effect of affinity quality: The authors could further analyze performance at different levels of correlation to determine whether there is a direct relationship between the quality of affinity estimates and predictive performance. For instance, controlled noise could be added to the c-MET binders to progressively reduce correlation and assess sensitivity.

### Soundness
2

### Presentation
3

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
The paper presents an enhanced version of the generative lead-optimization framework MOLLEO, replacing the traditional docking oracle with the AI foundation model Boltz-2, seeding the starting population from BindingDB instead of ZINC to bias toward known actives, and fine-tuning a compact Llama model on a semi-synthetic dataset to improve crossover and mutation operations. Empirical validation focuses primarily on the target c MET (with ABFE comparisons) and a second target BRD4 (via Boltz-2 scoring) and claims that the proposed method outperforms the original docking-based baseline in terms of predicted and simulated binding affinities while maintaining medicinal chemistry viability.

### Strengths
1. The motivation to replace docking with a modern AI oracle is good and aligns with recent advances in binding-affinity modelling. The authors present data showing that Boltz-2 better correlates with ABFE on c MET than docking.
2. The switch to a BindingDB-derived starting pool is sensible: it reflects realistic chemical spaces of known actives and helps the algorithm exploit known chemotypes while still maintaining diversity. The clustering and fingerprint details are clearly described.
3. The semi-synthetic SFT for a Llama model is well described with prompt templates, training curves and validation splits. The ablation shows benefit of tuning the LLM operator.
4. The ABFE implementation is described in greater detail than is often seen: the authors provide schedule detail, pose preparation with alignment, and run time data which enhances reproducibility.
5. The authors include medicinal-chemistry relevant metrics and apply novelty filtering against the starting pool, which demonstrates awareness of downstream requirements beyond raw binding affinity.

### Weaknesses
1. External validity is limited: the strongest evidence comes from one target (c MET) for ABFE validation. For BRD4, only oracle scores (Boltz-2) are shown. That raises concerns about generalisability across target families.
2. The ABFE protocol lacks sufficient detail on convergence diagnostics (e.g., number of replicas, uncertainty estimates, window closures) and the starting-pose difference between arms (docking vs Boltz-2) may bias the comparison.
3. Baselines are not fully aligned: the change in oracle, starting pool and LLM are all conflated. There is insufficient control to isolate each contribution (oracle swap, starting pool shift, LLM fine-tuning) and the claim of GPT-4.1-mini being stronger than GPT-4 is not substantiated.
4. Novelty and chemical novelty analysis is limited: the paper uses a Tanimoto similarity cutoff against the starting pool but does not report scaffold novelty, nor similarities against broader medicinal-chemistry databases like ChEMBL or patent literature.
5. Statistical reporting is incomplete: confidence intervals and error bars are missing. Effect sizes and sample sizes vary across arms (e.g., 10 vs 20 samples) which complicates interpretation of statistical significance.
6. The synthetic SFT dataset raises quality questions: chemical validity and label fidelity of synthetic-data edits are not clearly verified (e.g., no PAINS filtering, no human spot checks) and the relative contribution of SFT vs other changes (oracle, seed pool) remains ambiguous.
7. Crucially, the choice of Boltz-2 as the oracle is not justified with full benchmarking against experimental binding affinity data on the chosen targets. Without that, the practical usefulness of the gains is unclear: if the correlation between Boltz-2 and experiment is low then improvements in the generative workflow may not translate into real-world value.
8. Similarly, the practical utility of metrics like QED and SA score are in question. Those are frequently referenced in scientific literature, however, it is a common knowledge that these metrics are poor proxies for real-world drug-likeness and synthetic accessibility.
9. The discussion section lacks deeper reflection on failure modes, target limitations, and trade-offs (e.g., cost vs quality, when to use this method vs classic workflows).

### Questions
1. On oracle choice and experimental correlation. What is the correlation between Boltz-2 predicted affinities and experimental binding measurements on c MET, BRD4 and ideally one more independent target family? Please report Pearson and Spearman coefficients with confidence intervals, calibration plots, and address any systematic bias across affinity ranges. How do these correlations compare with docking scores (AutoDock Vina or similar) and ABFE estimates? Given that Boltz-2 is cited as “near-FEP accuracy” in benchmarks, please provide specifics on this for the chemical space relevant to your study.
2. If the correlation to real experimental binding affinity is moderate, what are the practical implications of your workflow for medicinal-chemistry decision-making (hit-to-lead, lead optimisation)? What threshold would you consider sufficient for this method to be actionable in a drug-discovery funnel, and in which scenarios (library size, novelty requirement, target difficulty) would this workflow add meaningful value?
3. On ABFE protocol. Could you provide full details on how many independent replicas per ligand were run, the resulting uncertainties (e.g., standard error, confidence intervals) and convergence diagnostics (e.g., window overlap checks, cycle closure if any)? Also clarify whether both arms (docking-oracle vs Boltz-2) used identical simulation schedules, starting poses, restraints and stopping criteria, and provide the wall-clock time per ligand. If starting poses differ, then how do you mitigate pose-biasing the ABFE comparison?
4. On SFT dataset quality. How did you verify chemical validity and label fidelity of the semi-synthetic dataset? Did you apply automated checks (valence rules, PAINS filters, structural alerts) or human spot-checks? What fraction of generated edits did you discard for implausibility? Please provide ablation results isolating the effect of SFT only (keeping starting pool and oracle fixed) and report whether the fine-tuned LLM or the dataset quality was the limiting factor.
5. On novelty and chemical novelty. Could you expand novelty analysis to scaffold-level novelty, and maximum similarity to any known binder across BindingDB or ChEMBL? Also please show sensitivity of results to different similarity thresholds rather than a single cutoff. Provide counts of unique scaffolds generated and compare to baseline.
6. On baselines and fairness. Please include control runs that isolate each change independently: (a) original LLM + docking oracle + ZINC start, (b) original LLM + Boltz-2 oracle + ZINC start, (c) original LLM + docking oracle + BindingDB start, (d) fine-tuned LLM + docking oracle + BindingDB start, all under equal compute budget and same number of generations. Also consider including a small ABFE-guided genetic algorithm (without an LLM) to establish a cost-quality upper bound.

### Soundness
2

### Presentation
3

### Contribution
2
