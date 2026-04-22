# OmegAMP: Targeted AMP Discovery via Biologically Informed Generation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Deep learning-based antimicrobial peptide (AMP) discovery faces critical challenges such as limited controllability, lack of representations that efficiently model antimicrobial properties, and low experimental hit rates. To address these challenges, we introduce OmegAMP, a framework designed for reliable AMP generation with increased controllability. Its diffusion-based generative model leverages a novel conditioning mechanism to achieve fine-grained control over desired physicochemical properties and to direct generation towards specific activity profiles, including species-specific effectiveness. This is further enhanced by a biologically informed encoding space that significantly improves overall generative performance. Complementing these generative capabilities, OmegAMP leverages a novel synthetic data augmentation strategy to train classifiers for AMP filtering, drastically reducing false positive rates and thereby increasing the likelihood of experimental success. Our in silico experiments demonstrate that OmegAMP delivers state-of-the-art performance across key stages of the AMP discovery pipeline, enabling us to achieve an unprecedented success rate in wet lab experiments. We tested 25 candidate peptides, 24 of them (96%) demonstrated antimicrobial activity, proving effective even against multi-drug resistant strains. Our findings underscore OmegAMP's potential to significantly advance computational frameworks in the fight against antimicrobial resistance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces OmegAMP, a diffusion-based framework for controllable antimicrobial peptide (AMP) generation that couples a biologically informed embedding with a flexible conditioning scheme over AMP label, length, charge, and hydrophobicity.

It proposes two targeting modes Property Conditioning (ranges) and Subset Conditioning (derived from exemplar sequences)—to steer generation, including toward species-specific activity profiles.

The authors further design a stringent AMP filtering stage using XGBoost classifiers trained with a weighted loss and synthetic negative augmentation (random, shuffled, mutated), drastically reducing false positives on challenging negatives (signal, metabolic, added-deleted).

Extensive in silico evaluations show SOTA generation and classification metrics, and wet-lab validation on 25 peptides yields a 96% hit rate with high potency, including efficacy against multi-drug resistant strains.

### Strengths
Originality: The biologically informed, invertible residue embedding tied to a diffusion model plus the dual conditioning strategy (property/subset) is a novel and practical way to achieve fine-grained and species-targeted controllability. The synthetic-negative training and weighted loss for classifiers directly address the chronic high-FPR problem in AMP screening.

Quality: The work presents thorough evaluations: ablations on embedding scales, conditional control MAEs, classifier robustness on multiple hard negative sets, external dataset testing, and comprehensive wet-lab validation with MIC distributions across 17 strains. Clear reporting of metrics (AUPRC, Prec@100, LR+, diversity/uniqueness/novelty, fitness) and careful cross-validation support the claims.

Clarity and significance: The paper is well-structured with precise methodological detail (conditioning objective, CADS, embedding injectivity/decoding, classifier features/loss), clear figures/tables, and actionable guidance for practitioners. Demonstrating near-perfect wet-lab hit rates and strong potency against MDR pathogens indicates high potential impact for antimicrobial discovery.

### Weaknesses
Conditioning expressivity: The conditional-based generation is not a new idea (Accelerated antimicrobial discovery via deep generative models and molecular dynamics simulations). Besides, the current cond(·) vector captures AMP label, length, charge, and hydrophobicity, but omits other relevant biophysical/ADMET factors (e.g., protease stability, hemolysis, toxicity, aggregation, secondary structure motifs). Extending the conditioning space and validating multi-objective trade-offs would increase practical utility.

Dataset and bias considerations: Generative training uses AMP-like general peptides from Peptipedia and EV labels from DBAASP with standardization filters, which may introduce selection bias and label noise. More analysis on data quality, potential leakage, and sensitivity to dataset composition (e.g., species distribution) would strengthen reproducibility and fairness claims.

### Questions
Could the conditioning vector be expanded to include toxicity/hemolysis proxies or structural targets (e.g., predicted helicity, aggregation propensity), and how would that affect classifier precision and wet-lab hit rates?

For species-specific generation, what is the sensitivity of Subset Conditioning to the size/quality of the exemplar set, and can you provide guidance on the minimal number and diversity of exemplars needed to achieve gains?

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
3

### Summary
The paper proposes an integrated framework, OmegAMP, for antimicrobial peptide (AMP) discovery, comprising: diffusion-based sequence generation (with biologically inspired residue-level embeddings and flexible conditional control) and low–false-positive filtering classifiers (XGBoost with synthetic negative data augmentation and a weighted loss).

### Strengths
The paper proposes an integrated framework, OmegAMP, for antimicrobial peptide (AMP) discovery, comprising: diffusion-based sequence generation (with biologically inspired residue-level embeddings and flexible conditional control) and low–false-positive filtering classifiers (XGBoost with synthetic negative data augmentation and a weighted loss).

### Weaknesses
1. Scoring generated samples with the authors’ own classifier inevitably introduces system-internal consistency bias (despite providing HydrAMP-MIC as a secondary scorer). It is recommended to include more external independent classifiers or larger-scale blinded experimental test sets to reduce evaluator–generator coupling.
2. While “random/shuffled” negatives are reasonable, introducing only five point mutations does not necessarily render sequences inactive, leading to potential mislabeling. Although the paper uses a weighted loss to down-weight non-EV data, it is advisable to quantitatively report sensitivity of performance to mislabeling rates, or increase mutation strength/proportion as a control, to more robustly define non-AMPs.
3. The generative model’s “large-scale general peptides” are derived from Peptipedia’s predicted label set (non-EV), which may be noisy. Although the authors acknowledge and distinguish this, they should more clearly quantify how such noise affects the generative distribution and controllability (e.g., a training ablation without that data).
4. The “conditioning vector” for subset conditioning includes only global attributes (length, net charge, hydrophobicity, AMP tag), without explicitly encoding species/strain information or sequence motif/positional patterns. If species selectivity relies on finer sequence–structure features, aligning global distributions alone may be insufficient.
5. The claimed species-specific “success” is mainly indirectly supported by elevated classifier scores, while feature analysis indicates “mean net charge” is highly dominant; this may cause species specificity to appear merely as matching certain global attribute distributions.

### Questions
Introduce third-party independent classifiers or conduct “blinded scoring” with collaborators to reduce system-internal bias.
Systematically study how different mutation ratios/patterns affect classifier FPR/TPR and report sensitivity curves, or align training with a larger EV non-AMP set to mitigate mislabeling risk.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes OmegAMP, a controllable diffusion-based framework for antimicrobial peptide (AMP) sequence design. OmegAMP represents each amino acid using physiochemically-inspired property values to obtain biologically-informed peptide embeddings. It further proposes property conditioning techniques to guide the generation toward desired functional characteristics. In addition, the authors introduce synthetic negative sample augmentation for classifier training to reduce false positives in AMP prediction.

### Strengths
- The authors introduce practical conditioning methods that offer fine-grained control over desired AMP properties, leading to state-of-the-art generation performance.
- Comprehensive experiments are presented, covering both in silico and wet-lab validation. The 96% experimental hit rate is particularly impressive.
- The authors theoretically and empirically justify the synthetic negative sampling strategy, alleviating concerns about inadvertently labeling true AMPs as negatives.

### Weaknesses
- Could you clarify the motivation for including “is AMP” in the conditioning vector? The proposed model is specialized for AMP generation, so, as stated in the manuscript, this entry is always 1 for all training and inference samples.
- The authors mention that PC offers flexibility but may not capture inherent correlations between properties. However, empirical results show PC often performs better than SC. Could the authors provide more intuition for this?
- The embedding quality should be compared against ESM2 embeddings, which are frequently used in AMP generation baselines.
- The reconstruction loss (Eq. 3) seems to minimize the discrepancy against the target sequence’s embedding. In that case, shouldn’t Eq. 3 use $E$ instead of $s$?
- Minor) Table 11 would be more informative if presented in the main manuscript, since controllability result is central to fully assessing the generation performance.

### Questions
- How does the performance change when conditioning on the average values of each numerical property (e.g., charge, hydrophobicity) within a predefined range, instead of randomly sampling the property values?
- How many AMPs are used to construct $\mathcal S_{\text{target}}$ in the SC?

### Soundness
3

### Presentation
3

### Contribution
3
