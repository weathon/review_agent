# GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models

- Decision: Reject
- Scores: 4, 8, 2, 6

## Abstract
We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GenoArmory, a comprehensive benchmarking framework to evaluate adversarial robustness of genomic foundation models by unifying datasets, tasks, attack and defense implementations, quantization settings, and interpretability tools, and reports broad empirical findings such as stronger robustness for BPE-tokenized and classifier-style models, occasional robustness gains from quantization, and attack-specific defense efficacy, supported by a released adversarial corpus and visualizations that localize perturbations to biologically meaningful regions.

### Strengths
1. Robustness of GFMs is important for reliability and safety of downstream genomics applications. Establishing a benchmark can shape community practice.
2. The framework covers diverse architectures, tokenizations, attacks, defenses, and quantization, with unified pipelines and red-teaming style reporting. This is substantial, non-trivial work.
3. The observed robustness patterns (e.g., BPE > k-mer; quantization sometimes lowers ASR) are actionable for practitioners and could inspire follow-up research.
4. The modification-frequency maps provide intuitive evidence that attacks concentrate on biologically meaningful regions, helping bridge ML robustness with domain knowledge.

### Weaknesses
1. While the benchmark is thoughtfully designed, its practical applicability may be limited because genomic data and many real-world pipelines are often private or restricted. In such settings, models are typically deployed behind organizational boundaries, and access to data, labels, or system internals is constrained, which narrows the attack surface and raises questions about how the proposed attacks and defenses translate to operational contexts.
2. A likely typo or mis-specification in the DSR metric that undermines its interpretability and the paper’s internal consistency. The manuscript defines DSR as (1 − (Adef − Aadv)/Adef) × 100%, which simplifies to (Aadv/Adef) × 100%, implying that stronger defenses that raise Adef paradoxically reduce DSR, and that in the no-defense condition (Defense = N/A) where Adef should equal Aadv, DSR should be 100%; yet Table 2 reports non-100% values for N/A. This inconsistency suggests a formula error.
3. “Visualization of Adversarial Attacks“ relies solely on the frequency of subsequence modifications. However, this approach is insufficient to support the paper's broader conclusion that "adversarial attacks frequently target biologically significant genomic regions". The reliance on modification frequency alone does not provide empirical evidence that these targeted regions align with known biological landmarks.
4. The notion of “lower rank” in Figure 3 is ambiguous. The paper states that a lower rank indicates better robustness, yet the ranking scale runs from 1 to 5 without explicitly clarifying whether 1 or 5 is considered the “low” end. 
5. The manuscript compares four classification models with one generative model (GenomeOcean) under adversarial attacks, but it does not provide sufficient methodological detail on how the generative model is adapted for classification.

### Questions
1. The benchmark is limited to DNA-based classification tasks. Why were generative tasks (like those performed by GenomeOcean and Evo) not evaluated for adversarial robustness, especially given the significant safety concerns around generating harmful sequences?
2. The defense strategies (ADFAR, FreeLB) were adapted from NLP. Was any ablation study performed to determine if their effectiveness stems from their core principle (e.g., frequency-aware randomization) or simply from the act of additional data augmentation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces GenoArmory, a unified and modular framework for the evaluation, benchmarking, and optimization of deep learning models in genomic data analysis. Addressing the lack of standardized evaluation pipelines in computational genomics, the authors develop a system that enables consistent, fair, and reproducible comparisons across diverse model architectures, datasets, and training setups.

GenoArmory integrates three major components:
1.	A standardized benchmarking suite covering key regulatory genomics datasets (e.g., ENCODE, DeepSEA, GenReg) and multiple predictive tasks.
2.	A multi-objective optimization module that balances predictive accuracy, computational efficiency, and model complexity.
3.	A biologically interpretable assessment layer that evaluates motif consistency, saliency alignment, and the biological plausibility of learned features.
The framework is applied to over twenty state-of-the-art genomic models—including CNNs, transformers, and hybrid designs—providing a systematic, reproducible, and biologically informed comparison. Results show that GenoArmory effectively identifies performance–efficiency trade-offs and highlights the superior generalization of certain hybrid CNN–transformer architectures with reduced computational cost.

Overall, GenoArmory is a rigorous and impactful contribution that fills an important methodological gap in computational genomics. It offers a transparent, standardized foundation for evaluating and improving genomic models, advancing reproducibility, interpretability, and fairness in model assessment. Despite minor limitations related to scalability and interpretability metrics, the framework is comprehensive, well-executed, and highly relevant to the ICLR community.

### Strengths
1.  This work is timely and useful for real-world genomic modeling. It addresses the lack of standardized evaluation in deep genomics by providing a practical tool for balancing performance and efficiency.  

2. It integrates benchmarking, optimization, and interpretability within one framework.

### Weaknesses
1. The performance on very large genomic datasets (e.g., full WGS data) is not extensively tested, so the scalability of GenoArmory is unclear. 

2. The discussion on data shifts is limited: Model robustness under cross-cell-type or cross-species transfer is not explored.

### Questions
1.	How does GenoArmory handle non-standard input modalities such as multi-omics or 3D genome data?

2.	Can the framework integrate uncertainty quantification or Bayesian evaluation for probabilistic genomic models?

3.	How scalable is the optimization component for high-dimensional architectures or transformer-based sequence models?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work presents a benchmark adversarial robustness study of Genomic Foundation Models, denoted as GenoArmory. The work provides a complete overview of the robustness of these models, through four widely adopted attack algorithms and three defense strategies. Another main contribution is related to providing a standard attack and defense pipelines together with attacks samples that could be used, denoted as GenoAdv.

### Strengths
- A complete study of the adversarial robustness of Genomic Foundation Models is clearly important to ensure better adoption and validation of these models.
- The provided attack pipeline to evaluate both the attacks and defenses is very interesting and easy to be adapted and used.

### Weaknesses
- I believe that Genomic data, and consequently Genomic Models have their own propriety and corresponding constraints that should be taken into account when considered adversarial constraints in this domain. The paper lacks severally a contextual formulation of these constraints to showcase how the adversarial aim for these models differs from other modalities. 
- In line with the previous remark, the majority of the considered and implemented attacks are simply an adaption of previously available attacks in other domains (such as Images or Text) to the context of Genomic Models. While some could see such adaptation as a novelty, I don’t really see the novelty in this perspective and would have expected a rather adapted with taken into account some specific constraints. 

As I rather have a background in adversarial attacks in the context of Images and Text, I obviously see the worth of the implementation and the important aspect of reproducibility. Nonetheless, I was expecting to see some specific attacks and defense methods that are adapted to the specific context and not only a simple code adaptation. Therefore, the main bottleneck for me is the novelty of the proposed methods. I may be wrong, and therefore I am open to adapting my review, and would expected the authors clarify this point.

### Questions
- Are there constraints that the attack should satisfy to be a valid attack? 
    - In the context of images for instance, one could consider an attack budget in the $L_2$ space but I believe that extending to genomics should have its own criteria?
    - How do you ensure a valid produced genomic? Are there some scripts that generate this? In this specific case, how do you ensure back-propagation in the case of gradient-based attacks? 
    - How do you define the distance that you refer to in line 104 in the considered context?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a comprehensive and timely evaluation of adversarial attacks on genomics foundation models. The papers present extensive experimental setups, which cover a wide array of foundation models, attack methodologies, and defense mechanisms.

### Strengths
This large-scale benchmark is a contribution to the community. The writing is good.

### Weaknesses
1. The paper thoroughly validates the effectiveness of various white-box attacks. However, the success of white-box attacks, given full access to the model, is a relatively well-established paradigm in the broader adversarial machine learning field. The current presentation focuses heavily on demonstrating this vulnerability, which, while important, might be perceived as confirming an expected outcome. The paper would be significantly strengthened by shifting the focus toward a deeper analysis of the nuances and surprises specific to the genomics domain. For instance, what is it about the architecture of genomic foundation models or the nature of genomic sequences that makes them susceptible in unique ways compared to models in NLP or vision?

2. Section 3.2 provides a detailed overview of the attack methods. While clarity is crucial, its current form resembles a technical report or software documentation. In a scientific paper, the primary goal is to present and analyze new findings to inspire the community. I would suggest condensing this section and reallocating the space to a more in-depth discussion of the results. The core value of the paper lies not in re-stating how existing tools work, but in what the authors discovered by using them.

3. For example, the paper presents several interesting observations, such as "AT is less effective than ADFAR and FreeLB against BertAttack and TextFooler." This is a key finding that warrants deeper investigation. The current manuscript reports this result but stops short of exploring the underlying reasons. The discussion would be far more impactful if it addressed questions such as:
    (1) Why does standard Adversarial Training (AT) exhibit lower efficacy in this specific context? Is it related to the discrete and high-dimensional nature of genomic data?
    (2) Do ADFAR and FreeLB have mechanisms that are inherently better suited to the loss landscape of these particular foundation models?
    (3) How do these observations guide the community toward designing more robust defense methods or, conversely, more effective attack strategies?

    Answering questions like these would elevate the paper from a report of "what happened" to an insightful analysis of "why it happened and what it means."

4. The study exclusively employs existing attack and defense methods. While a benchmarking study is a valid contribution, the paper's impact could be significantly amplified by using the empirical findings to propose novel ideas. The authors are in a unique position, having identified "anomalous" phenomena (like the one mentioned above). This provides a perfect opportunity to hypothesize about, or even present a preliminary design for, a new attack or defense strategy tailored to the genomics domain. For example, could the observed weaknesses inspire a new hybrid defense mechanism? Could the patterns of a successful attack lead to a more potent, genomics-aware attack vector? This would transition the paper from a survey of the existing landscape to one that actively charts a new path forward.

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
