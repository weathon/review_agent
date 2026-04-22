# Differentially Private Synthetic Data Generation with Diversity via APIs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Synthetic data has emerged as a key solution for preserving the privacy of original data in fields dealing with sensitive information, such as healthcare and finance. Recent advancements in foundation models have significantly improved the quality of synthetic data. However, most high-performance foundation models are only available as black-box APIs, limiting fine-tuning capabilities and requiring private data containing sensitive information to be transmitted to external servers. To address this issue, PE was introduced as a privacy-preserving synthetic data generation method that leverages genetic algorithms with black-box foundation models.
Nevertheless, due to its evolutionary process, PE tends to repeatedly focus on a limited subset of samples, leading to a significant reduction in the diversity of the generated synthetic dataset. Since diversity is a crucial factor for enhancing the utility of synthetic data and ensuring robustness across various scenarios, we propose Div-PE, an improved approach that overcomes the diversity limitations of PE through a sample-variant two-stage voting mechanism. This method enhances data diversity and yields a 17.2\% gain in FID and an 11.0\% increase in downstream accuracy on ResNet-18, averaged over ImageNet, Camelyon17, and UTKFace. Furthermore, Div-PE demonstrates its versatility by delivering strong experimental results not only on image data but also across other modalities, including tabular and text data, validating its applicability to a wide range of data types.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies that previous DP data synthesis method PE tends to repeatedly focus on a limited subset of samples, leading to a significant reduction in the diversity of the generated synthetic dataset. To solve this challenge, Div-PE is proposed by keeping an additional sample for each class. Privacy analysis and converge analysis is all considered. Dataset of multiple modality is included.

### Strengths
1.	This paper observes a very important drawback of PE, i.e. the decrease of synthetic data diversity with the use of VARIATE_API on the top-voted synthetic sample within each class. Another sample “non-winning” sample that is not selected by PE is selected and preserved to the next generation round to increase diversity.
2.	Variation prompts are carefully designed to further increase the diversity.
3.	Experiment on 3 modality is included, including image, text and tabular.

### Weaknesses
1.	Lack of baseline for tabular data. In Table 1, no baseline method is included for tabular datasets. As GreaT is applied to serialize table rows into natural language, Aug-PE is a direct baseline that can be compared. I would like to see the comparison.
2.	Lack of hyper-parameter selection study. As this is a paper considering differential privacy, one very important hyper-parameter is the differentially privacy budget $\epsilon$. The robustness of the proposed Div-PE under different $\epsilon$ should be studied but is currently missing in the main paper (please move it into the main paper), and lacks the comparison with baselines.
3.	Lack of in-depth study of the proposed Div-PE. Many hyper-parameters are selected without given a logic, i.e. it’s impact on the final performance is not studied. For example, $T, N_{can}$.

I think the overall idea is good and the proposed solution is interesting, but given the lack of experiments and mistakes (see Questions) contained in the paper, I cannot give a positive score for the current version which is not ready for being accepted as a top-tier conference paper. I will increase my score if the revison is good.

### Questions
1.	If I understand correctly, in Algorithm 1, each $S_t$ contains $2N_{syn}$ samples as 2 step voting each contribute $N_{syn}$ samples. Therefore, within each iteration, should the number of total candidates for step-1 voting be $2\times N_{syn}\times N_{can}$ but not $N_{syn}\times N_{can}$?
2.	The paper mentioned that, the second voting uses the selected synthetic data instead of private data to avoid privacy budget increasement. Is there any experimental comparison results on this? 
3.	I am curious, do you have any more detailed explanation on why removing adaptive variation (in Table 2) results in a sharp increase in FID and a dramatic decrease in Recall? The current analysis is too brief for me to understand.
4.	See weaknesses.

### Soundness
2

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
5

### Summary
This paper proposes Div-PE, a privacy-preserving framework for generating synthetic data using black-box foundation model APIs (e.g., Stable Diffusion for images, Llama-2 for text/tabular). It builds on PE, which uses evolutionary algorithms to guide synthetic data toward private distributions without fine-tuning or exposing raw data. However, PE suffers from diversity collapse, where generations converge to variations of a few high-fitness samples, reducing utility in downstream tasks. Div-PE addresses this via a two-stage voting mechanism that promotes balanced selection, ensuring broader ancestral lineages survive while maintaining DP guarantees.

### Strengths
- The diversity of synthetic images is important in the DP image synthesis area.

- The paper is easy to follow and well-written.

### Weaknesses
- Enhancing prompts to improve generative diversity is still fundamentally constrained by the model's inherent capabilities. This approach does not fundamentally overcome the limitations of the model's generative power. If the model lacks the ability to produce images similar to sensitive ones, such methods will not perform well. The author is encouraged to provide a more in-depth discussion and investigate how Div-PE performs in scenarios where the model's generative capacity is limited.

- The benefits and motivation behind the two-stage generation approach are not clearly articulated. Why does it work? In fact, the quality of synthetic images is generally low compared to real sensitive images, and using synthetic images for voting inherently introduces a lot of noise, which can negatively impact the voting results.

- The method relies on LLMs to generate a diverse prompt set. However, if the public information ($I_pub$ is insufficient or the LLM exhibits bias (e.g., cultural bias), the initial synthetic dataset ($S_0$) may lack diversity, potentially hindering convergence in subsequent iterations. The paper does not provide robustness evaluations, such as performance under noisy public information. Furthermore, both the adaptive mutation (Eq. 3) and demonstration-based mutation (Eq. 2) depend on the voting score ($V^{(1)}$), which may be unstable under noise ($\sigma$), leading to misleading guidance—where superior samples incorrectly influence inferior ones. The paper does not demonstrate the stability of these mechanisms under high noise (i.e., low $\epsilon$) conditions.

- The first stage assumes that each private sample contributes only one vote (i.e., sensitivity = 1). However, in multi-class settings, cross-class interference may increase the actual sensitivity, which is not addressed in the paper.

- Although the method draws inspiration from “natural ecology” to prevent monopolization, it does not model realistic evolutionary dynamics—such as mutation rate decay—which may lead to collapse over long-term iterations.

- This paper does not discuss or compare the approach of using public data for pretraining combined with DP-SGD-based differentially private image synthesis. I’m curious how Div-PE performs relative to such methods.

- The authors did not investigate how different APIs affect the quality of generation.

- Although the authors claim that Div-PE improves the diversity of the generated dataset, the results in Table 1 suggest otherwise. In fact, the recall is lower than the baseline, and the best value is incorrectly marked. Recall is a metric where higher values indicate better performance.

- Moreover, it is unclear how Camelyon17 achieves an accuracy close to 86.1% despite having a very high FID. The same concern applies to ImageNet. These experimental results are not convincing.

### Questions
Please refer to the weakness.

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
The paper introduces Div-PE, a framework for DP synthetic data generation using Private Evolution (PE). Div-PE introduces a two-stage voting mechanism that aims to mitigate the diversity collapse observed in prior PE methods where dominant samples monopolize the final synthetic dataset reducing the overall diversity. This sampling procedure relies on the post-processing property of DP and has no additional privacy cost. Empricially, the authors show Div-PE strengthens both utility and diversity across various modalities including image, text and tabular benchmarks.

### Strengths
- The paper is well-presented and clearly written.
- The paper highlights and solves a key limitation of PE regarding diverse data generation and proposes a well-motivated approach to fix this problem.
- The empirical results cover three different modalities (text, image and tabular) across multiple benchmark datasets and show the proposed method consistently improves utility and diversity over standard PE.

### Weaknesses
- Comparisons compare only against other PE baselines and existing DP synthetic data baselines are missing (particularly for the tabular setting), making it difficult to contextualize results against the broader DP synthetic data literature.
- While Div-PE claims similar overhead to standard PE, there is no direct measurement of overhead, particularly for the two-stage sampling procedure or the Auto-Prompt generation.
- The role of Auto-Prompt appears dominant in improving diversity, as suggested by Table 2. The contribution of the new two-stage voting seems unclear.

### Questions
1. The ablation in Table 2 suggests Auto-Prompt contributes most to diversity improvement. Is this ablation with BISTAGE for every option? I would have preferred to have seen a more detailed ablation across multiple datasets to really understand the contribution of each component. How does PE+Auto compare with just BISTAGE?
2. Related, how does auto-prompt work for tabular data generation? How exactly do you modify the GReaT encoding in this setting?
3. Could the authors provide more detailed runtime results? Specifically, what is the actual overhead of BISTAGE+auto?
4. How does the method perform across different DP budgets? Are diversity gains consistent under stronger privacy?
5. What accounts for the discrepancy between Table 2 and Figure 6 in coverage scores? I can't seem to find what the exact experimental setup (dataset etc.) was used for Figure 6? The Figure 6 ablation seems to imply that auto-prompt gets you most of the way in terms of best FID/coverage trade-off?
6. For the tabular setting, have the authors thought about comparing against existing tabular DP-SDG methods such as DP-CTGAN or SOTA marginal-based methods like AIM?

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
The paper proposes an improvement to a previous private evolution (PE) method to generate DP synthetic data using a foundation model API, with the aim of improving the diversity of the generated samples. The issue with the existing method is that all of the generated samples become very similar with many iterations of the algorithm, since they all become variations of a single example. The paper's solution, called Div-PE, is to ensure that variations of all initial examples are kept throughout the process. This is done by adding an additional step on top of the selection step of PE where poorly-performing examples are selected based on their similarity to well-performing examples. The second step uses the same DP statistics than PE, so it does not bring extra privacy cost. Div-PE is compared against PE and another variant of it on 7 datasets of image, text and tabular data. Div-PE outperforms the others in almost all settings.

### Strengths
The paper identifies a clear weakness in PE and provides a novel solution, which is demonstrated to solve the problem. Synthetic data generation with DP is an important problem, and algorithms effectively using API-only foundation models are especially useful due to the prevalence of these foundation models. Previous works in this area have not generated tabular data, which is a welcome addition in this paper.

### Weaknesses
The paper is missing many important explanations that are needed to fully understand the results. There are also many minor issues that together significantly reduce the clarity of the paper. In particular, Section 3 should explain SEED_API, VARIATION_API, the distance function $d$ and how different degrees of variation can be obtained from VARIATION_API in practice. Also, many important experimental details are missing. These are critical for reproducing the experiments and fully understanding their significance:
- Foundation model for tabular data is not specified.
- Not clear whether precision and recall in Table 1 are classification metrics or synthetic data evaluation metrics.
- It is not clear how SEED_API, VARIATION_API and the distance function are implemented in the experiments, or how different degrees of variation are obtained from VARIATION_API.
- Results do not have uncertainty estimates.

The paper is also missing comparisons with two recently published baselines improving the original PE: Tan et al. (2025) and Zhang et al. (2025). Another baseline that should be included is a conventional DP tabular data generator such as AIM (McKenna et al. 2022) for the tabular datasets.

Minor points:
- Figure 1: text is too small, and panel (a) is smaller than the others for some reason.
- Not clear why only one lineage surviving is important based on Figure 1. The synthetic data in panel (b) seems to have slightly better diversity than in panel (c). Figure 2 does a much better job of communicating the issue.
- Lines 147-148: "differ by at most one individual" is ambiguous. It could mean that one individual is added or removed, or that one individual is changed.
- Algorithm 2, line 14: should $u$ be used instead of $j$? Also, $S\_{syn}$ is not defined.
- Line 256: I don't understand what "vote within their own groups" means. Based on Algorithm 2, it looks like all the selected samples just vote together.
- Algorithm 1, line 13: the arguments are in a different order than the parameters of Algorithm 2. Also, the distance function is missing.
- Equations (2) and (3) are not reflected in Algorithms 1 and 2.
- Table 1: not all best values are bolded, for example recall on ImageNet.
- The proposed algorithm is called "DPSDivA" in Appendix C.2.

References:
- R. McKenna, B. Mullins, D. Sheldon, G. Miklau (2022) "AIM: an Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" Proceedings of the VLDB Endowment
- B. Tan, Z. Xu, E. P. Xing, Z. Hu, S. Wu. (2025) "Synthesizing Privacy-Preserving Text Data via Finetuning *without* Finetuning Billion-Scale LLMs" ICML
- J. Zhang, Y. Liu, J. Fu, Y. Hua, T. Zou, J. Cao, Q. Yang (2025) "PCEvolve: Private Contrastive Evolution for Synthetic Dataset Generation via Few-Shot Private Data and Generative APIs" ICML

### Questions
- Why does Private-PE degenerate into generating variants of a single example? Intuitively, candidates from more than one example should always get selected, since variants of a single example can only be nearest to a subset of training samples, while variants of another example would be nearest to another subset.

### Soundness
3

### Presentation
1

### Contribution
4
