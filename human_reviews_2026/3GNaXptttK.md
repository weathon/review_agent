# Head-Level Mechanistic Attribution for Hallucination Control: Training-Free Counteractive Pruning in LVLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Large vision-language models (LVLMs) excel at multimodal tasks but often generate instance-level object hallucinations, describing nonexistent objects.
Since existing methods overlook functional conflicts within attention heads and lack principled, fine-grained attribution and intervention at the head level, hallucination suppression is often accompanied by a substantial loss of semantic informativeness.
To overcome these limitations, we propose HACP, a unified framework that enables fine-grained internal hallucination control via precise intervention at the attention head level. Specifically, we introduce InfoSpectralScore, a novel attribution metric based on eigen-decomposition with spectral variance and entropy penalties, which allows for the accurate identification of hallucination-inducing heads. We further develop a dynamic, training-free pruning strategy that adaptively suppresses hallucination-prone heads while reinforcing faithful heads during inference. 
Extensive experiments across multiple LVLMs and benchmarks demonstrate that HACP achieves state-of-the-art hallucination mitigation, substantially reducing hallucinations while better preserving caption informativeness compared to existing approaches, thus offering a robust and transferable solution for controllable and interpretable multimodal generation. The source code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training-free method to mitigate hallucinations in LVLMs. The approach includes head-level pruning by identifying head-level semantic attribution via a self-defined infospectral score and reweighting the output vector of each head by pushing it away from hallucinated directions and toward faithful directions.

### Strengths
1. This paper proposes a principled, fine-grained attribution and intervention framework at the head level to mitigate object hallucination, which is an interesting direction.

### Weaknesses
1. The writing in both the method and experimental sections is quite confusing. A major revision is recommended to improve clarity.
2. The method itself is not intuitive, well motivated, or mathematically solid. The method is limited in downstream application. The estimation of statistical variables (e.g., hallucinated direction, infospectral score threshold) depends on a batch of samples, which makes it difficult to adapt to cases where only a single image is available.
4. It would be helpful to include more recent LVLMs such as Qwen2.5-VL and LLaVA-v1.6 in the experimental results.
5. Both CHAIR and POPE focus only on object existence hallucination. Including a more extensive benchmark such as MME[1] would strengthen the paper.

[1] MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models

### Questions
1. Line 122: “LVLMs still suffer from instance-level hallucinations, largely due to functional conflicts among attention heads.” Is this supported by prior work, or is it an overclaim?
2. How do you define the hard100 subset of POPE evaluation? Why report only this subset instead of the full set as in previous papers?
3. Line 363: “For each model, we define three attribution layer groups: LLaVA uses [5,18], [19,26], and a merged [5,26]; Shikra-7B uses [3,13], [14,28], and a merged [3,28].” Could you clarify what this grouping means and how it’s used in the method?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the critical problem of instance-level object hallucination in Large Vision-Language Models (LVLMs), where models describe objects not present in the visual input. The authors argue that current mitigation techniques often force a trade-off, reducing hallucinations at the cost of semantic informativeness (i.e., caption quality and detail). 

To address this, the paper proposes HACP (Head-level Attribution for Counteractive Pruning), a unified, training-free framework that operates at inference time. The framework has two core innovations:

1. InfoSpectralScore: A novel, semantics-based attribution metric designed to identify both "hallucination-inducing" and "faithful" attention heads. This metric is derived from an eigen-decomposition of head-level output embeddings and is regularized by spectral variance and entropy penalties to capture semantic capacity.

2. Dynamic Counteractive Pruning: A new intervention strategy. Instead of just ablating (zeroing out) problematic heads, this method dynamically "suppresses" the output of hallucination-prone heads while simultaneously "reinforcing" the output of faithful heads during inference.

The authors conduct extensive experiments on multiple LVLMs (e.g., LLaVA, Shikra, Qwen) and benchmarks (CHAIR, POPE). The results demonstrate that HACP achieves a new state-of-the-art in hallucination mitigation, significantly reducing hallucination scores while, crucially, preserving or even improving semantic informativeness (F1 score) compared to prior methods.

### Strengths
1. The paper addresses a significant and high-impact weakness of LVLMs. The core idea of "functional conflicts" among attention heads is an insightful way to frame the hallucination problem, and the goal of breaking the trade-off between faithfulness and informativeness is a key challenge for the field.

2. The proposed method is well-motivated and novel. The InfoSpectralScore is a principled, semantics-based metric that goes beyond simpler attribution methods like KL divergence. Its construction from spectral decomposition, variance, and entropy (Eq. 7) is a solid methodological contribution. The Counteractive Pruning strategy is a clear improvement over conventional ablation. The idea of not just silencing bad heads but also amplifying good ones (Eq. 12) is intuitive and, as shown by the results, highly effective.

3. The experimental results are the paper's strongest point. The method is shown to be effective across multiple models and benchmarks. The key finding, highlighted in Tables 3-5, is that HACP breaks the faithfulness-informativeness trade-off. While competing methods like SPIN and MLIH achieve low hallucination scores at the expense of massive drops in F1, recall, and caption length, HACP reduces hallucinations while increasing the F1 score and preserving recall.

### Weaknesses
1. The paper is not clear about the practical computational overhead. 

- The attribution step (Algorithm 1) requires running inference on an "attribution set $D$" for every head in the target layers. This seems to be a very expensive pre-computation.

- More concerning is the "Task-Specific Automated Pruning Pipeline" (Algorithm 3), which suggests running a Bayesian optimization loop for $T$ iterations per-instance. It is unclear if this was used for the SOTA comparisons (Tables 3-5). If it was, this would represent a massive, per-sample computational cost not incurred by the baselines, making the comparison unfair.

- The per-token latency added by the dynamic pruning (Algorithm 2) is not quantified.

2. The method introduces several new hyperparameters: $\alpha$ and $\gamma$ for the InfoSpectralScore, $\mu$ and $\lambda$ for pruning, and the choice of target layers. The paper states that "Grid search for hyperparameters ($\mu$ ,$\lambda$) is conducted independently on each split". This suggests the method may be highly sensitive to these settings and would require a costly, task-specific tuning process to work, undermining the "plug-and-play" benefit.

3. The quality of the method seems highly dependent on the "attribution set $D$". Section 4.3 and Table 2 explicitly show that using a "hallucination-focused" attribution set is far more effective than a random one. This creates a practical chicken-and-egg problem: to fix hallucinations, one must first curate a dataset of inputs that are known to cause hallucinations.

4. The paper contains minor but confusing inconsistencies in its mathematical formulation.

- $\alpha$ Overload: The symbol $\alpha$ is used for two different purposes: first as a threshold (e.g., 0.9) to determine $k^{*}$ for the EigenScore, and second as a regularization weight for the SpectralVar term in Equation. This reuse of a key symbol is confusing.

- Entropy Formulation: There is an inconsistency in the "Spectral Entropy" formulation. Equation 6 correctly defines SpectralEntropy with a negative sign, consistent with Shannon entropy. However, Equation 7, which claims to add an "entropy penalty", adds the term $+\gamma \cdot [\sum p_i \log(p_i + \epsilon)]$. This is the negation of the defined SpectralEntropy term. While the intent (penalizing low-entropy/sparse distributions) is clear, the inconsistent naming and sign in the final equation is notationally imprecise.

### Questions
1. Regarding Computational Cost (Weakness #1):

- Was the "Automated Pruning Pipeline" (Algorithm 3) with its per-instance Bayesian optimization used to generate the SOTA comparison results in Tables 3-5?

- If yes, how is this a fair comparison to baselines? If no, how were the hyperparameters (which are noted to be grid-searched independently for each split 29) actually set for the SOTA comparison?

- What is the one-time, per-model setup cost (in hours/GPU) for the attribution step (Algorithm 1)?

- What is the per-token latency (in ms) added by the dynamic pruning mechanism (Algorithm 2) during generation?

2. Regarding Practicality (Weakness #2 & #3):

- Given the apparent hyperparameter sensitivity, how would a practitioner realistically set ($\mu$, $\lambda$) for a new model or task without a costly grid search or the per-instance optimization from Algorithm 3?

- How is the "hallucination-focused" attribution set created in a general setting? Does this not assume one has already solved the problem of identifying hallucination-prone inputs?

3. Regarding Methodology (Weakness #4):

- Can the authors clarify the reuse of the symbol $\alpha$?

- Can the authors correct the sign inconsistency between the definition of SpectralEntropy in Equation 6 and its use in Equation 7?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces HACP, a training-free framework for controlling object hallucinations in large vision–language models (LVLMs) at the attention-head level. It identifies that individual attention heads can have conflicting roles, some contributing to faithful grounding and others to hallucinations, and addresses the lack of fine-grained head-level intervention mechanisms. The authors propose InfoSpectralScore, a semantics-based attribution metric combining eigenvalue analysis with spectral variance and entropy regularization, to distinguish hallucination-prone from faithful heads. Using this attribution, they implement dynamic counteractive pruning that suppresses selected hallucination heads and reinforces faithful ones during inference, with an adaptive, task-specific pipeline.

### Strengths
1. The proposed approach operates in a training-free manner, thereby avoiding additional training cost and data requirements.
2. The work explicitly addresses the issue of functional conflicts among attention heads, which has been largely overlooked in prior studies.

### Weaknesses
1. In line 305, the authors state “We evaluate HACP on LLaVA-1.5 (7B, 13B), Shikra-7B, and Qwen2.5-VL-7B-Instruct.” However, I could not find corresponding experimental results for Qwen2.5-VL-7B-Instruct in the subsequent sections. It would be helpful to include these results to assess the method’s effectiveness on this more recent model.
2. The evaluation relies on a relatively limited set of datasets. It would strengthen the work to include additional hallucination benchmarks such as HallusionBench[1] or CRPE[2] for a more comprehensive assessment.
3. The paper lacks experiments on out-of-domain data. It would be valuable to understand whether the proposed method provides generalization benefits when applied to other datasets beyond those used in the current evaluation.
4. Although the method is claimed to offer interpretability, the presented evidence is rather limited. The attribution of functional roles to attention heads is mostly supported by quantitative metrics and a small number of qualitative cases, without more systematic visualizations or behavioral analyses to substantiate this claim.

[1] Guan T, Liu F, Wu X, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 14375-14385.

[2] Wang W, Ren Y, Luo H, et al. The all-seeing project v2: Towards general relation comprehension of the open world[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 471-490.

### Questions
As described in Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper targets the hallucination reduction problems in large vision-language models (LVLMs). It proposes to intervene the attention heads at a fine-grained level to mitigate LVLM hallucinations. The proposed strategy dynamically pruning or modifying the hallucination-prone attention heads, which is training-free. Experiments demonstrate improved performance in LVLM hallucination reduction.

### Strengths
- The paper proposes a mechanistic approach to control the VLM hallucination on the attention head level, through effective inference time intervention.

- Experimental results across three VLMs and two benchmarks show improved performance of the proposed approach.

### Weaknesses
- The paper writing could be improved. The novelty may also be limited given prior research in interpreting hallucination with attention heads and general mechanistic interpretability methodologies.

- Computational cost may be high. The identification of specific attention heads that are hallucination prone needs to be computed on a batch of images before inference, requiring specific head intervention.

- Different images may have different optimal head configurations. Instance level optimization on the attention head identification and intervention is time consuming, compromising the applicability of the proposed approach.

- The hyper-parameters are selected via grid search, which is also expensive considering the procedure of the approach.

- For experiments such as on CHAIR, there is no sensitivity/statistical significance analysis showing the randomness of the results, as we know the scores could vary with a large variance. The paper could benefit from testing on more recent benchmarks and models, such as InternVL, PaliGemma, Phi vision, Meta PLM, AMBER, MME, THRONE.

- The approach is image/benchmark specific, as the optimal attention heads may be different for different styles and sources of images. This impedes the practical usage of the approach.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
