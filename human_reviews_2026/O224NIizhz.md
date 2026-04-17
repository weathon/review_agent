# Flow Along the $K$-Amplitude for Generative Modeling

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 2

## Abstract
In this work, we propose K-Flow, a novel generative learning paradigm that flows along the $K$-amplitude domain, where $K$ is a scaling parameter that organizes projected coefficients (frequency bands), and amplitude refers to the norm of such coefficients. We instantiate K-Flow with three concrete $K$-amplitude transformations: Fourier transformation, Wavelet transformation, and PCA. By incorporating the $K$-amplitude transformations, K-Flow enables flow matching across the scaling parameter as time. We discuss six properties of K-Flow, covering its theoretical foundations, energy and temporal dynamics, and practical applications. Specifically, from the perspective of practical usage, K-Flow allows for steerable generation by controlling the information at different scales. To demonstrate the effectiveness of K-Flow, we conduct experiments on both unconditional and conditional image generation tasks, showing that K-Flow achieves competitive performance. Furthermore, we perform three ablation studies to illustrate how K-Flow leverages the scaling parameter for controlled image generation. Additional results, including scientific applications, are also provided.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes K-Flow, a novel generative modeling paradigm that conducts flow matching in the K-amplitude domain—where the scaling parameter k organizes projected coefficients (e.g., frequency bands) and "amplitude" refers to the norm of these coefficients. By instantiating K-Flow with three classic transformations (Fourier, Wavelet, PCA), the work addresses a key limitation of conventional flow matching (FM) models: the lack of explicit utilization of natural data’s frequency hierarchy. Through extensive experiments on unconditional/conditional image generation, controllable generation, and image restoration, the paper demonstrates K-Flow’s competitive performance and unique steerability. The work’s theoretical grounding (six core properties of K-Flow), architectural flexibility (backbone-agnostic design), and practical value (training-free restoration, unsupervised frequency editing) make it a meaningful contribution to generative modeling research, well-aligned with ICLR’s focus on innovative and rigorous methods.2. Strengths

### Strengths
Innovative Theoretical Framework with Clear Motivations:K-Flow fills a critical gap in existing FM literature by leveraging the inherent frequency structure of natural data. Unlike conventional FM models—whose frequency progression is unstructured and unquantified—K-Flow formalizes the K-amplitude space, where k systematically organizes frequency bands and amplitude reflects signal energy. The six properties of K-Flow (from theoretical foundations like scale-energy alignment to practical benefits like explicit steerability) provide a rigorous framework for understanding and extending the paradigm, distinguishing it from ad-hoc frequency-aware modifications.

### Weaknesses
Lack of Direct Comparison with Frequency-Aware Diffusion Models
While K-Flow is positioned as a flow-matching innovation, the paper does not explicitly compare it to frequency-aware diffusion models. Adding such comparisons would clarify K-Flow’s advantages in terms of both performance and computational efficiency, especially in scenarios where diffusion and flow-matching paradigms overlap.

### Questions
Regarding the observation that MLP heads outperform Transformer heads in text adherence, could you provide a more in-depth analysis? For example, from the perspective of token-level alignment between text prompts and generated images, or exploring whether the attention mechanism in Transformer heads interferes with text-image alignment?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel generative modeling paradigm called **K-Flow**.  The core idea is to replace the traditional **time variable** \( t \) in *Flow Matching* with a **scale parameter** \( k \).  This enables the model to evolve from noise to data **along a multi-scale trajectory** (e.g., from low to high frequencies).  The authors instantiate K-Flow with three types of **K-amplitude transformations**: **Fourier Transform**, **Wavelet Transform**, and **PCA**.  They claim that this approach provides a **more natural multi-scale modeling framework** and allows for **finer-grained controllable generation**.

### Strengths
1. **Novel framework:** Replacing the “time” variable \( t \) in flow models with a “scale parameter” \( k \) is a novel and conceptually elegant idea.  
2. **Generality:** The framework is general and can be combined with various linear transformations such as Fourier, Wavelet, and PCA.  
3. **(Limited) Qualitative results:** Although the CDR metric has certain flaws, the visual robustness of K-Flow under the “Drop” condition in **Figure 4**, as well as the controllable generation results in **Figure 5**, are qualitatively interesting.

### Weaknesses
1. **Logical contradiction in the CDR metric:**  
   In Sec. 4.2, the CDR metric is used to demonstrate controllability, but its mathematical definition in Appendix F.3 directly contradicts both the textual explanation in the main paper (“higher CDR indicates performance degradation”) and the visual evidence in **Figure 4**.  
   This inconsistency invalidates the quantitative conclusions of the ablation study. The authors must correct the metric (most likely the formula is inverted) and reanalyze the results.

2. **Unexplained performance inconsistency:**  
   The paper must address why the **K-Flow (Wavelet)** variant performs best on **Table 1 (CelebA-HQ)** but performs poorly on **Table 2 (ImageNet)**—significantly worse than the LFM baseline.  
   This raises serious concerns about the method’s robustness and generalization capability.

3. **Hyperparameter fragility:**  
   The authors must explain why the number of K-amplitude frequency bands (e.g., 2 vs. 3) leads to completely opposite performance trends across datasets (Appendix F.2 vs. F.6).  
   This indicates that the method is fragile and sensitive to hyperparameter tuning, undermining the claim of “natural” multi-scale modeling.

### Questions
1. **Regarding CDR:** The CDR formula in Appendix F.3 ($FID_{bef} / FID_{aft}$) seems to contradict the conclusions in the main text and the visual evidence in Figure 4. According to this formula, LFM's CDR = 3.25 would imply that its $FID_{Drop}$ quality is much better than $FID_{Undrop}$. Could this formula be a typo (e.g., with the numerator and denominator swapped)?

2. **Regarding performance:** Why does the K-Flow (Wavelet) variant perform the best in Table 1 but fall far behind the LFM baseline in Table 2? Does this suggest that wavelet transforms may not be suitable for complex class-conditional tasks like ImageNet?

3. **Regarding hyperparameters:** Why does increasing the number of K-amplitude bands (from 2 to 3) improve performance on the LSUN dataset (App F.2) but decrease performance on the CelebA-HQ dataset (App F.6)? Does this indicate that K-Flow is highly sensitive to this hyperparameter and that its optimal setting depends on the dataset?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a generative model based on flow matching that works in the K-amplitude space. The model leverages three K-amplitude transformations, Fourier, Wavelet, and PCA, to construct its generative process. Experimental results show that the proposed method achieves performance comparable to the baselines on both unconditional and conditional image generation on natural images and the molecular assembly dataset. Furthermore, the authors show that their approach enables multi-scale control over the generated information.

### Strengths
The main idea sounds novel and is supported by compelling experiments. The paper is well-written and easy to understand.

### Weaknesses
As discussed in your Related Works section, the following papers share significant similarities with your approach. I recommend including them as baselines for experimental comparison:

[1] Tian, Keyu, et al. “Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.” Best Paper, NeurIPS 2024.

[2] Ren, Sucheng, et al. “FlowAR: Scale-Wise Autoregressive Image Generation Meets Flow Matching.” ICML 2025.

Including these two baselines in your comparison would substantially improve your empirical evaluation. I would be happy to potentially raise my score.

### Questions
Please consider clarifying the following point:
Should there also be a summation in line 128, similar to the one in line 123?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Disclaimer: I found the paper hard to understand, I did my best to summarize what I understood.

This paper propose to perform a change of basis (Fourier, Wavelet, PCA) on the data, cluster the new representation into K ordered sub-basis and to do diffusion for each sub-basis in sequence (in the Fourier case we can think of it as frequency bands).
To clarify, what is typically referred as time $t$ in diffusion is now a range $t'\in[0,K]$ where $k=\lfloor t'\rfloor$ and $t=t'-k$.
In this new setting the new definition of time interleaves the classical time definition with ordered sub-basis.

The claimed benefit of such a proposal is to allow for more steerable generation, where the steerability is determined by the choice of change of basis and the sub-basis on which to do diffusion.
The evaluation is performed on generative tasks and an ablation demonstrates steerability by sampling only in some of the sub-basis or removing the label during inferences.

### Strengths
1. The idea seems novel and interesting.

### Weaknesses
1. The presentation/writing of the paper is very hard to follow, I had to read 3 times to begin understanding it (or so I assume). In particular, it's overloaded with notation complexities that could probably be simplified to make it more accessible.
2. The results are not particularly striking for unconditional and conditional generation, matching existing SotA but not apparently showing clear gains.
3. The choice of CelebA is questionable for unconditional generation as it is quite unimodal in the first place, unlike ImageNet.
4. The resolution used for ImageNet is not mentioned in the main paper, but I found near the end of the appendix that it must be 256x256.
5. The PCA variant of K-Flow is missing from Table 2.
6. The paper lacks a discussion of why the method does not perform better on generation tasks. 
7. In terms of steerability, the results do not seem particularly useful. In one case the class label is dropped when $k$ a certain value (30%) and is still able to generate good samples (but since the label is always available I fail to see the relevance of this experiment). In another case, some sub-basis are frozen while some other basis are sampled from resulting in changing either low-frequency or high-frequency details. It works but I don't understand under what circumstances one would want to do this. It gives the feeling to be a solution to a non-problem or maybe better examples of steerability are needed to inspire readers.
8. Line 333, you mention "Date-dependent PCA", do you mean "Data-dependent"?

### Questions
1. The gains obtained from K-Flow appear marginal at best. Is it because the classical diffusion noise is already implicitly doing a K-Flow and therefore no gains are achieved from expliciting doing a K-Flow?
2. On the steerability front, it would be interesting to see differences k-band steering between Fourier, Wavelet and PCA. I assume steerability is heavily influenced by the choice of basis. Did you do such experiments?
3. Is the neural network actually conditioned on $(k,t)$ as a tuple or $t'$, or are there $k$ different networks?

### Soundness
3

### Presentation
1

### Contribution
3
