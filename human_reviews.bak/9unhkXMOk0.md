# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 1, 3, 3

## Abstract
Time series representations learned from high-dimensional data, often referred to as ”disentanglement” are generally expected to be more robust and better at generalizing to new and potentially out-of-distribution (OOD) scenarios. Yet, this is not always the case, as variations in unseen data or prior assumptions may insufficiently constrain the posterior probability distribution, leading to an unstable model and non disentangled representations, which in turn lessens generalization and prediction accuracy. While identifiability and disentangled representations for time series are often said to be beneficial for generalizing downstream tasks, the current empirical and theoretical understanding remains limited. In this work, we provide results on identifiability that guarantee complete disentangled representations via Contrastive Sparsity-inducing Learning, which improves generalization and interpretability. Motivated by this result, we propose the TimeCSL framework to learn a disentangled representation that generalizes and maintains compositionality. We conduct a large-scale study on time series source separation, investigating whether sufficiently disentangled representations enhance the ability to generalize to OOD downstream tasks. Our results show that sufficient identifiability in time series representations leads to improved performance under shifted distributions. Our code is available at https://anonymous.4open.science/r/TimeCSL-4320.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes the identification guarantees of learning representation for time series data. Specifically, this paper proposed a new method, called Contrastive Sparsity-inducing, to leverage the assumption of sparsity in data structure. Then, a TimeCSL framework is proposed to learn a disentangled representation with the constraint of sparsity. Extensive experiments are conducted to evaluate the proposed method.

### Strengths
1) The codes are released for better reproduction.  

2) The experiments on public datasets and synthetic datasets are conducted.

3) Figure 1 proposes a good point to show the motivations.

### Weaknesses
I have three big concerns about this paper.

First, though some definitions and assumptions were listed, I didn't find the strict theorem and corresponding proof about the identification results. Some related work is referred to, such as  (Lachapelle et al., 2022). However, it is still not clear how this method can directly help prove the identification.

Second, lots of existing ICA-based work on identifiable disentangled time series representation is missing to compare in evaluation, like TCL, PCL, TDRL, and so on.   Discuss with them can further help highlight the contribution. Besides, it is better to show the difference between the key assumption 4.1 and assumption 6 in Sparse ICA (Ignavier Ng, 2023, Neurips).

Third, the results seem not reliable. The performance of S3VAE+HFS, C-DSVAE + HFS, SparseVAE, and TimeCSL are totally the same on two different datasets. The authenticity of experimental data is doubtful.  Due to time limitations, the code is not checked. Will check it before the rebuttal phase.

### Questions
Please refer to the weaknesses section.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The work discusses identifiability that guarantees disentangled representations via Contrastive Sparsity-inducing. Following this, a new framework called TimeCSL is proposed to learn a generalised disentangled representation that maintains compositionality. The results show the efficacy of the proposed method compared to various baselines.

### Strengths
1. The paper is easy to follow. 
2. The evaluation in Table 1 is very comprehensive with many cases and baselines considered.
3. All pre-trained models are accessible along with guidelines.

### Weaknesses
1. There are a few typos and broken references in the paper for e.g. guarantee (misspelled in Line 77), broken reference in line 409, figure reference broken in line 502. I will advise doing a spell check etc.
2. Could the authors discuss using Normalising Flows (or Diffusion Models) instead of a VAE in their framework? Flows, for example, are bijective transformations. Some results for identifiability with Flows exist for image data [1]; maybe they can be directly applied to temporal data.
3. Could the authors provide justification or empirical evidence for the realism of Assumption 4.1 in practical time series scenarios? It will be interesting to see how sensitive the method's performance is to violations of this assumption.

[1] DISENTANGLEMENT BY NONLINEAR ICA WITH GENERAL INCOMPRESSIBLE-FLOW NETWORKS (GIN) by Peter Sorrenson, Carsten Rother, Ullrich Kothe

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper "IDENTIFIABILITY GUARANTEES IN TIME SERIES REPRESENTATION VIA CONTRASTIVE SPARSITY-INDUCING" proposes Contrastive Sparsity-Inducing Learning to help improve model generalization and interpretability. A slot-wise identifiability has been proved in the theorem part. Additionally, it implements the TimeCSL framework, which enhances performance across various existing models. The authors provided lots of experiments to support their conclusions.

### Strengths
- Question is interesting. 
- The motivation is clear. 
- The figure is intuitive.

### Weaknesses
- Section 2: The setting is not clear. 
    - The author claims that $x$ is an additive mixture of y from $y _ 1$ to $y _ n$ and noise. However, in the next paragraph, the dataset is defined as $D= ${$ x _ i, y _ i $}$  _ {i=1} ^ N$, which indicates that $y$ is the label. This is confusing.
    - If in the first paragraph, y is the latent variable. It is also confusing about the mixing function. What does $n$ means in line 119. Does it mean that at each time step, y is a $n$ dimensional vector?
    - If so, in line 117, when $C=1$, does it infer that the mixing function is not injective?
- Equation 2.1: The reconstructed in reconstrcution loss should be $x$ rather than $y$.
- Line 144: What is gθ♯p(z)? What is $M^1 _ + (X)$? Definition is needed.
- Line 162: What is 'recouvering'? Is it 'recovering'? 
- Line 162: y is not given in Eq 2.2 (generating process) and emerge here sharply, it is not clear what this sentence mean here.
- Definition 2.2: Usually we call it identifiability rather than disentanglement.
- Line 257: unfinished setence, if what?
- Section 4.1: Why call entries i with $\frac{|\mu _ i|}{\sigma _ i}>1$ importatant? Some discussion is needed.
- Line 262: Usually the term component-wise identifiability is used.
- Assumption 4.1: assumption is not aligned with Equation 4.1. Text says two samples x and x', while Equation says all $i$ that does not include k, which infers much more observables.
- Experiments are not reliable
    - Table 1: The performances of S3VAE+HFS, C-DSVAE+HFS, SparseVAE, TimeCSL are EXACTLY the same, for REFIT and REDD. As far as i know, the two dataset is not that same.
    - Table 1: why is R2 larger than 1 (in the table it shows "RMIG", I am not sure what it is)
    - line 404: split 70/20/20, which sums up to more than 100
    - Table 2: why "higher is worse" for MCC？Why the larger R2 is, the smaller MCC is, which is strange.

### Questions
I am interested in the detailed implementation of the model. At the same time, the code seems not runnable. It will be helpful if a more detailed document can be added to the code.
- Code is not runnable. 
    - There is no file named 'main.py'.
    - The code looks like for image data rather than sequence data.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper aimed at ensuring identifiability in time series representation learning by leveraging contrastive sparsity-inducing mechanisms. The authors address challenges in disentangling time series data by proposing a structured, sparsity-enforcing learning method that improves interpretability, robustness, and generalization, especially in source separation tasks.

### Strengths
1. **Theoretical Contributions**: The proposed framework is supported by theoretical insights that demonstrate the efficacy of contrastive sparsity in ensuring identifiable representations.

2. **Thorough Experimental Validation**: The paper offers a detailed experimental analysis, evaluating the proposed method on multiple datasets and providing insights into its performance under various settings.

### Weaknesses
1. **Unclear Problem Definition**: While the paper addresses time series representation learning, the data generation process presented is not inherently temporal (line 157). Furthermore, although the authors claim that their method accommodates "statistically dependent latent factors" (line 78), the source separation example provided involves independent sources, and dependent latent relationships are not clearly illustrated within the problem set. Without a well-defined problem scope and explicit assumptions, comparing the limitations with prior work becomes challenging.

2. **Unclear Theorem Proof**: A rigorous mathematical proof of identifiability, grounded in a clearly defined problem setting and set of assumptions, would strengthen the paper. Explicit derivations would provide the necessary foundation for understanding the theoretical claims presented.

3. ** Incomplete Manuscript:** Some information is missing in the main manuscript (lines 409, 502), and sections of the appendix (A3, A4, and C) appear unfinished or repetitive. This lack of completeness makes it difficult to thoroughly verify the experimental setup and interpret results accurately.

### Questions
1. Figure 1 is visually appealing but lacks sufficient detail to fully understand the concepts it illustrates. Could the authors provide a more comprehensive explanation of each component and its role within the model?
2. In line 76, the authors state that "sparsity alone is insufficient to ensure reliable identifiability, and thus, generalizability." Could you expand on this claim? What specific limitations of sparsity do previous studies identify in the context of identifiability?
3. Could the authors clarify the relationship between Y and Z? Is Y intended to represent the ground-truth sources (line 118) and Z the estimated latent variables (line 123)?
4. What data are provided as inputs to the model? In line 376, the tuples (x, y, x', y', i) are sampled. Are all these elements necessary in the model?
5. Since it is affine-wise identifiability, why use R2 instead of MCC as the evaluation metric?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a method (TimeCSL) to obtain disentangled representations from high-dimensional time series data through Contrastive Sparsity-inducing Learning. They use Partial Selective Pairing as the contrastive objective, and train a modified VAE to obtain disentangled representations. The authors argue how this formulation improves the compositional generalization of the obtained representations. Experimentally, the paper shows the effectiveness of their formulation for the separation task.

### Strengths
1. The authors release a substantive set of pretrained baselines that could be useful for future research.
2. The showcased experimental results indicate that the proposed method TimeCSL outperforms the considered baselines.

### Weaknesses
## 1. Presentation and organization

The paper's key findings could be enhanced through a more structured organization and clearer presentation of the material. Apart from numerous grammatical/spelling errors (See Questions for a non-exhaustive list) that make it difficult to read the paper, there are several major issues due to the presentation.

### a. Unclear problem setting

I am very confused about the problem setting due to various mistakes/omissions in the presented notations. 

a(i): What are the "unobserved States sources" $s_1, ..., s_5$ in Figure 1? There is no mention of these sources in the main text or appendix, however, they seem to be an important part of how the problem is being modeled.

a(ii): Lines 120-140 describe the details of a "VAE", however, the classic VAE reconstructs the input $x$ (hence, it is an *auto*-encoder). However, Equation (2.1) in the paper talks about the reconstruction of $y_i$, given $z$ and $x_i$. What is $y_i$ in this context? The notation $y_i$ was used to indicate the sources in line 118, but this does not make sense in the context of Equation (2.1). Similarly, what is $\mathcal{Y}$?

a(iii): Lines 170-173 are unclear. "$z$ must be generated by $g_\theta$", but $g_\theta$ does not generate $z$, it decodes $z$. The grammar errors in line 172-173 obfuscate the meaning entirely. 
 
a(iv): The different "views" $x$ and $x'$ considered in the contrastive formulation are not explained. Are they different time-series samples?

Overall, it is not clear if this problem statement is the same as the source separation problem tackled in nonlinear ICA or not. If not, how?

### b. Organization of Section 4

b(i): What is $z$ vs $\hat{z}$? This is not adequately explained in Section 4. Similarly, in lines 263, what are $f_\phi$ and $\hat{f}_\phi$? 

b(ii): Assumption 4.1 has missing details that make it hard to understand. What is $\mathcal{I}$? Does Equation (4.1) need to hold for any pair $x$, $x'$ or some pair? These details are unclear/missing.

b(iii): Lines 288-289: "according to Assumption 4.1, the sparsity-inducing nature ... existence of a source" - I don't see how this statement follows. What does the sparsity inducing nature mean in this context?

b(iv): It is unclear what the "claimed" theoretical contribution is. It would be beneficial to write a formal mathematical statement in the form of a theorem block and provide a proof. 

b(v): The discussion about compositional generalization is difficult to follow. The first equation in Equation (4.6) doesn't quite make sense, since $g_\theta : \mathcal{X} \rightarrow \mathcal{Z}$, i.e. the image space of $g_\theta$ does not align with the domain space of $\hat{g}\_{\theta}$. Similarly, in line 356, I'm unsure how $\hat{g_\theta}(\hat{z}) \approx \hat{z}$.

b(vi): The optimization objective/algorithm is unclear. In equation (4.8), what are $z, z', \hat{z}, \hat{z}'$? Note that the latent variables $z$ appear in the objective of the VAE loss $\mathcal{L}_\text{VAE}$, and so without additional context, it is unclear how they are computed in the other terms. Additionally, what are the indices $\mathbf{i}$ and how are they calculated?


## 2. Unsubstantiated/Wrong Claims

There are several occasions in the paper where claims are unsubstantiated or (in some cases) wrong. 

2(a): Lines 49-50: "The risk of ill-defined may lead to unstable and unreliable model outputs, where minor perturbations in data or hyperparameters can yield significantly different results upon retraining." - is this an observation by the authors? If so, I don't see the evidence presented in the paper. If it appears in prior work, then there should be a citation.

2(b):  Line 203-204: "This is the first identifiability study in real world of time series representation" - This seems like too strong a claim. The authors go on to cite papers that, in fact, tackle identifiability of time-series representations with real-world applications. Similarly, lines 195-196: "this work is the first to address identifiability and generalization in time series representations for separating sources in real scenarios" is wrong, see [1] for a method that does source separation with real-world applications. 

2(c): Lines 209-210: "we place no assumptions on $p(z)$", however, in line 159, the authors assumed a particular form for $p(z)$, i.e. a GMM. This should be appropriately qualified in the text.

## 3. Missing Details

Several important details about the method/experimental settings are missing.

3(a): The details about the implementation of the loss-terms/neural network architectures used are not present.

3(b): Several subsections in the Appendix are empty (A.4, C).

3(c): The definitions of DCI/RMIG are not mentioned in the main text, but presented in the tables.


## References:
[1] Hyvärinen, Aapo, Ilyes Khemakhem, and Hiroshi Morioka. "Nonlinear independent component analysis for principled disentanglement in unsupervised deep learning." Patterns 4.10 (2023).

### Questions
1. What is a "latent slot"? This seems to be non-standard terminology. Do you mean latent dimension?

2. Line 159: "$z$ follows a Gaussian mixture model" - Can you provide an equation to show what the mixture components are, and an intuition as to why this assumption is useful?

3. In Line 144-145, what is $\mathcal{M}_+^1(\mathcal{X})$?

4. Line 257: "non-zero components of $\hat{z}$ and $\hat{z}'$", but they are technically not non-zero, rather they have a small magnitude as defined by the condition on the ratio of mean and variance. 

5. The setting considered in Line 119 seems to indicate that the observed signal is a linear combination of unobserved sources. Why can't we use linear ICA? Would it make sense to include it as a baseline?

6. Comments on Figure 1.

6.a. The figure is not mentioned anywhere in the text

6.b. There are 4 OFF/ON views, but 5 state variables. 

6.c. The numbering of the slots is inconsistent (1.1...1.5) and (1.2, ... 5.2). 

6.d. What is "stop-gradient"?

6.e. The figure is cut off under the second view.

6.f. $x'$ is used in the caption, but $\tilde{x}$ is used in the figure.

7. Grammar/Spelling Errors

7.a. Line 43: weaker->weakly

7.b. Line 49: "risk of ill-defined <missing word?> may" 

7.c. Line 77: "there <is> a need". "garantee" -> guarantee

7.d. Lines 94-95: "we propose a .... learning out-of-distribution data" -> incorrect grammar

7.e. Lines 104-105: $d$ is used in some places, $d_\mathcal{Z}$ in others.

7.f. Line 134: missing parenthesis around $x_i, y_i$.

7.g. Line 162: recouvering -> recovering

7.h. Line 173: "We give in Section 4.3. intuition and theoretical behind" -> incorrect grammar.

7.i. Line 182: extra )

7.j. Line 193: extra }

7.k. Line 195: "This work best of our knowledge.." -> incorrect grammar

7.l. Line 203: "As this is the first identifiability study in real world of time series representations" -> incorrect grammar

7.m. Line 257: "indicating that if <missing fragment>, then"...

7.n. Line 317: $k$ and $p$ were defined but not used correctly

7.o. Line 374: "leanred" -> learned

7.p. Line 377: "we show a how for" -> incorrect grammar

7.q. Line 404: "70/20/20 train/val/test split" does not sum to 100

7.r. Line 409: Missing reference (?)

7.s. Line 502: Missing reference (??)

### Soundness
1

### Presentation
1

### Contribution
2
