# Conditional Flow Matching for Conformal Regression

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
This paper introduces Conditional flow Matching for conformal Regression (CMR), a novel framework that synergizes simulation-free conditional flow matching with conformal prediction to generate reliable and efficient prediction intervals. Unlike traditional methods that rely on quantile regression or fixed histograms, CMR leverages Continuous Normalizing Flows (CNFs) trained via Conditional Flow Matching (CFM) to accurately model complex, multimodal conditional distributions. To ensure finite-sample coverage guarantees, we introduce a novel nonconformity score defined as the minimum number of generated samples required for the shortest interval to encompass the true outcome. This mechanism allows CMR to dynamically adjust interval widths based on the learned probability density. Extensive experiments on simulated and real-world datasets demonstrate that CMR consistently produces narrower prediction intervals while maintaining the required marginal coverage and achieving superior tail coverage compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript proposes Conditional Flow Matching for Conformal Regression (CMR), which first learns a conditional flow model via the conditional flow-matching objective and then constructs conformal prediction intervals by selecting the shortest continuous interval among samples generated from the learned conditional distribution.  Experiments and ablations span 1D/2D simulations and 12 real datasets, reporting Coverage, Interval Size, WSC, and TCR, with visual comparisons that highlight the method’s effectiveness.

### Strengths
1. The overall approach is simple (in a good way), principled, and clearly presented.
2. The paper offers a comprehensive evaluation across multiple datasets with diverse metrics.
3. Figure 1 is illustrative and clearly highlights the key distinction between the proposed method and CHR.
4. The methodological transparency is commendable; the building blocks and concise appendix code substantially aid understanding.

### Weaknesses
1. The draft needs a clearer background on CHR -- what it is, its limitations, and why CMR addresses them. At present, the main text references differences from CHR without first defining that baseline. It would also help to expand the discussion of conditional flow matching (motivation, objective, and practical choices). 
2. Please fix notation inconsistencies -- for example, the source distribution shifts from $p_0(y)$ to $p_0(z)$, and $\mu(y\mid x)$ is introduced in Eq. 5 but immediately becomes $u_t(y\mid x)$ in the very next line. Finally, verify dataset citations and include the necessary details to ensure correctness and reproducibility.
3. The contributions naturally decompose into (1) CFM for quantile regression and (2) CMR for calibration. To substantiate these, the experiments should separately demonstrate: CFM vs. existing quantile regressors (NN, RF); CMR vs. other conformal methods; and the combined method vs. existing combinations. Table 1 adequately supports the third comparison, but Table 2 tries to cover the first two at once and remains incomplete. For instance, CFM with NN/RF baselines are missing. Consider disentangling these ablations so each contribution is evaluated cleanly.
3. Relative to CHR (with NN or RF), the proposed method appears more computationally demanding. While this may not be a major drawback, please add a computational analysis -- including Big-O complexity and empirical runtimes -- for training, calibration, and inference. This will provide a more complete assessment of performance and help readers understand the practical trade-offs.

### Questions
1. The overall approach is simple (in a good way), principled, and clearly presented.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The method first trains a conditional generative model for the outcome using flow matching. After training, the model is used to generate multiple samples of the outcome for each input x. These samples are then used to construct a set of candidate prediction intervals. For each calibration point, the conformity score is defined as the length of the shortest interval among these candidates that contains the true observed outcome. The resulting conformity scores are then used within the standard conformal prediction framework to obtain prediction intervals with marginal coverage guarantees. The method is evaluated in simulation studies, where it appears to produce narrower (more efficient) intervals than existing approaches.

### Strengths
The use of flow matching and a simulation-based conformity score based on the minimum distance of generated intervals is novel. By targeting the interval width, it seems reasonable that this conformity score would be more efficient (less wide) than other approaches. Simulations appear to show this.


The idea of using the shortest interval formed from model samples (rather than quantile-based intervals) is novel and could yield tighter prediction intervals in practice.


Theoretical marginal coverage guarantee is provided. The method is model-agnostic post-training: any generative model capable of conditional sampling could in principle be plugged into the calibration step. The authors propose using normalized flow matching but this doesnt appear necessary for the theory.

### Weaknesses
The writing and presentation could be improved significantly. Key components of the method—such as flow matching, the calibration procedure, and the role of several variables—are described with imprecise or undefined notation. The explanation of flow matching, in particular, is difficult to follow, and the reader is asked to accept it as a black-box procedure for conditional density estimation and sampling.

The key contribution of the paper---the calibration algorithm---is poorly described. Several indices are introduced ($j, k, s, m$), but it is unclear which quantities are fixed and which are being optimized or iterated over. For example, the algorithm selects the shortest interval containing at least $s$ samples, yet $s$ is never defined. Without a clear specification of these components, it is difficult to evaluate the method,


There is no theoretical justification or conceptual intuition given for why the proposed conformity score should yield more efficient (narrower) intervals than standard conformal methods. The paper argues for improved efficiency based on empirical results, but does not explain the mechanism or provide supporting analysis.

The proposed conformal prediction method only requires a conditional generative model for the outcome, yet the paper devotes substantial attention to normalizing flow matching. It is unclear why this modeling choice is emphasized, since the method does not introduce any new developments in flow matching and the theory does not appear to rely on this specific class of models. Flow matching seems to be just one possible generative model that could be plugged into the procedure, rather than a core contribution of the work.

### Questions
1. In equation 5, what is mu(Y|X)?
There is no u_t in equation 5, yet is mentioned directly below it.
What is p_t?

2. Am I correct that Equation 6 means that y is normally distributed given x and z_0 with mean y_1 + (1-t)z_0 and variance sigma^2? This is confusing as y_1 is itself a random variable drawn conditional on x. So does this not specify a distribution for y | x, z_0, y_1? Which then implies one for y|x, z_0 after marginalizing out y_1?


3. What is CHR? The abbreviation does not appear to be defined.


4. The calibration procedure (the main contribution of the paper) in Section 3.3 is not clearly specified. In particular, the quantity \(s\) in Step 3 is never defined: it is unclear whether \(s\) is fixed, varies with the desired coverage level, or is meant to be tuned or iterated over. The text suggests that $s$ determines how many sampled predictions must lie inside the interval, but this is never stated formally, and no guidance is given for how $s$ should be chosen. Algorithm~1 does not resolve this ambiguity, as $s$ does not appear as an input, output, or tunable parameter. As written, the calibration step is not reproducible: the paper does not explain how $s$ is selected and how it relates to the target miscoverage level $\alpha$. What is s in the simulations?

5. Similarly, what is j in  Section 3.3 I and Section 3.4? What is the minimization performed over in equation (13)? Is j and k varying?  Are the intervals minimized over constrained to contain s points?  

6. Is the proposed method restricted to conditional flow matching? or would any generative model work.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposed using a type of CNF called the condtional flow matching (CFM) to learn the conditional distribution of a one-dimensaional output y given multi-dimensional feature x. Besides using this tool of CFM in the conformal prediction framework, their key novelty is to find the shortest interval based on a sample of y|x from the learned model. This new idea of forming short intervals is named "shortest interval conformal", which is similar to the existing CHR method, but with slightly different execution. 

(Note that I am not sure if by CNF they meant CONDITIONAL normalizing flow or CONTINUOUS normalizing flow, because of the inconsistency between the abstract and the main section that introduced the backbones of their method.)

Some empirical examples where used to compare the new method to existing ones.

### Strengths
The topic of conformal prediciton is welcoming. 

The use of CFM is an interesting idea. Though using this powerful tool only on a one-dimensional output y seems a bit of an overkill.

### Weaknesses
0.0 The abstract mentions “conditional normalizing flow” twice, while Section 2.2 refers to a “continuous normalizing flow.” This inconsistency is quite confusing.

0. The focus on using connected intervals and finding the shortest one may put too much attention on a not too important part of conformal prediction. Also, the examples and illustrations seem either unnatural (e.g. Figure 1, a clearly bimodal density of y) or not clearly explained, so the conclusions about the benefits of CMR are not very convincing.

1. The writing can be much improved. Right now, the descriptions are redundant in some places and lack information in others. Below are a few examples.

1.1 sec 3.1 is not clear enough. It should be written such that a reader with decent machine learning background can understand the idea without consulting the original Conditional flow matching paper. E.g., in equation (6), it's not clear why x and z_0 where conditioned upon in symbols, but not y_1, and please indicate t is the (time) index of the flow, and do you expect t to take on discrete or continuous values and in what range? For another example, it was not self-evident what a "vector field" means in this context.

1.2 sec 3.3 step 3 description can be improved. It would be cleaner to give "k-j+1" its own symbol, say s, and say, for 2<=s<=m, consider intervals with end points y^(j) and y^(j+s-1) etc. I think steps 3 and 4 are trying to describe a rather simple and clear procedure to find the shortest interval that covers the true y and record the "Length" of that interval, and the steps can be made shorter and easier to grasp.

Minor but hurts clarity: First paragragh of sec 4.2 Real data. What are the letter labelled data? What does it mean to "rescale the response by the mean absolute value"? What does it ment that the "split has been validated" in a cited paper? (BTW, in Latex, `` '' specifies a proper pair of quotation marks.)

Sec 4.3. Consider including a table listing all the names with short description of each. Here, the logic/terminology of the first two paragraphs confuses me:
I thought CFM is the conformal part of CMR, what does it mean to "demo how CFM and CMR joinly enhance efficiency...". 

2. There are places where the statement lacks rigorosity:

2.1 Theorem 1. It was not clearly stated what data was used to train the Conditional flow in CFM, and if that data is conditioned upon when stating the coverage property.

2.2 When stating "key difference from CHR", note that CHR was not cited or defined until much later, and the author claimed "CMR ... a method that theoretically yields smaller intervals compared CHR". Given the intervals are random, more careful (probabilistic) statement about length comparison are likely needed. And I don't see this statement offically proved in the paper, including Appendix A.3.

Figure 1, this example of the density of y is a rather extreme case with bimodal distribution, where connected interval is clearly not the go-to-choice for prediction region. This looks like a very artificial example to try to find an advantage of CMR over CHR. (minor suggestion for the caption: better say "horizonal axis" instead of "x-axis" since x represents something else.)

### Questions
L48 What does (3) "... handling data from multiple distributions" mean?

L51 "However, these methods often struggle ..." Do you mean all the methods mentioned in this paragraph, or just those from (3)? This is an overly general and wide ciriticism, hence not the most informative and can be hard to justify.

Equation (5). What is \mu(y|x) and where did u_t(y|x) appear in the equation? Is this a typo such that \mu is the same as u?
As someone who knows about normalizing flow and ordinary differential equations, but not the CFM method for continuous NF before, I am not able to understand the details of Equation (6) without consulting the original paper. Could the introduction to CFM in this paper be more self-contained?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the usage of generative models for conformal prediction. The idea is to generate conditional samples from the fitted generation model and construct conformal score based on this sample by considering intervals of minimum length covering the observation. The concrete implementation is done via conditional normalizing flows trained with flow matching. The authors show the marginal validity of the resulting method and perform extensive experimental evaluation.

### Strengths
- General methodology is sound

- Numerical results show good performance of the method (with some caveats; see below)

### Weaknesses
- Usage of normalizing flows in conformal prediction framework is not new. The work [1] already considered the direct usage of normalizing flows, though via explicit usage of density (not sampling). Conditional generative models were used in the work [2], though not precisely normalizing flows. 

- The authors should better explain the difference with CHR method: CHR is not properly described in the paper apart from the illustrative example (which doesn't give details). Also, based on Table 2, CHR combined with CFM works mostly better than the proposed CMR method. It seems that the benefits of the method could be better explained and motivated.

- Theoretical results are standard (as any marginal validity  result for split conformal prediction).

- Experimental results deserve better visualization (tables are way too large to be informative).


Literature
[1] Colombo, N. (2024). Normalizing flows for conformal regression. arXiv preprint arXiv:2406.03346.

[2] Wang, Z., Gao, R., Yin, M., Zhou, M., & Blei, D. (2023, April). Probabilistic Conformal Prediction Using Conditional Random Samples. In International Conference on Artificial Intelligence and Statistics (pp. 8814-8836). PMLR.

### Questions
1. Can you explain difference between CMR and CHR in more detail?

2. Can you tell what novelty do you see on the usage of normalizing flows?

3. Can you check references in Section 4.2 that seem to be broken/strange?

4. Can you explain the last contribution of the paper?

### Soundness
2

### Presentation
2

### Contribution
2
