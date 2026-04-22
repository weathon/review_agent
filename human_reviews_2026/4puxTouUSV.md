# TRACEDET: HALLUCINATION DETECTION FROM THE DECODING TRACE OF DIFFUSION LARGE LANGUAGE MODELS

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Diffusion large language models (D-LLMs) have recently emerged as a promising alternative to auto-regressive LLMs (AR-LLMs). However, the hallucination problem in D-LLMs remains underexplored, limiting their reliability in real-world applications. Existing hallucination detection methods are designed for AR-LLMs and rely on signals from \emph{single-step} generation, making them ill-suited for D-LLMs where hallucination signals often emerge throughout the \emph{multi-step} denoising process. To bridge this gap, we propose \textbf{TraceDet}, a novel framework that explicitly leverages the intermediate denoising steps of D-LLMs for hallucination detection. TraceDet models the denoising process as an \emph{action trace}, with each action defined as the model’s prediction over the cleaned response, conditioned on the previous intermediate output. By identifying the sub-trace that is maximally informative to the hallucinated responses,  TraceDet leverages the key hallucination signals in the multi-step denoising process of D-LLMs for hallucination detection. Extensive experiments on various open source D-LLMs demonstrate that \textbf{TraceDet} consistently improves hallucination detection, achieving an average gain in AUROC of 15.2\% compared to baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1) This paper proposes TraceDet, a hallucination detection framework designed specifically for Diffusion Large Language Models (D-LLMs). The authors formulate the denoising process as an action trace and apply information bottleneck principles to identify the most informative sub-trace for detecting hallucinated outputs. Experiments on LLaDA-8B-Instruct and Dream-7B-Instruct across three question-answering datasets demonstrate an average AUROC improvement of 15.2% over baseline methods.
1) This work addresses the underexplored problem of hallucination detection in D-LLMs, where generation proceeds via multi-step denoising rather than single-step autoregression. To the reviewer's knowledge, existing hallucination detection techniques are primarily designed for Auto-Regressive LLMs (AR-LLMs) and fail to exploit the intermediate dynamics of D-LLM decoding. The approach is strongly motivated by identified literature gaps: existing AR-LLM detectors rely on single-step signals, which fail to capture D-LLM dynamics, as empirically evidenced by the paper's observations (Figure 1) and baseline underperformance.
3) The most significant concern relates to impact. While technically valuable, the community actively working on or using D-LLMs remains relatively limited; thus, the work's impact is currently more potential than immediate. Furthermore, the reliance on an "inspired by" claim for the IB principle (Section 3.3) represents a missed opportunity for deeper theoretical insight. Finally, by testing exclusively on QA datasets, the paper proves its concept in a controlled environment but does not address more challenging, unstructured tasks where hallucination detection is arguably most critical. These limitations collectively suggest that while the paper constitutes a solid contribution, its overall significance is not yet fully established.

### Strengths
1. The problem formulation is sound. Hallucination in the emerging class of D-LLMs is fundamentally different from that in AR-LLMs. The denoising process naturally introduces three distinct mistake patterns that classical methods for AR-LLMs cannot identify or methodically benefit from.
2. The methodology is innovative and intuitive. Leveraging the trajectory trace and extracting partial steps successfully aligns with the identified problem. The introduced Information Bottleneck (IB) principle naturally connects token-level entropy with uncertainty identification in a theoretical and principled manner. The correctness of the derived lower bound for optimization is carefully verified.
3. The evaluation evidence is sufficient and rigorous. The experimental results are convincing. The paper demonstrates consistent gains compared to output- and latent-based methods across comprehensive QA benchmarks. In addition to the main experiments, the ablation studies effectively demonstrate both the practical value and efficiency claims.

### Weaknesses
1. The assumptions underlying the IB principle application are not formally justified. The application of IB is presented more as an inspiration than a formally justified choice. The paper provides limited empirical evidence or theoretical citations to establish a solid foundation for its core methodology, which undermines the completeness of the research, despite the mathematical soundness of the IB derivation and promising results.
2. The scope is limited, and the impact should be further emphasized. While the paper aims to address hallucination detection, the experiments are confined to multiple-choice, open-ended, and contextual QA datasets. Hallucinations primarily occur in open-domain generations, whereas QA benchmarks are often considered closed-domain. The reviewer suggests evaluating hallucination mitigation for more open-ended and real-world generation tasks.

### Questions
1. What underlying causes drive the observed hallucination patterns? Can you provide analysis linking specific denoising dynamics to the emergence of hallucinations?
2. Could you elaborate on the key challenge that "the hallucination-relevant action may be sparse and unevenly distributed across the action trace, and not every action contributes equally to the emergence of hallucination"? Is there any theoretical or empirical support to justify the necessity of selecting a sub-trace A_sub?
3. How would TraceDet perform on long-form generation tasks (e.g., summarization, creative writing) where hallucinations are more prevalent and consequential? What adaptations would be required for such scenarios?
4. Is the assumption of independence for A_sub reasonable? Although Figure 5(b) shows the impact of different τ values, this choice remains questionable and warrants further justification.

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
The paper introduces an information bottleneck based approach (called TraceDet) for identifying hallucinations in diffusion large language models (D-LLMs). TraceDet relies on intermediate denoising steps generated during inference of a D-LLM to identify potential hallucinations in the outputs. On a high level, it works by first identifying a subset of most informative actions (predicted masked tokens) pertaining to the factuality of the output and predicts a binary label for hallucination. The paper compares TraceDet against existing approaches for hallucination including both output-based and latent space-based methods, showing favorable results.

### Strengths
- D-LLMs are an emerging area of research and practical interest (esp when compared to Autoregressive, transformer based models). The paper addresses the challenge of detecting hallucinations in D-LLM outputs, while leveraging the information from intermediate denoising steps during inference. This research direction is pretty novel, as far as I know.
- TraceDet is based on well established mathematical concepts (namely, information bottleneck). Though I haven't checked the formulation thoroughly, to the best of my knowledge it appears correct and I am happy to see methods with some theoretical grounding.
- The paper uses well established benchmarks in the area of hallucination detection, which makes judging the work easier.

### Weaknesses
- Please correct me if I am wrong, but don't many if not all of the baselines not involve any training of a hallucination detector? They are purely inference time, statistics based methods. I believe this makes the comparisons unfair. I am giving a weak reject for this reason alone.

**Minor Typos/Grammar** (these played no role in my score):
- There is an extra space after decimal point (15. 2%) in the abstract.
- In Section 3.1, do you mean something like "... remasking which retains the highest confidence tokens ..." instead of "... remasking which retains the most confidential tokens ..."?
- $s_t$ is missing TeX formatting in Section 3.2 -> Action.
- "d" should be capitalized in "d-LLM" in Section 3.2 -> Transition and in some other place in the manuscript.

### Questions
- If more comparisons with methods involving training a detector are provided, esp with similar training complexity, I am happy to raise the score. If not, please provide a justification for why such additional comparisons are unnecessary.
- Since IB is a form of regularization, can we please see comparison with other kinds of regularization? For example, one could add some L2-regularization in the TraceDet w/o Masking and see if that how that compares.

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
4

### Summary
In this paper, the authors study the detection of hallucinations in text generations of Diffusion Large Language Models (DLLMs), as opposed to the majority of existing literature which strictly analyzes autoregressive LLMs. Motivating using an information bottleneck perspective, the paper introduces TraceDet, to identify the most informative and relevant action sub-trace during the denoising steps using a network g that predicts the mask, and another classifier f to detect hallucinated responses from truthful ones over the selected sub-sequences. Overall, TraceDet is shown to be effective on DLLMs on standard datasets such as TriviaQA, HotpotQA and CommonsenseQA.

### Strengths
1) The paper studies a highly pertinent problem of detection of hallucinations in DLLMs, and is the first to the best of my knowledge to provide a dedicated analysis. While this can potentially limit the applicable scope since most LLMs in use today are autoregressive, the findings of the paper have great scope toward future development and understanding of DLLMs.


2) The paper is well-written and introduces each key component in a clear and concise manner. The TraceDet framework introduced by the paper is motivated well, and analyses across the denoising trace rather than specific single-point failure modes that might occur. For instance, the representative D-LLM hallucination patterns such as Interleaving, Persistent errors and Inconsistent guesses help back the need for such trace-based analysis.


3) The proposed method TraceDet demonstrates consistent and notable performance gains over a comprehensive suite of existing baseline methods that were originally developed for autoregressive LLMs, including both output-based and latent-based techniques on standard datasets such as TriviaQA, HotpotQA and CommonsenseQA. 


4) The paper also presents a fairly detailed analysis on the sensitivity of relevant hyperparameters such as the masking ratio $\tau$ and the regularization weight $\beta$, as well as re-masking strategies such as low-confidence,  entropy, random and top-k based masking. The latter further indicates that the proposed method would likely be effective in future variants of DLLMs as well, given the fairly consistent across different strategies.

### Weaknesses
1) MDP Formulation: The paper formulates the denoising process as a Markov Decision Process, however the proposed method does not adequately functionally utilise this MDP framework, given that the information bottleneck perspective is adopted. Further incorporation of MDP theory could potentially have been leveraged to help mitigate hallucinations, rather than focussing on detection alone.


2) Evaluation Metrics: The empirical evaluations rely solely on AUROC. In real-world applications (and in imbalanced datasets, which hallucination detection often is), AUROC when used in isolation can fail to adequately capture detection performance. Could the authors kindly provide other standard detection scores such as TPR at low FPR and the F1 score, to better analyze the practical efficacy of TraceDet?


3) The paper introduces an extractor $g_{\theta}$ and a predictor $f_{\phi}$ but provides limited detail on their architectural complexity. A more in-depth discussion of these components, along with an analysis of their training and inference costs relative to the main D-LLM, would be beneficial for assessing the method's practical efficiency. For instance, could these then be used to guide fine-tuning of the main DLLM to directly reduce generation of hallucinated responses?


4) The analysis in Section 4.3 with maximum token entropy did not appear to be clear - it  does not really help highlight how TraceDet is helping improve the discriminability, especially with Figure 3a. Furthermore, is it expected that the averaged trace entropy for truthful samples is in general higher than that for hallucinated samples?


5) The paper compares TraceDet to baselines adapted from autoregressive LLMs, many of which utilize multiple completed generations. A more direct and potentially stronger baseline would be to adapt these consistency-checking principles within the D-LLM's trace. For example, applying a consistency check across intermediate steps of a single generation would directly engage with the paper's core hypothesis.
 



Minor Typos:

Equation 1: $min_{f \in H}$ L(Y, h(r0)) , should likely be $min_{h \in H}$

Line 197: asked tokens from $st$ ---> asked tokens from $s_t$ 

Line 229 to 231: Minor grammatical errors

### Questions
Kindly refer to the questions mentioned in the weaknesses section above.

In Equation 1, the summation over i is quite unclear. This could be improved by providing more details in the main paper, as is currently given in Appendix D.

### Soundness
3

### Presentation
3

### Contribution
3
