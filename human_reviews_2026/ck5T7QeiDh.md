# Confident and Adaptive Generative Speech Recognition via Risk Control

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Automatic Speech Recognition (ASR) systems frequently produce transcription errors due to acoustic variability, which require post-processing correction methods. Recent approaches leverage Large Language Models (LLMs) for generative ASR error correction using N-best hypotheses but rely on fixed set sizes regardless of input complexity and do not provide performance guarantees. We propose an adaptive framework that dynamically determines the optimal number of hypotheses for each input using risk control. This mechanism leverages ASR confidence scores and applies  Learn then test (LTT) to control the expected relative word error rate degradation compared to the best achievable performance for a given model and hypothesis set. Experimental results demonstrate that our approach provides theoretical guarantees with high-probability bounds while matching or exceeding fixed-size correction baselines and requiring fewer hypotheses on average, achieving substantial computational savings under diverse acoustic conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
LLMs are used to generate a speech transcription using N-best hypotheses from an ASR model. The paper proposes to use conformal prediction to reduce the size of the hypothesis set, thereby reducing computations in the LLM. 
Experiments show that a substantial reduction in the size of the hypothesis set is achievable without compromising performance (WER).

### Strengths
- The paper proposes a new application of conformal prediction to LLM-based error correction in ASR. 
- Extensive experiments are shown on three datasets. 
- There is a reduction in the size of the hypothesis set.

### Weaknesses
- I believe that the paper is an application of an existing concept of conformal prediction to an existing problem of generative error correction in ASR, and hence, the contribution lacks novelty.
- I strongly argue that some concepts are treated very superficially. Conformal prediction calibrates the threshold on a score function (confidence measure) of hypotheses without altering their order. The paper uses likelihood values as a measure of confidence (although it does not explicitly mention this). However, previous research on ASR has shown that likelihood values are a poor indicator of confidence [Li et al., Ravi et al.]. These likelihood values are often highly overconfident, and they are not able to correctly order the hypotheses. Simple thresholding may not yield significant performance improvements. BTW, [Li et al., Ravi et al] perform only top-label calibration, but works such as [Popordanoska et al.] can provide canonical calibration, which can be used to provide N-hypotheses.
- Again, the title of the paper highlights conformal risk control, but the experiments do not talk about controlling the risk. One would expect a detailed analysis of expected WER (while setting the threshold) vs the achieved WER. It appears from Table 1 that they do not match.

[Li et al.] Confidence Estimation for Attention-Based Sequence-to-Sequence Models for Speech Recognition, ICASSP 2021.
[Ravi et al.] TeLeS: Temporal Lexeme Similarity Score to  Estimate Confidence in End-to-End ASR, IEEE TASLP, 2024.
[Popordanoska et al.] A Consistent and Differentiable Lp Canonical Calibration Error Estimator, NeurIPS, 2022.

### Questions
- Please address my concerns listed in the "Weaknesses" section.
- Line 274 talks about the monotonicity assumption being violated 20% of the time. I would request details here. I expect this number to be much higher in the case of ASR likelihoods.
- Some minor typos, e.g., line 189.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the computational inefficiency of GER in ASR, where LLM typically operate on a fixed-size N-best hypothesis list. The authors propose an adaptive framework that dynamically determines the optimal number of hypotheses $n \le N$ for each input1. The core of the method is the application of conformal risk control (CRC) to this selection process2. Instead of controlling for absolute error, the framework is cleverly designed to control the expected relative word error rate (WER) degradation compared to the best achievable performance (the "post-LLM oracle") for that specific input. This is achieved by calibrating a threshold $\lambda$ on normalized ASR confidence scores, which in turn determines the set size $n$4. Experiments on TedLium-3, CHIME-4, and Common Voice show that this adaptive approach significantly reduces the average number of hypotheses (e.g., 57.1% on TedLium-3) while maintaining or even slightly improving WER compared to the fixed N=5 baseline.

### Strengths
The paper's central claims are supported by experiments. The use of standard benchmarks (HyPoradise) , strong ASR and LLM models (Whisper and LLaMA-2-7B) , and relevant metrics  provides a strong empirical foundation. The ablation study in Appendix D (Table D.2) effectively justifies the experimental design choice of training the H2T model on fixed N=5 sets while evaluating on variable-sized inputs.

However, there is a significant gap between the theoretical claims and the implementation. The Conformal Risk Control framework (Angelopoulos et al., 2024b) formally requires a bounded, monotone loss function to provide its distribution-free guarantees. The authors forthrightly acknowledge in Section 5.2 that their chosen loss function (Eq. 10) is not strictly monotone, with violations observed in approximately 20% of samples.

They provide an empirical justification for this deviation, noting that applying the standard monotonizing procedure actually degrades performance. This is because their method's advantage comes, in part, from exploiting these non-monotonic cases where a smaller hypothesis set genuinely outperforms a larger one (a key insight illustrated in Fig 1a and Table 2) .

While the empirical results are strong and the risk is empirically controlled, this violation means the method does not inherit the strict theoretical guarantees of CRC. The paper is thus a work of "empirical risk control" inspired by CRC, rather than a direct application of it. This theoretical-practical disconnect is the primary weakness in an otherwise sound paper.

The paper introduces a novel application of CRC to a practical and important problem in ASR: the computational overhead of fixed-size N-best lists for LLM-based error correction . The results demonstrate a clear path to more efficient and robust ASR systems.

### Weaknesses
- Theoretical Gap (Monotonicity Violation). This is the most significant weakness. The CRC framework's guarantees depend on a monotone loss function. The authors explicitly state their loss (Eq. 10) is non-monotone for ~20% of samples. Their justification for proceeding is empirical: enforcing monotonicity (as suggested by Angelopoulos et al., 2024b) hurts performance because it prevents the model from exploiting "good" non-monotonic cases where fewer hypotheses are better . This is a reasonable empirical argument, but it invalidates the theoretical guarantees. The paper should be more precise in framing its contribution as an empirically robust method inspired by CRC, rather than one that provides CRC's guarantees.

- The adaptive score function $\phi_{\gamma}$ (Appendix B.1) 44, while well-motivated by dataset noise characteristics 45, introduces dataset-specific hyperparameters $\gamma$ and $\tau$ (Table B.1). This adds a tuning step that moves away from the "distribution-free" simplicity often associated with conformal methods. A sensitivity analysis on these parameters would strengthen the paper.

### Questions
1. I would like to see if the method also works for speech translation correction alike GenTranslate (Hu et al.) N-best data in ACL 2024. (https://huggingface.co/PeacefulData/GenTranslate) This additional non-ASR only study would largely increase the impacts of the work in the GER community. I would consider to increase my score.

2. Choice of Risk Target $\alpha$: Appendix B.2 53states that $\alpha$ is chosen based on the validation set's degradation statistics (e.g., 90th percentile). I assume the different red operating points in Figure 2 55are generated by varying $\alpha$ (e.g., from 80th to 95th percentile ranges). Could the authors confirm this and perhaps briefly discuss the relationship between the chosen $\alpha$ and the resulting (WER, Avg. Set Size) trade-off? 

3. Sensitivity of Score Function: The score function (App B.1) 49is tuned per dataset (e.g., $\gamma=1.0$ for TedLium-3, $\gamma=0.0$ for Common Voice). How sensitive is the method to these choices? What would the performance-compute trade-off (Fig. 2) 51 look like for Common Voice if the TedLium-3 parameters ($\gamma=1.0, \tau=0.05$) were used? This would help clarify if this tuning is a minor optimization or critical for the method's success.

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
This paper proposes an adaptive framework for generative ASR error correction (GER) that addresses the limitations of fixed-size hypothesis sets by leveraging conformal risk control (CRC). The framework dynamically determines the optimal number of ASR hypotheses for each input using confidence scores and CRC, which controls the expected relative word error rate (WER) degradation compared to oracle performance. Evaluated on three datasets (TedLium-3, CHiME-4, CommonVoice) with varying acoustic difficulties, the method achieves substantial computational savings (up to 57.1% reduction in hypothesis usage) while matching or exceeding the performance of fixed-size GER baselines. Key contributions include the adaptive hypothesis selection mechanism, the first application of CRC to GER, and empirical validation of robustness across diverse acoustic conditions.

### Strengths
The adaptive hypothesis selection approach effectively resolves the inefficiency of fixed-size sets, balancing performance and computational cost—a critical practical challenge in LLM-augmented ASR.
Integrating CRC provides statistical guarantees on relative WER degradation, addressing the lack of reliability assurances in existing GER methods and enhancing real-world applicability.
Comprehensive experiments across datasets with different acoustic characteristics (noise, accents, recording conditions) demonstrate the framework’s robustness and generalizability.
The ablation studies and detailed case analyses (full set required, single hypothesis optimal, performance plateau) provide deep insights into the mechanism’s behavior and validate design choices.

### Weaknesses
The loss function’s monotonicity violations (≈20% of samples) are acknowledged but not fully resolved; the decision to use the unmodified loss without monotonicity enforcement relies on empirical robustness rather than theoretical justification.
The framework’s compatibility with larger LLM architectures (beyond LLaMA-2-7B) and scalability to extremely large datasets are not evaluated, leaving uncertainty about its performance in more resource-intensive settings.
The score function’s parameters (γ, temperature τ) are tuned per dataset, which may limit adaptability to unseen acoustic environments without retuning.

### Questions
How does the framework perform when applied to larger LLMs (e.g., LLaMA-2-13B/70B or GPT-3.5/4) and does the computational savings persist relative to the increased inference cost of larger models?
Have you explored methods to automate the tuning of score function parameters (γ, τ) across unseen acoustic conditions, or is manual calibration required for each new dataset/environment?

### Soundness
3

### Presentation
3

### Contribution
3
