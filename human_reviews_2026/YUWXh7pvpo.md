# Toward Optimal ANC: Establishing Mutual Information Lower Bound

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Active Noise Cancellation (ANC) algorithms aim to suppress unwanted acoustic disturbances by generating anti-noise signals that destructively interfere with the original noise in real time. Although recent deep learning–based ANC algorithms have set new performance benchmarks, there remains a shortage of theoretical limits to rigorously assess their improvements. To address this, we derive a unified lower bound on cancellation performance composed of two components. The first component is information-theoretic: it links residual error power to the fraction of disturbance entropy captured by the anti-noise signal, thereby quantifying limits imposed by information-processing capacity. The second component is support-based: it measures the irreducible error arising in frequency bands that the cancellation path cannot address, reflecting fundamental physical constraints. By taking the maximum of these two terms, our bound establishes a theoretical ceiling on the Normalized Mean Squared Error (NMSE) attainable by any ANC algorithm. We validate its tightness empirically on the NOISEX dataset under varying reverberation times, demonstrating robustness across diverse acoustic conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a unified bound of any active noise cancellation algorithm based on two components such as information theoretic and support based derivations, and demonstrates the effectiveness with experiments on a benchmark dataset called NOISEX. The theory gives a tight bound and the empirical study confirms the bound appropriately.

### Strengths
- The theoretical bound is well derived based on sufficient mathematical derivation.

- The experiments on a benchmark dataset confirms the theoretical bound effectively.

### Weaknesses
- The usefulness of the theory is strictly bounded by the size of dataset, which might be a critical limitation for practical usefulness.

- The experiments are too limited to draw any interesting conclusion. Larger and more datasets should be used to verify the proposed theory.

### Questions
- What is the practical usefulness of the proposed theoretical bound?

- How can you argue on the general applicability of the proposed theoretical bound?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new benchmark and evaluation framework for active noise cancellation (ANC) systems, addressing the lack of standardized and reproducible evaluation methods in this field. The authors identify that much of the existing ANC research relies on isolated simulations and ad-hoc metrics that do not translate well to real-world use cases. To tackle this issue, they present a hardware-in-the-loop testbed and a unified set of evaluation metrics that combine objective and perceptual measures. The framework includes a curated dataset and baseline comparisons of both classical and learning-based ANC approaches under realistic noise scenarios. The overarching goal is to enable consistent, comparable, and physically grounded assessment of ANC systems across research and industry settings.

### Strengths
(1) The paper’s emphasis on realistic, hardware-in-the-loop evaluation is timely and significant. Most prior studies have focused on simulation-based analysis, which often overlooks key practical factors such as transducer response, delay, and spatial sound propagation. By integrating real-world noise environments into the testing framework, the authors make an important step toward closing the gap between theoretical ANC models and deployable systems.

(2) The proposed metric suite is another valuable contribution. It combines objective measurements such as noise attenuation, latency, and power consumption with perceptual or user-centered measures, providing a more comprehensive picture of ANC performance. This balanced evaluation approach aligns well with how ANC systems are judged in consumer and industrial contexts, where both signal fidelity and user comfort matter.

### Weaknesses
(1) The statement that existing ANC research lacks real-world evaluation or unified metrics is somewhat overstated. There is a substantial body of work, particularly from the audio engineering and acoustic signal processing communities, that includes hardware-based testing and adherence to industry standards for headphone or ear-cup ANC evaluation. The contribution of this work would be better framed as extending these established practices into a machine learning–oriented benchmarking context rather than claiming an entirely unexplored area.

(2) The diversity and representativeness of the proposed dataset are not clearly established. Real-world ANC applications span a wide range of acoustic conditions—engine rumble, wind, speech interference, irregular transients, and user motion artifacts. Without sufficient coverage of such variations, the benchmark may not generalize across different use cases or device form factors. More details on the environments, noise categories, and device configurations would strengthen the paper’s claims.

(3) The baseline comparisons presented appear limited. If the evaluation includes only a small set of algorithms or omits the most recent adaptive and hybrid ANC techniques, it becomes difficult to assess the framework’s true benchmarking value. A more extensive comparison including both classical adaptive filtering approaches and advanced learning-based systems would offer a fairer and more informative reference point.

(4) The long-term impact of the benchmark depends heavily on its accessibility and community uptake. Without a clear plan for public release, maintenance, or integration with open repositories, the benchmark risks becoming another isolated dataset. The authors should describe mechanisms for community engagement, version control, and contribution guidelines to ensure the framework remains relevant and widely adopted.

(5) The inclusion of a reproducible experimental protocol would substantially enhance the benchmark’s utility. Providing a detailed description of test scripts, noise source configurations, user movement patterns, latency budgets, and power constraints would make it easier for other researchers to replicate results and compare new methods under standardized conditions. This level of procedural transparency is essential for transforming the proposed framework into a shared community standard.

### Questions
(1) Can the authors provide a quantitative comparison between their proposed benchmark and existing industry-standard ANC evaluation methods—highlighting where their framework introduces new machine-learning–oriented metrics or testing conditions—to clarify its distinct contribution and justify the claim of novelty?

(2) Can the authors include detailed statistics or visual summaries of the dataset’s diversity (for example, number of environments, noise categories, and device configurations) and report benchmark performance across these subsets to objectively demonstrate its coverage and generalization across realistic ANC scenarios?

(3) Can the authors expand the baseline evaluations to include both classical adaptive filtering algorithms and recent learning-based ANC systems, presenting standardized performance metrics across all methods to substantiate the benchmark’s comprehensiveness and relevance to the broader research community?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies the theoretical lower bound of the normalized mean squared error (NMSE) of an active noise cancellation (ANC) system under suitable assumptions. The lower bound is the maximum between a support-based bound and an information-theoretical bound. The support-based bound is derived based on the lack of frequency component modeling in the secondary path. The information-theoretical bound is derived from the mutual information between the disturbance and the cancellation signal. Several experiments are conducted to show the bounds and the gap between the unified bound and the actual NMSE. It is shown that the information-theoretical bound increases with a larger reverberation time and the support-based bound leads for low reverberation times. The experiments are also conducted with different noise types, and the bounds show a consistent trend with the reverberation time.

### Strengths
This is a well-written paper with good clarity. As a reader, I enjoy reading the paper. The theoretical bounds derived are verified in simulations and they capture the trend of the measured NMSE.

### Weaknesses
The information-theoretical bound is not surprising given that we know the mutual information and differential entropy rate of the disturbance process. For the support-based bound, it is trivial that the lack of frequency component modeling establishes a lower bound for NMSE. Therefore, the novelty of this paper seems weak.

### Questions
1.	For the support-based bound, why does it have a larger bound when the reverberation time is small?
2.	In practice, $supp(P)/supp(S)$ should be always 1. Was some sort of thresholding applied in the simulations?
3.	There are several assumptions made in the paper including the reference signal being WSS and the primary path being LTI. Are they realistic assumptions?
4.	Loudspeakers are always nonlinear. Why do we only consider the linear case? If we model some nonlinearities, would the conclusions change?

### Soundness
2

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
4

### Summary
This paper proposes a theoretical framework to characterize the fundamental performance limits of Active Noise Cancellation (ANC) systems. The authors derive a unified lower bound on achievable normalized mean squared error (NMSE), combining two terms: (1) an information-theoretic bound relating residual error to the mutual information ratio 
I(y;d)/H(d) between disturbance and cancellation signals, and (2) a support-based bound quantifying the irreducible error arising from spectral regions where the cancellation path has no gain. The unified bound is defined as the maximum of these two limits. The paper validates the framework empirically using multiple deep-learning ANC models (DeepANC, ARN, DeepASC) on the NOISEX dataset under varying reverberation times, showing that all models remain above the proposed theoretical floor.

### Strengths
The paper focuses on an information-theoretic perspective to ANC and prrovides a clean intuitive separation between algorithmic (information) and physical (spectral) performance limits.
The paper primarily, could help researchers reason about 'how close' learned ANC systems operate to theoretical limits, serving as a conceptual benchmark.
Finally, the paper evaluates multiple datasets, reverberation conditions, and baseline models to demonstrate empirical consistency of the bound.

### Weaknesses
While the paper’s conceptual framing is interesting, its theoretical and methodological depth remains limited. The main derivations rely heavily on established principles from information theory and signal processing, notably the Shannon rate-distortion lower bound and classical spectral support arguments derived from Parseval’s theorem. The proposed 'unified' bound is constructed heuristically by taking the maximum of these two well-known limits, without formal justification or proof that this composition represents a true optimality condition. The derivation employs standard simplifying assumptions common in adaptive filtering; such as approximate Gaussianity, wide-sense stationarity, and linearization to make the analysis tractable. While these assumptions are reasonable for first-order theoretical treatments, the paper does not adequately discuss their scope or limitations, particularly in nonstationary or nonlinear ANC scenarios where the bound’s validity may break down. Moreover, the empirical validation focuses primarily on visual trends across noise types and reverberation times but does not provide quantitative evidence of how close existing models actually come to the theoretical limit. Without metrics of bound tightness or uncertainty, the results remain largely illustrative. Finally, the narrative tends to reiterate the same conceptual message about unifying information-theoretic and physical constraints without developing deeper analytical or practical insight. Overall, the work reads more as a pedagogical synthesis of established ideas than a substantive theoretical advancement.

### Questions
How does your formulation relate to classical estimation limits such as the Cramér-Rao or Bayesian bounds in the linear-Gaussian ANC setting?
While the conceptual link between information content and achievable ANC performance is valid regardless of the estimator used, the specific empirical demonstrations rely on a highly approximate kernel-based mutual information estimation procedure. Would you agree that given the estimator’s bias and instability for continuous correlated data, the numerical values of I(y;d) and the claimed 'tightness' of the bound should be interpreted qualitatively rather than quantitatively?

### Soundness
2

### Presentation
3

### Contribution
2
