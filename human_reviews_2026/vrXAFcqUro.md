# INFER : Learning Implicit Neural Frequency Response Fields for Confined Car Cabin

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
Accurate modeling of spatial acoustics is critical for immersive and intelligible audio in confined, resonant environments such as car cabin. Current tuning methods are manual, hardware-intensive, and static, failing to account for frequency selective behaviors and dynamic changes like passenger presence or seat adjustments. To address this issue, we propose INFER ( Implicit Neural Frequency Response fields), a frequency-domain neural framework that is jointly conditioned on source and receiver positions, orientations to directly learn complex-valued frequency response fields inside confined, resonant environments like car cabins. We introduce three key innovations over current neural acoustic modeling methods: (1) an end-to-end neural frequency response field that directly learns frequency-specific attenuation in 3D space; (2) perceptual and hardware-aware spectral supervision that emphasizes critical auditory frequency bands and deemphasizes unstable crossover regions; and (3) a physics-based Kramers–Kronig consistency constraint that regularizes frequency-dependent attenuation and delay. We evaluate our method over real-world data collected in multiple car cabins. Our approach significantly outperforms time- and hybrid-domain baselines on both simulated and real-world automotive datasets, cutting average magnitude and phase reconstruction errors by over 39\% and 51\%, respectively. Our experiments show that INFER achieves state-of-the-art performance frequency response modeling in automotive spaces.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Authors introduced INFER, a novel spectral-domain framework that models acoustic propagation in
confined environments using implicit neural representations. By operating directly in the frequency
domain, their method enables perceptually grounded supervision, hardware-aware weighting, and
physically consistent regularization through the Kramers–Kronig constraint. They did extensive evaluations on real and simulated car cabin datasets demonstrate that INFER substantially outperforms prior time-domain and
hybrid approaches, achieving over 50% improvement in phase accuracy and 39% in magnitude fidelity relative to the best baseline.

### Strengths
INFER substantially outperforms prior time-domain and hybrid approaches, achieving over 50% improvement in phase accuracy and 39% in magnitude fidelity relative to the best baseline. 

They  first propose to encode KK-consistent complex attenuation in a neural acoustic renderer, preventing non-physical phase behavior and improving both interpretability and generalization.

Their proposed total loss is novel and it is a key for achieving the best accuracy.

### Weaknesses
Authors are solving a specific problem of neural modeling for frequency response field for confined car cabins.
Can this approach be scaled on other applications?

### Questions
It is surprising that this approach, based on vanilla sequence of fully connected layers is outperforming other methods.
Please explain why?
E.g. that means that the baseline is weak, or method in this paper uses additional data which are not used by the baseline?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Implicit Neural Frequency Response fields (INFER), a neural acoustic field learning framework for confined car cabin.

Key contributions are:
- First to apply the end-to-end frequency domain modeling for car cabins.
- Proposed Kramers–Kronig physical consistency regularization to enforce spectral attenuation and phase delay across frequencies.
- Evaluation on both COMSOL simulated data and real car cabins show that INFER outperforms the other baselines (NAF, INRAS, and AVR).

### Strengths
- **Originality:**
    
    INFER is the first to offer fully frequency-domain neural acoustic field modeling for car cabins.
    Integrating Kramers–Kronig grounded physics in a regularization is novel and well-motivated.
    
- **Quality:**
    
    The technical presentation is thorough, underlying principles are detailed, and the method is carefully justified physically and psychoacoustically.
    The evaluations on both synthetic and real environments shows fair benchmark comparisons. Multiple metrics are reported and samples are visualized to support claims.
    
- **Clarity:**
    
    The paper generally presents its methodology clearly. While there are some omissions, it provides relatively detailed figures, including hyperparameters.
    
- **Significance:**
    
    *If the author releases the code and data* as promised, this study will be significant as a benchmark. Yet, this reviewer cannot find neither code nor data from the supplementary material at this moment.

### Weaknesses
1. **Novelty:**

    The method builds on existing ideas in neural implicit fields for acoustics (NAF, INRAS, AVR).
    While the application of frequency-domain learning and KK regularization is novel *for car acoustics*, such novelty, as suggested by its distinctions from prior research, is somewhat limited.
    The frequency-domain modeling is not actually a particularly unique method and has already been applied in many other spatial audio studies [1,2]. As the author also explained, the fact that the frequency-time hybrid approach AVR already exists is in a similar vein. Reports of AVR's inferior performance compared to NAF make one wonder again whether the advantages of frequency-domain modeling truly is significant in this context. For this reason, this reviewer cannot agree with the authors' claim that the 'end-to-end frequency-domain forward model' is novel, as insisted in the key contributions.
    
2. **Ablation study:**

    The paper only reports on the proposed approach for the specific case, which is of a car cabin. This alone does not undermine the author's claim, but it does imply that the audience of interested readers may be somewhat limited. Without ablation studies, it is hard to provide meaningful insights unless one is specifically interested in car cabins. For instance, it remains unclear whether frequency modeling or the KK regularization trick also benefits learning indoor acoustic fields (where NAF, INRAS, and AVR were tested on). Therefore, the author's claim is valid only for ‘car cabins’.
    
3. **Generalization:**

    The datasets (COMSOL, BUCK, Tesla Model X) are well chosen, and providing details about the measurement setup is much appreciated.
    However, it is extremely difficult to determine how many source-receiver pairs were sampled per scene and where they were sampled.
    To my understanding, like many other neural acoustic field learning works, this paper also tackles the problem of generalization within a scene (not across scenes). In this case, it is critical to see how the approach generalizes well for the training data's sparsity and its sampling distribution, as the key issue is "how sparse the source-receiver pairs used for training can be".
    Yet, there are no experiments addressing this at all.
    
4. **Interpretability and Physical Plausibility:**
    
    The paper enforces KK relationships in loss, but the qualitative impact on physical interpretability is mostly assumed.
    (For example, do reconstructions violate causality if KK is removed? Do phase/magnitude predictions by INFER become less plausible without such regularization?)
    More ablation studies on the necessity of KK, and analysis of physical interpretability (not just metric fidelity), would strengthen the claims.
    
5. **Baseline Fairness:**
    
    The authors explain that all baselines are re-implemented in their codebase. The effort is to be commended, but there could be unintended pitfalls.
    For example, the baseline was not properly verified, and it is questionable that NAF—which appeared first among NAF, INRAS, and AVR, and has generally been reported to perform worse than INRAS or AVR—shows the second-best performance after INFER.
    It is unclear whether the automotive cabin environment is exceptionally unique, as there is no way to know (since no comparative experiments were conducted).    
    
[1] Lee, J. W., & Lee, K. (2023). Neural fourier shift for binaural speech rendering. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE.

[2] Di Carlo, D., Nugraha, A. A., Fontaine, M., Bando, Y., & Yoshii, K. (2024). Neural Steerer: Novel steering vector synthesis with a causal neural field over frequency and direction. In 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW) (pp. 740-744). IEEE.

[3] Wang, M. L., Sawata, R., Clarke, S., Gao, R., Wu, S., & Wu, J. (2024). Hearing anything anywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11790-11799)

### Questions
1. Regarding Appendix A7 (Evaluation Metrics), it seems like KK Violation Metric was used, but this metric does not appear anywhere in the paper. Please report this metric?

2. The paper states that “hardware-aware weighting” is a key contribution, but nowhere does it explain *where it originated from* or *how the values were determined*. From reading section 4.3, it seems like $w(f)$ is what the authors mean by "hardware-aware weighting," which turns out to be a heuristic "frequency-dependent weighting" in Appendix A3. If this actually improves performance, an ablation study should be included. For example, it should report how much performance drops when this trick is omitted from the proposed methodology, or whether applying this trick to the baseline architecture yields the same performance gains. (The same applies to the KK loss function.)

3. Although the content is repeatedly mentioned under ‘Weakness,’ is the proposal in this paper only applicable to car cabins? Are there no experimental results conducted in a room? Which part of the proposal specifically functions for the car cabins?

4. When will the code and dataset be made available as stated?

### Soundness
3

### Presentation
3

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
This paper presents INFER, a deep learning algorithm that predicts an acoustic transfer function at a 3D point. INFER is specially designed for environments with non-standard shapes and multiple materials with different absorption and dispersion properties, such as the interior of a car (which is the setting where INFER is tested).

INFER predicts both a complex attenuation factor (with independent predictions for magnitude and phase) as well as a directional retransmission factor. The final transfer function is computed by casting 32 rays from each source and accumulating the effects of each ray across 64 points, until we arrive to the query point. Once the transfer function is accumulated, a loss with 6 terms is computed that ensures the predictions of the network are physically realistic.

Evaluations on both simulated and recorded data such as a model cabin and an actual cabin of a Tesla Model X car show that INFER model the transfer function significantly better than other neural network methods which model the transfer function in the time domain.

### Strengths
* INFER applies machine learning techniques to an interesting domain (transfer function estimation in complex environments), which would be difficult to solve with traditional signal processing techniques.
* The manuscript contains a useful primer on the physics of audio propagation, making the paper accessible to non-acoustics-experts.
* The paper is really well motivated, particularly around the need for frequency domain modelling for the neural network.

### Weaknesses
* **W1**: While the motivation, related works, and acoustics primer are really well developed; the machine learning aspects, the ray marching strategy, and the evaluation are missing important details and ablations (see questions below for more details).

* **W2**: There is nothing in INFER that limits it to car cabins. In principle, INFER should be able to approximate the transfer function in any environment. Consequently, the contribution of this work could be strengthened if there was an evaluation on other domains (eg. normal rooms, open environments, other complex environments). At the very least, a discussion of what environments INFER is ideal for should be added.

### Questions
### Machine Learning Questions

* **Q1**: What is the training set of INFER? Is it a random split of the datasets presented in sec 5.1?
* **Q2**: How many parameters does each network (attenuation, retransmission) have?
* **Q3**: What layer from the attenuation network is used for conditioning the retransmission network?
* **Q4**: [less important] Could you obtain better results with a convolutional neural network (possibly with parallel branches of multiple kernels)? After all, temporal and frequency features in a STFT are highly locally correlated.
* **Q5** [less important] A diagram of both networks would help better understand the ML contribution.

------

### Ray Marching Questions

* **Q6**: Can you provide further details on the ray marching setting? How do you ensure all the rays converge on p? Do you produce new rays at intermediate points? Do you perform any culling?. Overall, ray marching is only briefly described in the manuscript, but it deserves a more thorough explanation since it is a key part of INFER.

* **Q7**: What are the consequences of increasing/decreasing the number of rays/points for the ray marching?

-----

### Evaluation Questions

* **Q8**: What is the performance with the simulated (COMSOL) module? I was expecting to find those results in Table 1.
* **Q9**: What is the spread of the prediction errors across different 3d positions and frequencies for each method. Perhaps a heat map plotting {position-across-line x frequency x amplitude/phase error} would be helpful to illustrate the behaviour of INFER?
* **Q10**: What exactly is being reported in Tables 1 & 2? I assume it's the mean absolute error but this needs to be explicitly specified.
* **Q11**: How were the T60 and EDT errors computed? I assume directly from the resulting transfer function, but this needs to be explicitly specified.
* **Q12**: For Table 2, the frequency breakdowns were computed with the simulated, buck or Model X datasets?
* **Q13**: Can you provide errors for a higher range of frequencies (for instance third-octave)?
* **Q14**: The loss has many sub terms and the contribution of each one is not well understood. How were the relative loss weights established? Could you ablate the contribution of each term? For instance you could evaluated INFER with $\\{\lambda_\text{spec}, \lambda_\text{mag}, \lambda_\text{phase}, \dots \\} = 0$ and record the decrease in performance for each setting.
* **Q15**: What are the exact terms in $\lambda_\text{aux}$? This should at least be defined in the appendix.
* **Q16**: How were the $\omega$'s determined? What frequencies were de-emphasised and why?
* **Q17**: [less important]: I would suggest a user study where some participants (ideally $N>15$) listen to the sounds in the car cabin convolved, and then the same sounds convolved with different transfer functions (INFER, INRAS, AVR, NAF). Participants would then rate their subjective impressions of which one is closer to the baseline. I hypothesise INFER would do better than other baselines, but more importantly such an experiment would tell us how far we are from an "ideal" transfer function estimation method.
* **Q18**: Will the training and evaluation datasets be released? Sec 8 mentions "demo" datasets.

-----

### Nitpicks (do not affect rating, no need to follow up on rebuttal)

* **N1**: Lines 70-75 discuss related work, and indeed the same points are repeated in sec 2.1. I would suggest removing them or heavily summarising them.
* **N2**: $\Omega$ is undefined in eq. (3). I assume it is the volume being modelled, but this needs to be explicitly specified.
* **N3**: Sec 3.3 uses $G(x,x')$ but the rest of the manuscript seems to refer to the same concept as $\delta(x)$. I would stick to one notation to ease readability.
* **N4**: The bibliography needs cleaning up: some surnames are in all-caps, the same conference is sometimes in title-case, sometimes not.
* **N5**: What is TOF in Fig1?
* **N6**: In 4.2 $\hat{n}$ has unit length right? If so I would explicitly mention this.
* **N7**: I would suggest using another symbol for the smoothing filter around line 328. $\mathcal{S}$ is being used to denote the retransmission function.
* **N8**: In line 403 the model outputs $\sigma, \beta$ and $\mathcal{S}$ rather than $H$ correct? H is computed using equation (7) afterwards, isn't it?. If so line 403 needs to be rephrased.

### Soundness
2

### Presentation
2

### Contribution
1
