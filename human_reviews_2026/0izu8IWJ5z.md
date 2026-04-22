# Neural Gaussian Radio Fields for Channel Estimation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Accurate channel state information (CSI) is a significant bottleneck in modern wireless networks, with pilot overhead consuming 11-21\% of transmission bandwidth and feedback delays causing severe throughput degradation under mobility. To address this, this work introduces a new class of neural fields designed for coherent wave-based phenomena, called neural Gaussian radio fields (nGRF), which combines the efficiency of explicit primitive-based representations with a novel differentiable operator. nGRF replaces view-dependent, computer graphics-centric rasterization with direct, complex-valued aggregation in 3D space that natively models wave superposition and interference. Consequently, this reframes the learning objective from a function-fitting task to a well-posed source-recovery problem. In evaluations, nGRF demonstrates superior performance across diverse environments. In indoor scenarios, it achieves 10.9 dB higher prediction SNR than state-of-the-art methods while reducing inference latency from 242 ms to 1.1 ms (a 220$\times$ speedup). For large-scale outdoor environments where existing approaches fail, nGRF achieves an SNR of 26.2 dB. Furthermore, the proposed method reduces the required measurement density by 18$\times$ (0.011 vs. 0.2-178.1 measurements/ft$^3$) and cuts the training time from hours to minutes (a 180$\times$ reduction), enabling rapid adaptation to dynamic environments. The code and datasets are available at https://github.com/anonym-auth/n-grf.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Neural Gaussian Radio Fields (nGRF), an explicit neural field model for channel estimation that represents wireless environments with 3D Gaussian primitives. Each Gaussian acts as a localized “radio modulator,” and the channel is rendered via complex-valued aggregation that models wave superposition.

### Strengths
+ The paper introduces an explicit and computationally efficient neural field representation for RF propagation, combining Gaussian primitives with complex-valued aggregation to model multipath interference directly.

### Weaknesses
- Each Gaussian primitive encodes only a single complex amplitude, conflating emission and propagation effects. Under Huygens’ principle, an RF source should be characterized by two distinct electromagnetic attributes, one governing emission (source excitation) and another governing spatial attenuation or scattering response. Collapsing these into a single term limits physical interpretability and prevents accurate modeling of distance-dependent phase and amplitude variations in realistic propagation.


- While synthetic evaluation is acceptable for a representation-focused study, the paper lacks any downstream validation to demonstrate practical impact. The experiments stop at SNR-based reconstruction on ray-traced data, without evaluating how the predicted channels improve real tasks such as beamforming accuracy, localization, or channel prediction under mobility.

- The paper omits a key baseline, WRF-GS (Wen et al., 2024), which shares nearly identical objectives, modeling wireless radiation fields using 3D Gaussian primitives for fast, physically grounded channel reconstruction. Since WRF-GS already achieves millisecond-level inference with high fidelity, excluding it weakens the comparative analysis and makes the reported speed and accuracy improvements difficult to substantiate.

### Questions
Please see the points raised in the Weaknesses section.

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
The paper proposes Neural Gaussian Radio Fields (nGRF): an explicit 3D-Gaussian–primitive representation to render complex-valued MIMO channels via direct 3D field aggregation (as opposed to NeRF-style implicit fields or 2D Gaussian splats). Claimed benefits include SNR gain over SOTA, lower inference latency, drastically reduced measurement density, and major cuts to pilot overhead.

### Strengths
1. CSI overhead and channel aging are well framed as bottlenecks; quantitative context is provided.
2. Direct 3D aggregation of anisotropic Gaussians is physically more interpretable than NeRF-style volumetric integration; the “localized radio modulator” interpretation is appealing.
3. Large SNR/latency and data-efficiency gains across indoor and large-scale outdoor scenarios, if reproducible, would be impactful for AI-native CSI estimation.

### Weaknesses
1. It is unclear whether results are simulation-only or include OTA/hardware-in-the-loop; claims of 26.2 dB SNR outdoors and millisecond-scale inference require hardware validation given calibration/clock/CFO issues and non-Gaussian clutter.
2. Treatment of frequency selectivity, Doppler/aging, CFO/phase, antenna mutual coupling, and mobility trajectories is not explicit; rendering complex H(f,t) rather than per-snapshot H appears under-specified.
3. Reporting measurements/ft³ lacks frequency-dependent coherence justification; it’s difficult to compare against pilot designs tied to coherence time/bandwidth.

### Questions
1. Do you have over-the-air or trace-driven evaluations (mmWave/mid-band) to support the 1.1 ms inference and outdoor SNR claims?
2. How are Tx/Rx positions obtained and encoded in real systems, and what is their signaling/estimation cost? Reconcile this with the 0.2% pilot claim for 100 MHz NR？
3. Derive formal time/memory complexity vs Gaussians/antennas/subcarriers; provide sensitivity to Gaussian count and pruning strategies？

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents nGRF, a new neural Gaussian field formulation for MIMO channel estimation in wireless networks.
Different from the black-box methodologies, nGRF represents the propagation environment as a set of explicit 3D Gaussian primitives, each acting as a learned local radio modulator. The method performs direct complex-valued aggregation in 3D space, and models wave interference and superposition natively.
This brings efficiency gain by eliminating view-dependent rasterization and costly ray tracing.

### Strengths
1. Novel representation: Introduces an explicit, physics-informed Gaussian primitive formulation that preserves the wave superposition principle, unlike alpha-composited 3DGS models.
2. Level of magnitude acceleration in training and inference compared with NeRF2 / NeWRF, while maintaining state-of-the-art accuracy.

### Weaknesses
1. Evaluation is only on synthetic data. The dataset is simulated based on ray-tracing within an ideal room setting with tidy, homogenous materials & flat surface, which is not convincing. Real-world measurements is necessary to strengthen empirical claims. The author can reuse NeRF^2's open source dataset.
2. The generalizability is only demonstrated via sparse sampling. Despite making sense, it far from adequate to represent real-world settings. Different room layouts, room size, obstacle material or wireless environments should be involved to confirm
3. The motivation of how MIMO leverages channel estimation is unclear. I think it's also a drawback of NeRF^2 paper. A deployment requirement statement is needed.

### Questions
1. The paper mentioned initializing with LiDAR geometry hurts performance, which is surprising. What's your analysis and understanding on it?
2. Do you think MIMO should also look at SNR for each RX unit, or should have more systematically communication-level metrics for evaluation? (like Packet reception rate)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- The paper tackles the problem of radio frequency (RF) propagation modelling and specifically, to learn a model that predicts the channel state information (CSI) at novel unseen locations of the scene.
- The approach extends the line of work around physically-constrained rendering (e.g., NeRF, 3DGS), which also relates to more recent wireless/RF NeRF approaches.
- The novelty of this work lies in leverating a 3D Gaussian Splatting based formulation: the propagation environment is represented as a set of high-dimensional gaussians

### Strengths
1. Physically-motivated rendering: The approach relies in a physically-constrained approach, which have previously shown to be beneficial to generalize and additionally interpret learnt parameters.
2. Evaluation is comprehensive: The approach is evaluated on three (synthetic) scenes, ablations are comprehensive and is accompanied by other interesting experiments (e.g., on generalization)

### Weaknesses
**1. Channel Rendering**
- It's somewhat unclear on why the channels are rendered in the manner proposed (Sec. 3.3). Specific points below.
- The spatial weighting $w_i$ term appears to upweigh contributions of gaussian "virtual transmitters" when $p_{rx}$ is close to the gaussian $\mu_i$. This seems intuitive, but however appears to overlook cases when there are obstructions. Specifically, for two equidistant rx locations (one with LOS and another with NLOS), it appears that weights would be similar.
- A related concern is the 3DGS equivalent notion of "depth compositing". If a surface (I suspect another gaussian) is in between the rx and tx, the formulation of how this gaussian attentuates the channel is not discussed.
- Furthermore, $w_i$ appears to only be a function of the receiver position, not the transmitter. Assuming reciprocity, wouldn't the weight be influenced similarly by a change of tx position?

**2. Analysis: Learnt Gaussians**
- While the overall quantitative results are promising, I find one particular analysis missing: distribution/insights of the gaussians learnt. On the vision/graphics side, this is fairly easy to intuit, that Gaussians primarily lie on surfaces of objects. However, in the RF setting, it's less clear what these represent. Are they representing the "surface" of an object, or virtual transmitters, or both?
- Further more, are variations of gaussian contributions smooth wrt locations of rx? 
- My overall concern is overfitting: that the parameters are learnt in a manner that is not "physically consistent" (e.g., phantom gaussians out of bounds of the scene) and leading to degenerate solutions.

**3. NLOS performance**
- From Eq. 3, I see that LOS and NLOS fields are learnt separately. This is a fair assumption if we know the geometry of the environment. However, this does not seem to be an assumption and makes me wonder how LOS/NLOS are determined in practice? 
- Additionally, with evaluation, going by the description in Appendix B., it appears that uniformly sampling rx locations would over-represent LOS rx locations. I request the authors to clarify how dominant are LOS rx locations (which can admit very easy solutions and does not test model's ability to generalize).

**4. Real-world evaluation**
- The paper is solely evaluated in synthetic scenarios. While I believe this has its own merits and good for the most part (allows more controlled evaluation), it would be interesting to see some real-world validation. 
- One suggestion would be the DICHASUS dataset, which has been used in other works for evaluating RF models. 

**5. (Minor) Directivity**
- Directivity of the antenna is not considered. This is a major factor in RF propagation.

**6. (Minor) Frequency Generalization**
- While the paper presents initial findings on frequency generalization, it appears to support it using a single qualitative example (Fig. 3b). I suggest a slightly more rigorous evaluation by quantitatvely including a reasonable size of examples.

### Questions
Please see the section above.

### Soundness
2

### Presentation
3

### Contribution
3
