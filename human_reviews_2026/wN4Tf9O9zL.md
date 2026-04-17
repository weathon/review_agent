# Physics-Informed Machine Learning under Climate Domain Shift: PDE-Free Physics Regularisation for Cloud Prediction

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
We study out‑of‑distribution generalisation in geophysical prediction and propose CC‑PINN, a physics‑informed multi-layer perceptron (MLP) that encodes the Clausius–Clapeyron thermodynamic relation as a gradient‑based regularisation term. Unlike prior PINNs, CC-PINN requires no explicit governing-equation. CC‑PINN introduces a lightweight constraint on humidity-temperature consistency without altering network architecture. Trained on atmospheric reanalysis data (temperature, pressure, relative humidity, specific humidity, vertical velocity) using modest computational resources, CC-PINN matches a capacity-matched MLP in-distribution and improves out-of-distribution performance. CC‑PINN achieves a 12.6\% reduction in global area-weighted RMSE over a capacity‑matched MLP baseline. Under a stringent covariate-shift test - training only on the polar latitudes - CC‑PINN reduces tropical area-weighted root mean squared error (RMSE) by 23.6\% relative to the baseline, while maintaining in‑distribution parity. Ablations show the performance gains are substantially attenuated when the physics term is removed, highlighting the role of targeted domain knowledge inclusion in improving extrapolation. These findings suggest that compact, domain‑motivated regularisation can deliver robust generalisation in scientific ML tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the out-of-distribution (OOD) generalisation problem in climate prediction, specifically, cloud fraction estimation by proposing a lightweight physics-informed MLP based neural network called CC-PINN. CC-PINN introduces a physics-based regularisation term derived from the Clausius–Clapeyron (CC) thermodynamic relation, which links saturation vapour pressure to temperature. This CC-based term is added as a gradient supervision constraint that aligns the model’s temperature sensitivity with the physically expected CC slope, enforcing humidity–temperature consistency without explicitly encoding a PDE.

### Strengths
It reframes physics-informed learning as soft inductive bias alignment rather than explicit PDE supervision.
It applies this principle to the domain of climate OOD generalisation which is not addressed by data-driven models.
The Polar and Tropics transfer setup offers a novel experimental protocol mimicking climate regime shifts, further reinforcing originality in evaluation design. The approach is architecture-agnostic and computationally lightweight, making it efficient without large compute budgets.

### Weaknesses
The evaluation uses only two fixed timestamps August 1, 2024 (training) and December 12, 2024 (testing), representing a single diurnal and seasonal pair. Since the stated goal is to test out-of-distribution (OOD) robustness under climate regime shifts, two discrete snapshots may not sufficiently capture the temporal variability.
The study evaluates only RMSE (with area weighting). Other metrics such as bias, correlation, and uncertainty quantification could strengthen the paper's proposed contributions. RMSE alone does not capture systematic bias, error asymmetry (e.g., over- vs. under-predicted cloud fraction), or uncertainty reliability, which are crucial for scientific interpretation.
In terms of climate variability qualitative evaluation is missing.

### Questions
Could the authors clarify why only two timestamps (August and December 2024) were chosen?
How representative are these two snapshots of broader seasonal and inter annual variability in cloud–temperature–humidity coupling?
Why was RMSE chosen as the sole evaluation metric? 
Can authors provide Global error maps (absolute/bias) for baseline vs CC-PINN for better evaluation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes adding a physics-based regularization term to better predict cloud fraction across the world. The Clausius-Clapeyron relation is a known thermodynamic relation describing cloud formation. The constraint is introduced in an established simple NN architecture and compared to the identical architecture without the constraint. The paper shows that on their test set, the cloud fraction prediction on ERA5 is improved.

### Strengths
The constraint is a nice physical relation; it is easy to add to the neural network presented, and it improves performance. The authors sufficiently show the effectiveness of the approach.

### Weaknesses
I am a bit confused by the train and test set consisting of only one time step; this does not seem to be enough to show the usefulness of the constraint. More extensive temporal evaluation is definitely necessary. Additionally, most forecasting models are probabilistic, I would therefore recommend to asssess the constraint at least additionally in a probabilistic setup.

### Questions
- What do you expect to change if you use more than one time step for training and testing?
- What would you expect to change if a better dataset is used than ERA5 (as this one is known to be lacking in cloud fractions)?
- Can you elaborate on why you do not do probabilistic forecasting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a physics-informed neural network (CC-PINN) based on Clausius–Clapeyron thermodynamic relation targeting geophysical prediction. In particular, CC-PINN  is deployed for Cloud prediction. To demonstrate, a prediction accuracy-based comparison between NN and PINN is performed.

### Strengths
Using PINN in such application is very useful  as PINN was proposed to integrate knowledge of physical laws (in the form of partial differential equations). This leads to improvement in NN prediction accuracy. However, the paper made an excellent claim, indicating that the proposed CC-PINN  can perform without explicit governing-differential equations. Instead, CC-PINN uses a constraint ingratiated into the loss function (normalised, area-weighted objective).

### Weaknesses
The paper presentation arises three main concerns of the paper claim:
- If eq(3) is used as a learning objective, then the proposed CC-PINN Is using differential equations to govern the training. Eq(3) needs the value of L_phy from eq(2), which is output of differential equations. Could you please clarify this point?
- The results demonstrated in Figure 1 are a comparison between MLP-based PINN and the baseline MLP, and show that MLP-based PINN exhibits less RMSE. But is  it a fair comparison? As MLP-based PINN uses the physical knowledge, having lower RMSE is a straightforward result.  Why is there no comparison with the state of the art paper? Such extra comparison can be very valuable to show the work uniqueness and to quantify its novelty.  
- Shouldn’t we compare first between analytical/ numerical Conventional methods and NN for this problem? Then, we can show CC-PINN is better. Such a comparison will show the actual motivation of the paper and provide more insights into physical perspectives i.e., the interpretation of prediction improvement in terms of RSME.

### Questions
To improve the paper readability, please consider the following points:
- In eq(1), what are these symbols? Please define all of them.
- In eq(3) na eq(5), the authors used “B”. As it is the same set, please define in eq(3), not later. 
- Why did you propose eq(4) since it was introduced before at the end of section 3.3.?

### Soundness
3

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
The paper introduces a novel method for improved out-of-distribution (OOD) generalization for deep learning-based cloud parametrization schemes in hybrid climate models. The authors advocate that a neural network predicting cloud fraction should have, up to constant scaling, the same partial derivative wrt. temperature than the derivative predicted by the Clausius–Clapeyron equation. To enforce this property, the authors propose to additional minimize the mean squared error between these two quantities. For validation, the authors train a small MLP to regress cloud fraction from atmospheric covariates, such as temperature and specific humidity, on ERA5 reanalysis data and compare results with and without the additional loss term. They find that when trained with the additional objective and on polar regions only (low temperature and humidity) the network better generalizes to the tropics (high temperature and humidity). The evaluation protocol is supposed to mimick the kind of distribution shift that is to be expected in a warmer climate.

### Strengths
The paper is overall well motivated and tries to tackle a highly relevant and under-explored problem. Namely, how physical priors can be utilized to make neural networks more robust to climate change and, thus, applicable for (hybrid) climate modelling. The argument for this is clearly laid out by the authors in the introduction. 

This problem also bears significance to the broader machine learning community as a case study for a strong and continuous form of distribution shift.
In the context of climate modelling, the concrete task of cloud fraction parametrization is well chosen, and again, the authors clearly explain its significance. While the Clausius–Clapeyron equation has been utilized before to improve robustness to a warming climate, its direct use by the authors as target for the temperature sensitivity of a neural network is a novel and welcomed contribution. If effective, the suggested method could act as straightforward and simple way to improve robustness in cloud parametrization schemes.

### Weaknesses
While the motivation and method of the paper are well founded, its main weakness is the lack of compelling evidence for the claims made therein. The paper could improve substantially by conducting more thorough experiments and evaluations.

**W1. Insufficient data for training and evaluation**

A major flaw is the use of only two (specific) time steps, i.e. hours out of roughly 400,000 available ones for training and evaluation. This problem is exaggerated even more by the fact that the grid cells stemming from a single time step are not independent but are highly correlated in space, especially for the temperature field or at pressure levels in the stratosphere. It is completely unclear whether the results presented in the paper are just a mere coincidence for one particular atmospheric state or whether they hold in the general. I suggest that the authors re-run their experiments on sufficiently long training, validation, and testing periods that span years to decades.

**W2. Missing comparison to baseline methods**

The authors do not compare their results to existing methods for OOD generalization. This makes it difficult to gauge the overall effectiveness of the proposed method to alternative approaches. Moreover, no comparisons to existing cloud fraction prediction methods are made. While the main focus of the paper is not on a state-of-the-art cloud fraction prediction, comparing the method to existing results 
would made it easier to understand how well the trained model solves the prediction task in the first place. I suggest that the authors include other baseline methods, both for OOD generalization, as well as cloud fraction prediction, in their evaluation.

**W3. No direct experiments on climate change induced distribution shifts**

The stated purpose of the paper is to improve robustness in face of a warming climate. However, the authors explicitly decide to use a spatial distribution shift (polar vs tropic regions) as a proxy for a warming climate. ERA5 back-extends to 1950 and, thus, already contains significant amounts of data under a warming climate. The authors could have tested their hypothesis directly on ERA5 by training up until a certain date, for instance the year 2000, and reserve data past that date for evaluation. Such an evaluation protocol would exactly match the training setup of the targeted use-case of a data-driven hybrid climate model. To demonstrate robustness in the face of even stronger shifts than those captured by reanalysis data, climate model runs could have been used as additional verification tool. While a neural network would only be able to act as emulator in that case, such experiments could still be insightful to explore the generalization behavior, in particular under different forcings.

**W4. Lack of qualitative evaluation and figures**

The main claims of the paper are primarily explored by investigating RMSE grouped by region or temperature. However, the paper is surprisingly void of figures, containing one in total. Possible approaches to generate more insights into the problem and increase confidence in the method could be, but are not limited to:

1. Comparing spatially resolved cloud fraction maps between prediction and ground truth.
2. Plotting RMSE as function of latitude and longitude, leading to a more detailed picture than mere grouping by latitudinal bands. E.g. to find differences between oceans and land surfaces.
3. Plotting cloud fraction prediction against temperature for a specific sample. For instance, to see differences in smoothness between the regularized and unregularized version of the model.
4. Scatter plots or kernel density estimates of temperature and humidity for both in-domain and out-of-domain data to visualize the underlying shift in the joint distribution.

**W5. Small model size**

Results are solely presented for a very small model with less than 500 parameters. Such model size might be particular appropriate for a hybrid climate model due to its low compute overhead. However, from a more theoretical point of view, the claim of the authors could be strengthened further if the method would yield comparable gains when scaled. 

**W6. Paper assumes (moderate) background in field-specific domain science**

The paper could be made more approachable for a general machine learning focused audience by explaining the geophysical background in more detail. For instance, by showcasing the Clausius–Clapeyron relation on a phase diagram and explaining its relationship to cloud cover.

### Questions
**Q1:** Equation 6 is missing the normalization factor $\frac{1}{\sqrt{\sum_{i \in B}{w_i}}}$ (compare with Equation 3). Is this on purpose? If so, this will make comparison between different latitudes void.

**Q2:** Why are two different tolerances $\tau_g$ and $\tau_s$ used in the definition of the directional agreement metric?

**Q3:** The Clausius–Clapeyron relation assumes thermodynamic equilibrium and an ideal gas. Can there be situations where imposing it as soft constraint on a neural network can be detrimental? Have you looked into samples that showed a particular pronounced discrepancy between the regularized and unregularized version of the model?

**Minor comments:**

1. Captions in Figure 1 are significantly too small and are illegible when printed.
2. Lines 191-192 seem to reiterate previously discussed points (compare with lines 166-171) and appear to be an artifact from a previous version of the text.
3. Consider incorporating Footnote 2 on Page 4 into the main text.
4. Table 3 on Page 7 is never referenced in the text and reiterates results from Figure 1. Either one could be placed in the Appendix.
5. Equations in Section 5.4 on lines 340-348 are not numbered.
6. The abbreviation *SEM* is first used on line 190 but not introduced until line 262.
7. The $\tau$ used in the definition of the tolerance-aware sign function is never introduced in the text.
8. Section 4.4 explains standard procedure. The text could be made more concise and clear by moving it to the Appendix.

### Soundness
2

### Presentation
2

### Contribution
3
