# Low-Pass Filtering Improves Behavioral Alignment of Vision Models

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Despite their impressive performance on computer vision benchmarks, Deep Neural Networks (DNNs) still fall short of adequately modeling human visual behavior, as measured by error consistency and shape bias. Recent work hypothesized that behavioral alignment can be drastically improved through generative - rather than discriminative - classifiers, with far-reaching implications for models of human vision. 

Here, we instead show that the increased alignment of generative models can be largely explained by a seemingly innocuous resizing operation in the generative model which effectively acts as a low-pass filter. In a series of controlled experiments, we show that removing high-frequency spatial information from discriminative models like CLIP drastically increases their behavioral alignment. Simply blurring images at test-time - rather than training on blurred images - achieves a new state-of-the-art score on the model-vs-human benchmark, halving the current alignment gap between DNNs and human observers. Furthermore, low-pass filters are likely optimal, which we demonstrate by directly optimizing filters for alignment. To contextualize the performance of optimal filters, we compute the frontier of all possible pareto-optimal solutions to the benchmark, which was formerly unknown.

We explain our findings by observing that the frequency spectrum of optimal Gaussian filters roughly matches the spectrum of band-pass filters implemented by the human visual system. We show that the contrast sensitivity function, describing the inverse of the contrast threshold required for humans to detect a sinusoidal grating as a function of spatiotemporal frequency, is approximated well by Gaussian filters of a specific width.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper demonstrates that applying low-pass filters (gaussian blur and resampling) during test time improves model alignment with human perception in the context of the model-vs-human (Geirhos et al., 2021) benchmark framework. The authors claim that this result is further explained by established models of human visual sensitivity in the context of fast presentation times. To support this framework, an optimal filter that maximizes alignment is also learned and was shown to be, qualitatively, a low-pass filter.

### Strengths
The work addresses a topical problem of human-model alignment, specifically in an overlooked area of alignment, error consistency.

It is a particularly simple and elegant hypothesis and result that a simple low-pass filter can improve error consistency, and as far as I know, this is a novel result.

The manuscript is well written and communicated; the authors do a good job at leading the reader from the motivation, background to their results and conclusions

Prior work on human visual processing well covered and incorporated into the work; the authors cite well known, tested, and accepted results from the human vision literature.

While the experiment of blurring images is generally simple, the authors explore many different methods and levels of blur, and the results demonstrating a peak of error consistency for a given blur is interesting.

### Weaknesses
The majority of the weaknesses of the paper comes from the connection to human vision. While well cited, there are many shaky logical steps to support the claim that this improvement in error consistency with human data is because of a match in a low-pass-filtered image to the human percept of an image when viewed for only 200ms. Generally, I feel the result of improvement in alignment in itself with just a simple blur is interesting, and the shaky claims that this blur amount is a match to the effective blur of human vision in the context of a short presentation time detract from this.

The largest example of this is a major discrepancy between Kelly’s work on contrast sensitivity to spatiotemporal frequency and limited presentation time of images. Specifically, in Kelly’s work, two types of stimuli were presented, a flickering grating of the form
$$
f_S(x, t) =\cos\alpha x \cos \omega t 
$$
And a moving grating of the form
$$
f_T(x, t) = cos \alpha (x - v t)
$$
In both cases, the time dependencies are sinusoidal with a precise frequency whereas in the case of limited presentation of stimuli, the time dependency is a boxcar function (with a Fourier transform of a sinc function which has a peak, in fact, at f = 0). It is erroneous to take the inverse presentation time to be the frequency. 
While Kelly's CSF might still be the best model available, the authors should at the very least acknowledge this discrepancy.

### Questions
The authors do not discuss how this result compares with other test-time pre-processing methods. Does previous work exist? What about a set of experiments that benchmarks various corruptions used in OOD-accuracy experiments from MvH (e.g. uniform noise, low contrast, etc.)

I recommend a key control experiment which would be to apply varying levels of high pass filtering as a test-time pre-processing manipulation. 

While a human psychophysics experiment may be difficult to do during the rebuttal period, testing a subset of the images in humans with longer presentation time with and without blur would be the key experiment to determine if it is indeed the magno stream behind the human baseline.

Are the authors referring to the magno/parvo streams when they discuss the relationship to human perception, and claiming that the magno stream with it's low spatial frequency sensitivity is behind this low-pass-filtered effect described?

Minor Points:
A “neural transfer function” for short presentation times is mentioned in line 069 without prior introduction. Is this essentially the CSF?

Equation 1 has a $H_\theta$ that should probably be $G_\theta$, the learned filter

Equation 3 should have “df” to end both integrals on the numerator and denominator rather than being after the fraction.

On figure 3, there should be axis labels and the standard deviation of the learned filter (either in Fourier or real space) should be reported and compared against the gaussian blur filters. Furthermore, it could be helpful to show images filtered by this filter.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors investigate how simple down-sampling or low-pass filtering operations increase the alignment between *artificial vision models* (either discriminative, bottom-up, or generative, top-down) and *human vision* in terms of error consistency, shape bias, and out-of-distribution accuracy. They find that low-pass filtering similar to the human Contrast Sensitivity Function is optimal to maximize error consistency with humans and to get similar shape-bias in image recognition.  This finding is important in the discussion about the appropriate strategy (discriminative or generative) for object recognition both in humans and machines.

### Strengths
The (apparently) simple experiments proposed here, removing high-frequency information either by down-sampling or by low-pass filtering the input images before feeding the models, actually addresses a *key discussion* in machine vision: is it better a bottom-up approach to analyze the images and then discriminate between classes? (as done in conventional discriminative classifiers), or is it better a top-down approach where one checks if the input is compatible to generated examples form a given class? (as done in the "generative" classifiers as the ones defined by Jaini et al. 23, ICLR 24). This question is also relevant in understanding human vision where both strategies are certainly applied (though it is not clear how).

The results presented here constitute an excellent counter-example that allows to tone-down the conjectures about the benefits of the generative approach exposed in the ICLR 24 spotlight paper of Jaini et al. 23. While Jaini et al. suggest that their system may be "more human" (in shape-bias and error consistency) because of the generative approach, this work shows that this similarity may be just due to the fact that the considered models work with down-sampled images, thus effectively giving the model a signal which passed through a bottleneck similar to the one happening at the human LGN. As a result, as the signal content given to the model approaches the content arriving to the human visual cortex, it is reasonable that (1) models working with such filtered inputs have a generally "more human" behavior, and (2) those models focus on shape rather than texture because the texture has been largely attenuated by the CSF-like filter.

This counter-example reasoning is nice, results confirm that the simple low-pass operation alone does the alignment job for discriminative models, and show that the optimal filter resembles the human filter.  

Finally, the scientific question-explanation logic of the work is very well posed, which (I think) is unusual in certain ICLR papers: the authors first state an important question in a compelling way and propose a simple hypothesis as an alternative explanation. Then, they experimentally show that this alternative may be true, thus questioning recent work, and motivate further research.

### Weaknesses
Weaknesses in this work are minor, mainly limited to (a) some notation issues, (b) mention to works that show the emergence of human-like Contrast Sensitivity in artificial nets, and (c) clarification of the discussion between using the low-pass in training or test time.

(a) Notation in Eqs. 1-3 can be more clear. I eleborate in the "questions" box below.

(b) In the Related Work section the authors mention the work of Subramanian et al. 23 on the frequency response of ANNs. Other works (e.g. Li et al. J.Vision 2022 and Akbarinia et al. Neural Nets. 23) specifically adress the issue of the emergence of Contrast Sensitivities in ANNs which may be similar or different(!) to the CSF of humans depending on the task and on the architecture indicating a non-trivial interplay between these issues.

(c) Finally, the discussion about whether the low-pass filtering is (or is not) required in the training of the models to increase the alignment with humans is not clear (to me). The presented experiments certainly show that using it in test time increases the alignment with humans, but it is not clear (to me) the behavior one could get with training based on low-pass filtered images (as is the case in infants with blurred vision and with the built-in CSF-like LGN bottleneck). May be this issue should be further clarified.

### Questions
(a) Regarding the notation question, consider that \sigma is used for different things: the blur extent, the soft-max function, and the noise used to initialize the all-pass filter to be optimized. Moreover, in Eq.1, I understand that b_{\sigma} is (or should be?) the filter, G_{\theta}, -as in Eq.3-, no?. Additionally, the parameters of b_{\sigma} are H_{\theta}?... is this H an entropy? or is it a generic way to call the parameters? (H is certainly a cross-entropy in Eq.2, but H in Eq.1 is confusing -to me-). I get the ideas and they are fine, but notation is confusing (to me).   

(b) Regarding related work, Li et al. 22 ( https://jov.arvojournals.org/article.aspx?articleid=2778843 ) and Akbarinia et al. 23 ( https://doi.org/10.1016/j.neunet.2023.04.032 ) show that this kind of bottleneck can also emerge when solving certain tasks implemented by certain (simple) architectures, but does not emerge in other (too deep) architectures.

(c) Please clarify the discussion about whether the low-pass filtering is (or is not) required in the training of the models to increase the alignment with humans.

(d) Incidentally (but this is a criticism/question to Jaini at al. 23 not to this work ;-) I think that the increased shape-bias found in their generative classification may not be surprising because the noise added to launch the inverse in the difussion process in Jaini et al. destroys the texture in the image, so naturally, the classification (finally done in the spatial domain) has to be more driven by shape. Could the authors elaborate on this?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper claims that recent improvements in human–model agreement for vision systems mostly come from an overlooked preprocessing step that removes high-frequency detail. The authors show that applying a simple low-pass filter to images at test time makes standard discriminative models behave more like humans on a benchmark of human vs model errors. They confirm this in three ways. First, simple blur or downsampling increases human-like behavior without retraining. Second, a learned frequency filter that is optimized for alignment ends up looking like a low-pass filter. Third, a filter shaped by the human contrast sensitivity function produces similar gains, which links the effect to known properties of early vision. The paper also maps the trade-off between accuracy and alignment to clarify what levels of agreement are realistically achievable.

### Strengths
- The writing is clear, the narrative is well structured, figures effectively support the claims. 
- The paper offers a simple, general, physiologically grounded intervention with immediate practical impact: prepend a low-pass filter at test time to improve human alignment without retraining.
- Reinterprets prior SOTA claims by pinning gains to an overlooked preprocessing step. The Pareto-frontier view of MvH is insightful.
- Broad cross-model evaluation; consistent trends; convergence of three lines of evidence

### Weaknesses
- Potential overfitting. The learned Fourier filter is optimized on MvH without a held-out set. But it makes sense in the context.
- l.290: Please redefine EC ans SB because it requires coming back to previous section to find their meaning.

### Questions
- Do the gains persist on other human datasets or tasks (e.g., free-viewing, longer exposures)?
- If you re-fit the Fourier filter on a subset of MvH and evaluate on held-out conditions, does it still outperform simple Gaussian/CSF filters?


In general, I enjoyed reading the paper, which was well written and interesting. I would strongly encourage acceptance once my minor comments have been answered.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the source of human-like behavior in modern vision models. Building on prior work suggesting that generative objectives (e.g., Imagen) yield superior behavioral alignment with humans, the authors demonstrate instead that the key factor is low-pass filtering, specifically, the effective downsampling to 64×64 resolution used in Imagen. Through extensive experiments across multiple architectures (ResNet, BiT-M, EfficientNet, OpenCLIP ViTs), they show that removing high-frequency content at test time (via Gaussian blur, resizing, or learned Fourier filtering) consistently: a) Raises shape bias (SB) toward human levels; b) Boosts error consistency (EC) beyond Imagen’s previous record (κ = 0.37 vs. 0.31); c) Only modestly reduces OOD accuracy (≈3–6 p.p.), revealing a clear accuracy–consistency trade-off curve.

The authors also derive a learned Fourier filter that optimizes EC directly, finding it converges to a low-pass profile nearly identical to the analytical Gaussian filter. Finally, they analyze the model-vs-human benchmark’s Pareto frontier, establishing the theoretical ceiling for behavioral alignment and showing that current models are still below this limit.

### Strengths
The paper is well-organized and well-written, with smooth narrative flow from hypothesis to methods to results. I think the paper provides a simple yet powerful reinterpretation of why generative vision models appear more human-like,  reframing a potentially deep theoretical question (generative vs. discriminative objectives) as a signal-processing issue (frequency content). As well, The authors test across numerous architectures, including large-scale CLIP models, and show strong generalization of the low-pass effect. I think they do a great job by providing a theoretical  link to the contrast sensitivity function (CSF) of human vision provides a compelling neurophysiological explanation, integrating psychophysical and computational perspectives.

### Weaknesses
The psychophysical alignment argument can  be strengthen  by testing on neural benchmarks (e.g., Brain-Score, Algonauts) to verify that low-pass filtering also aligns representational geometry with biological vision.

Also a control, may include testing under longer exposure times or more naturalistic conditions which could  clarify whether the low-pass benefit is tied to the MvH’s 200 ms presentation constraint or generalizes across temporal regimes.

While the learned Fourier filter converges to a low-pass profile, visualizing intermediate optimization stages could reveal whether specific spatial frequencies carry distinctive importance for behavioral alignment.

### Questions
Have you tested whether low-pass filtering also improves alignment on neural datasets (e.g., Brain-Score or fMRI RDMs)?

How sensitive is the optimal σ to model scale and patch size?

Could low-pass filtering be integrated during training to recover the lost OOD accuracy?

Is the Pareto frontier analysis robust to the noisy EC estimates reported by Klein et al. (2025)?

### Soundness
3

### Presentation
3

### Contribution
3
