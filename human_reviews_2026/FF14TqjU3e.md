# Faithfulness Under the Distribution: A New Look at Attribution Evaluation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Evaluating the faithfulness of attribution methods remains an open challenge. Standard metrics such as Insertion and Deletion Scores rely on heuristic input perturbations (e.g., zeroing pixels), which often push samples out of the data distribution (OOD). This can distort model behavior and lead to unreliable evaluations. We propose FUD, a novel evaluation framework that reconstructs masked regions using score-based diffusion models to produce in-distribution, semantically coherent inputs. This distribution-aware approach avoids the common pitfalls of existing Attribution Evaluation Methods (AEMs) and yields assessments that more accurately reflect attribution faithfulness. Experiments across models show that FUD produces significantly different—and more reliable—judgments than prior approaches. Our implementation is available at: https://github.com/LMBTough/FUD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a key limitation in evaluating attribution quality, which is the reliance on a baseline image in perturbation-based tests. Such tests modify the input image and measure the model’s response to infer the importance of different features. However, the choice of baseline image critically affects the results. Prior studies have shown that commonly used baselines, such as white, random, or black images, are often out-of-distribution (OOD) with respect to the evaluated model. As a result, the attribution quality becomes dependent on the model’s behavior under OOD conditions, which is not a meaningful measure. To address this issue, the paper introduces an optimization framework that formulates OOD deviation as a loss and adjusts the baseline to be in-distribution while ensuring that the inserted pixels do not encode features from other classes.

### Strengths
1) The paper describes and addresses important problems for the evaluation of attribution methods. 
2)  The algorithm and evaluation shows that the perturbed images are in-distribution.

### Weaknesses
1) The 2.3 Section is not well structured. It’s several paragraphs of text and heavy math with no clear organization describing the overall approach. 
    2) Previous work select a baseline image I and replace perturbed pixels from the baseline image. This work introduces an algorithm A that generates the perturbed pixels. Both methods introduce a dependence on the “score” on I or A. The paper should formally define what properties of a modified image are desirable upfront. In-distribution seems to be one such property. However, the visual similarity that is used in the evaluation seems to be presented rather ad-hoc. Are there other properties? 
   3) Overall, it would be good to state upfront what is the metric that is being optimized and why. This could help provide some structure to the FUD methodology section. 
  4) How does the proposed methods impact the very popular insertion/deletion scores? No such results are reported. 

Minor:
   1) Why 198 runs? Typically, we select even numbers like 100,200, or 500.

### Questions
1) How should we formally measure the quality of a perturbed image? What is the justification for this metric.
2) It's not clear how much the perturbation method modifies the image to make it in distribution. Can we measure the L2 norm or other relevant metric? What is the starting point for the optimization process?
3) IG defines conditions that a baseline image should satisfy. This method breaks there conditions. Does that make the axioms of IG invalid?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a solution to the longstanding baseline selection problem which exists with input perturbation-based attribution evaluation metrics. They propose to fix the two side effects of: OOD inputs and failed feature removal by performing a deletion metric that employs a score-based diffusion model to produce ID inputs that ensure feature removal. The result is an improved quantitative evaluation of attributions.

### Strengths
They highlight a clear and pervasive issue that has yet to be solved in the attribution field. This is not the first method to attempt to solve this problem with generative model infilling [1, 2], but it is the first metric to employ this operation, and it adds new layers of theory, which appears to improve its solution. 

The two tasks of the diffusion process: staying in distribution and not producing class-relevant features is a strong formulation. I have not seen the former issue addressed before.  

This is a strong alternative to approaches that aim to solve OOD by training the model on ablated inputs.  

[1] Explaining Image Classifiers by Counterfactual Generation 

[2] Explaining image classifiers by removing input features using generative models

### Weaknesses
Crucial references missing [1, 2]. They do not acknowledge that this is a process that has been used heavily in perturbation methods. Metrics are different, and their use is better motivated and better backed, but these papers must be mentioned and addressed.  

The computational overhead of FUD is surely significant. Not just the original training of the diffusion model on the dataset, but the employment of diffusion model inference at 11 perturbation steps for any evaluated attribution/image pair. Evaluation is not a run-time activity, so this is okay, but I would’ve liked to see an evaluation to understand the significance of the runtime penalty. 

It is hard to tell the impact of this metric past the theoretical improvements that it shows. It does surely improve the baseline selection problem, but how does doing so actually change what we know about the best attribution methods? It would have been nice to see if there are improvements in metric IRR or ICR [3] or in the rankings of current SOTA attributions.  

Section 2.3 is challenging to read to the extent that it takes away from the paper. While the notation appears correct, the formatting of the section leads to many things blending together and it becomes overly complicated to interpret. It would be good to separate intuition from derivation in this section to improve readability, accessibility, and reproducibility.  

[3] Sanity checks for saliency metrics

### Questions
Why was previous work which employed infilling or diffusion for attribution methods ignored? 

Otherwise, please see the weaknesses.  

My rating could be improved to an 8 if this question and the weaknesses are properly addressed. I have major concerns about the lack of references to relevant work, but I think the authors could clear this up with a proper reply and reasoning for the omission.

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
2

### Summary
Current methods for removing input feature contribution for measuring attribution method faithfulness do not truly remove input feature contributions. The authors introduce a new attribution measuring metric: Faithfulness Under the Distribution (FUD). FUD follows other metrics by inserting/deleting some number of pixels. However, instead of just blacking out the removed pixels, FUD uses a diffusion model to generate the image gradients, which are then used to determine the pixel values for the masked region. The FUD process also ensures that the generated pixels stay within the original data distribution of the non-perturbed image, keeping the image functionally the same for the next metric step. The authors provide comparisons against other metrics in terms of the intermediate images generated and some qualitative results.

### Strengths
- Originality: The authors use part of a previous approach for the metric, but significantly build off of it. (3/4)
- Quality / Clarity: Paper presentation is high quality. The motivation and experiments are easy to follow, but the methodology is not. (3/4)
- Significance: The authors have come up with a method for measuring attribution methods without introducing a significant amount of noise. (4/4)

Other Notes:
- Very strong introduction and motivation. Easy to follow and understand why/where the problems exist. 
- Framing the issue as an in-/out-of-distribution problem is particularly clever
- The approach tries to steer the masked regions towards an alignment with the unmasked regions, thereby mitigating negative effects of masking, which is a novel issue that has not been explored
- The experimental evaluation suite is wide

### Weaknesses
- There must be a significant runtime for these evaluations because diffusion models are computationally expensive. However, attributions are scored offline, so it is not a significant issue.

### Questions
- Can the authors provide any instances in which the metric performs worse than other metrics, or any kind of failure case analysis? Relying on a diffusion models seems like it should cause problems somewhere, even if it is in some small corner cases.
- How are pixels chosen for their inclusion in the image mask? Is it the same way as insertion/deletion?
- Could the authors re-explain the necessity of the inclusion of z? Is it to ensure that when gradient steps are taken, the steps are updating the masked region in such a way that the new pixels are aligned with the unmasked pixel class while not adding new features? But how does this ensure that new features are not added? 
- What is the relation of equation 2 to s_{theta}(x^t)? Is there any relation? Or is it used during the z "event"?
- In plain terms, can the authors explain the transition between x^t --> x^{t+1}? I am able to follow the the very high level idea of what is trying to be accomplished but some of the specifics of equations 1 and 2 are lost on me. 
- Is P(x) the one-hot encoded ground truth label for some image? Is \theta^* the parameters for the diffusion model in use? Are the gradients being referred to the image gradients or model gradients? How is equation 2 being used?

Final Review:
The paper proposes a very interesting solution to the issue of assessing attribution performance. I am able to follow at a high level what is trying to be accomplished, but the exact math is lost on me, hence the (2/5) confidence rating. Given this state, I rate the paper a (6/10). Based on my understanding of the high-level idea, I think it should be accepted, but I am not confident enough in my understanding of the math to rate it higher.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces FUD, a new evaluation framework for attribution methods (xAI)  that uses score-based diffusion to generate in-distribution masked samples.

### Strengths
- Strong results on multiple benchmarks
- Thorough method explanation and justification
- Good reproducibility, clear pseudocode + released configs/code

### Weaknesses
- Hard to follow. Related Work and key visual examples are in the appendix; in my opinion, they should be in the main paper. Also, I believe Tables 8 and 9 are very valuable for readers.

- Wording like “universally” is too strong (`Line 097`). The heuristic issues are already known and not universal across all AEMs. 

- Figure 4: At a glance, FUD’s results look similar to simple black/white occlusions -> the very issue the paper critiques (main motivation e.g. OOD samples)

### Questions
- Please refer to the weaknesses section. 

- Additionally, it would be helpful to provide an experiment on non-visual data, for example, on tabular data.

### Soundness
3

### Presentation
2

### Contribution
3
