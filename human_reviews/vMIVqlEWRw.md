# Robin: a Suite of Multi-Scale Vision-Language Models and the CHIRP Evaluation Benchmark

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
The proliferation of Vision-Language Models (VLMs) in the past several years calls for rigorous and comprehensive evaluation methods and benchmarks. This work analyzes existing VLM evaluation techniques, including automated metrics, AI-based assessments, and human evaluations across diverse tasks.  We first introduce Robin - a novel suite of VLMs that we built by combining Large Language Models (LLMs) and Vision Encoders (VEs) at multiple scales, and use Robin to identify shortcomings of current evaluation approaches across scales. Next, to overcome the identified limitations, we introduce CHIRP -  a new long form response benchmark we developed for more robust and complete VLM evaluation. We provide open access to the Robin training code, model suite, and CHIRP benchmark to promote reproducibility and advance VLM research.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper is of two parts. The first part trains a suite of Visual Language models combining LLM (Pythia) with a vision encoder (CLIP) across different scales. The major claim is that there is a slight relationship between LLM size and model performance while there is no clear relationship between vision encoder size and model performance. The paper then looks at different nuisance factors in terms of model answering: long form vs short form, wrong formatting and multiple possibilities for ground truth.

The second part introduces a new VLM benchmark (Robin) based on human + LLM captions and DALLe-3 based images. They perform pairwise comparisons of different models based on human preferences, GPT-4v and LLaVa. They notice that only AI evaluators correlate performance with VE size but not GPT-4v and LLaVa. The paper then studies properties such as agreement and contradictions between human raters and models.

### Strengths
The overall idea of careful comparisons of VLM’s and introducing benchmarks that can capture VLM nuances that current models cannot is quite interesting.

### Weaknesses
The reviewer was not able to find enough justification based on the experiments that current downstream benchmarks cannot capture scaling behaviour. This was based on both the pretraining loss plots (Fig 1) and experiments retaining the same datasets but with different metrics (Fig 4). 
Finally, the last paper feels more like a study evaluating multiple Robin models on the CHIRP benchmark. There are, however, no concrete recommendations on how a researcher can easily evaluate newly trained VLM models on these benchmarks and what nuances these benchmarks can capture as opposed to others.
See below for detailed feedback.

### Questions
Questions and detailed comments:

* The paper reports per-datasets results which is quite far-away from SoTa. For example, compare Table 5 of this paper with the published LLaVa results (See Table 3 and Table 4). This suggests that the models in Figure 1 are not optimally trained. Hence it is quite difficult to trust the results from Figure 1.
* The paper states that roughly 9% of GQA and 5% of TextVQA is incorrect and says that 3% difference in accuracy are noise. But all models are likely to get these questions wrong, if the metric in consideration is a perfect match between the generation output and ground truth. Thus this only tells us the upper bound of accuracy a model can achieve. 
* Figure 2 shows that bigger vision encoders do not lead to better performance on benchmarks. Based on this result, the paper hypothesizes that there are differences between the different vision encoder sizes but the benchmarks are unable to capture it. But Figure 1 shows that this property is also seen during pretraining with the pretraining loss. So based on Figure 1,  one cannot attribute the observation that “bigger vision encoders do not lead to better performance” solely to benchmarks.
* From Figure 4 left column, even though the automated evaluation is much lower than the human evaluation, they follow the same trend, that is they are correlated. If anything, this plot suggests that automated evaluation is as good as human evaluation in assessing model trends. Similar things can be said while comparing LLM-based evaluation and automated evaluation in the second plot from the left.
* The human grading plot is omitted in Figure 4, except for the first column. So one cannot assess how the human evaluations compare to automated evaluations when encoder size is varied.
* Figure 4 right shows that VLM’s predict that close to 100% accuracy while according to humans, the upper bound is more like 50%. This discrepancy indicates that the VLMs, at least within the context of the authors' methodology, are not suitable for use as an oracle for this particular task.
* The authors present a new benchmark called CHIRP. But I was not able to find concrete recommendations on how they plan to allow researchers to evaluate new models using the benchmark since all the reported results are in ELO score and require pairwise comparisons.
Given the results from Figure 4 right, the reader also becomes unsure about the efficacy of the GPT-4V and LLaVa based results in this section.
* I was not able to find a concrete explanation on why the CHIRP benchmark is better than other benchmarks. Comparing Fig 6 left Survey and Fig 4 left Human grading, both capture scaling properties. Similarly, Fig 6 second column and Fig 4 second column, seem to also capture scaling properties with vision encoder size. Thus more justification is required for the CHIRP benchmark.
* Table 2 shows that GPT-4V decision making is the most correlated with humans. However, Table 3 shows that GPT-4V is the most different from humans while comparing model size agreement by method. How is this the case?
* L500: Ultimately, both human and AI evaluations show that performance on CHIRP correlates with loss more than other evaluation tasks. The paper states this but unfortunately no evidence is provided.

Minor comments

* Are the plots in Figure 1 from the first stage or the second stage of LLaVa pretraining?.
* Please decompose Figure 3 into separate plots. It is super difficult to read.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes one major contribution: a dataset for the evaluation of VLMs, named CHIRP, focused on open-ended Q&A. The main goal of CHIRP is to solve the inconsistencies emerging in the analysis proposed between the long-form and short-form answers produced by a VLM in a Q&A task. More in detail, in the paper it is stated that VLMs have significant differences if 1) prompted to reply shortly to an input question requiring visual interpretation, or 2) left free of reasoning and provide a question as long as they want. A long preliminary part of the paper focuses on the presentation of Robin, a collection of VLM based on LLAVA that focuses on the ablation of the model sizes and the visual encoder. The paper's organization is as follows: first, Robin is presented and results are evaluated on existing benchmarks. Then, shortcomings of existing benchmarks are proposed. Finally, CHIRP is introduced and used for the evaluation.

### Strengths
1. There is some work behind this manuscript. I believe that the authors have contributed with a non-trivial effort in this paper, since many parts of the proposed analysis require effort, such as the training of the VLMs, and the design of the benchmark questions.  

2. The theme treated is important. It is challenging to find the perfect way to evaluate a VLM, and I believe that the focus on the effectiveness of different ways of prompting the model is interesting.

### Weaknesses
1. My biggest problem with this manuscript is that I believe that is not enough general in the observations that led to the analysis and to the introduction of the benchmark, which makes the entire analysis limited in scope. To clarify, the results are only on the proposed suite of VLM and motivated only by the performance on the proposed suite of VLMs. Although it is based on existing models, we have no guarantees that what is going on in the paper is an actual problem for VLM or it is just a consequence of the design choices of the authors. Instead of introducing a suite of foundation models for highlighting a problem, and then evaluating the proposed dataset on the trained models, I think it would have been more scientifically solid to test how existing VLMs (such as LLAVA, Chameleon, etc) would perform with LvSR and then see if the problem stands. Right now, the entire work seems designed for a problem that we do not know it exists for widely used networks, and it could be constrained to the authors' setup.
2. Similarly, how do existing models perform on CHIRP? We cannot judge one benchmark if the adequate benchmarking of existing VLMs is not performed, and does not reveal novel insights on the capabilities and limitations of existing models.
3. The motivation behind the benchmark is not very clear to me. While the authors claim that there are shortcomings with the evaluation with short answers or single words, which I agree with, there is an extensive literature on LLM evaluators [1], partially cited in the paper, which is taking into account other aspects rather than simply using exact string matching. Why the CHIRP dataset is different than just using LLM evaluators on the replies of the LLM for existing questions in specific benchmarks for some of the categories introduced, such as [2]? Additionally, why this should help in solving the problems with LvSR?
4. The presentation of the manuscript is very convoluted. Presenting first the suite of language models, then the problems of the benchmark, and finally the benchmark, makes the reading quite challenging and it is not clear what's the main point that the authors want to make. Is it related to the problems in the ground truth in existing benchmarks? Is it related to LvSR? Is it saying that the vision encoder does not impact much in the learning of the network?

[1] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS 2023  
[2] VIVA: A Benchmark for Vision-Grounded Decision-Making with Human Values, EMNLP 2024

### Questions
1. What is the main scientific finding that this manuscript wants to communicate to the community? What added value has the scale of the model and of the visual encoder?
2. Why not performing a comparison with existing networks / on existing benchmarks?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper studies the correlation between VLLM log-likelihood, popular benchmark performance and the human-perceived model quality. By studying the scaling ladder of VLLMs (changing LLM and vision encoder sizes) the authors show that some trends (e.g model quality improvement with larger vision encoders) are not clearly discernible using existing benchmarks, and propose a new benchmark called CHIRP along with a pair-wise model comparison questioner that is able to highlight these trends better.

### Strengths
* The paper is rife with interesting remarks and observations on benchmarking and evaluation of VLLMs. In a time of explosion of VLLM-related research, these observation could have a particularly significant impact on the research community.
* Benchmark-related aspects of the work appear sound.
* The work is well-presented; the authors take the reader of a bit of journey through their thought process, introducing various hypotheses and explaining how they tested them.
* The presented work paves the road towards better benchmarking of VLLMs

### Weaknesses
**Major**
* My main concern about the paper lies with its experimental (model training) part. The graphs in Figure 1 are a little bit suspicious as they do not demonstrate a monotonic increase in likelihood as model size grows. This hints at some training instability or other issue (this is also my interpretation of some of the outliers in Figure 10 in the Appendix). Upon reviewing the supplement I see that the authors used the same hyper-parameters for all model (Section A2). This is likely suboptimal. Depending on how these hyper-parameters were chosen (i.e. for which VE + LLM pair size?), certain model quality trends may be masked or exacerbated.

I would like to see the authors demonstrate that their observations are robust to this choice of hyper-parameters. This can take the form of performing a larger hyper-parameter sweep for some of the outlier models (e.g. ViT-B and ViT-g vision encoders and various LLMs) and showing that the log-likelihood curves still demonstrate the above non-monotonic trends.

* Related to this, I was also surprised by the observation that there exists "ideal" VE-LLM size combinations (line 495), and suspect that this too could be a consequence of how model hyper-parameters were chosen.

* At a high-level, while find the paper rich with valuable observation and analyses, I missed the bigger picture. What is the authors' recommendation for VLLM benchmarking given all that they've found? Should we be training largest models if we aim to improve model quality (as perceived by human evaluators), can we rely on other existing VLLMs when computing Elo rankings on the CHIRP benchmark? Is CHIRP the best benchmark we have right now for measuring model quality improvements? Etc.

**Minor**
* Figure 5 does not show ELO curve samples with low transparency as the caption suggests.
* As I understood, the authors plan to release their trained VLLMs. In this case, it would be great to also see model cards detailing the data used for training the said models; as well as bias / toxicity analyses. While I do find this to be a major point, it's listed under "Minor" as I do not believe it should be blocking the paper publication and rather be a prerequisite for responsible model release.

### Questions
See Weaknesses.

In order to improve the score, I would like to see a convincing argument (probably supported by experimental results) indicating that there are no major issues with the training of the VLLM scaling ladder.

### Soundness
3

### Presentation
3

### Contribution
3
