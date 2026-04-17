# Code2Video: A Code-centric Paradigm for Educational Video Generation

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
While recent generative models advance pixel-space video synthesis, they remain limited in producing professional educational videos, which demand disciplinary knowledge, precise visual structures, and coherent transitions, limiting their applicability in educational scenarios. Intuitively, such requirements are better addressed through the manipulation of a renderable environment, which can be explicitly controlled via logical commands (*e.g.*, code). In this work, we propose **Code2Video**, a code-centric agent framework for generating educational videos via executable Python code. The framework comprises three collaborative agents: *(i) Planner*, which structures lecture content into temporally coherent flows and prepares corresponding visual assets; *(ii) Coder*, which converts structured instructions into executable Python codes while incorporating scope-guided auto-fix to enhance efficiency; and *(iii) Critic*, which leverages vision-language models (VLM) with visual anchor prompts to refine spatial layout and ensure clarity. To support systematic evaluation, we build **MMMC**, a benchmark of professionally produced, discipline-specific educational videos. We evaluate MMMC across diverse dimensions, including VLM-as-a-Judge aesthetic scores, code efficiency, and particularly, **TeachQuiz**, a novel end-to-end metric that quantifies how well a VLM, after unlearning, can recover knowledge by watching the generated videos. Our results demonstrate the potential of Code2Video as a scalable, interpretable, and controllable approach, achieving 40% improvement over direct code generation and producing videos comparable to human-crafted tutorials.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on generation of educational videos via using a code-centric paradigm. The paper claims that by using pixel based methods, the results are not robust, explainable, and require a large amount of pre-training data. Whereas using a code-centric paradigm solves these issues and allow for longer and better videos. Accordingly, a new benchmark dataset is curated which consists of videos from the 3Blue1Brown YouTube channel with a corresponding novel evaluation scheme which utilises the knowledge that educational videos should not just be visually correct but represent good teaching resources is also proposed. A new method, named Code2Video, is presented which incorporates a planner, coder, and critic components to render an educational video via writing of code. Results show that pixel based methods indeed fail at the task and the proposed method outperforms naive LLM usage. A human study further verifies the claims.

### Strengths
* The idea of using code-centric generation for educational videos is a nice contribution and is clearly validated when comparing to the pixel based methods in the results both quantitatively and qualitatively.
* The reasoning behind why new evaluation metrics are needed and that the previous metrics which only capture visual fidelity is strong.
* The results show the benefits of the proposed method, whilst there is a tradeoff between the proposed method and the time and token requirements from an efficiency point of view, the results more than make up for it when compared to the Code LLM (i.e. baseline) methods with large improvements when the Code2VideoAgent is used.

### Weaknesses
## Weaknesses
* Has the owner of the 3Blue1Brown YouTube channel been contacted and informed that their videos have been used to construct a dataset for research? Have they licensed their use? Upon looking on their website (https://www.3blue1brown.com/faq#licensing) the use for research/AI is not specified as one of the use cases that do not require explicit permission from the owner.

* Sections 3 and 4 are missing a lot of details, especially within the main paper, and are very hard to read. Some of the information is given in the appendix, but this is not linked to and is disparate enough that it is hard to understand what the details of the paper are. Examples include:
  * $\mathcal{P}$ has not been defined within the main paper, there are links to the appendix but these do not specify what these are and how they are utilised in the main paper.
  * The reasoning behind how aesthetics, knowledge convey, and efficiency map to the evaluation metrics given is not clear and not well argued.
  * Figure 3 has not been referenced within the text of the paper.
  * $S$ and $\tilde{S}$ are not defined within the method section.
  * On line 293 it says that generation of manim code can exceed two hours, is this using the method as described without parallelising it?
  * Figure 4 is not clear from the accompanying text, for example, the effective debugging section does not seem to match well with figure 4 middle bottom, which showcases the error and then two scopes for the code with arrows.
* As far as I can tell, the method to cause the model to forget certain knowledge is just apply a prompt and does not require any fine-tuning which does not seem to be a principled way of evaluating a model in this respect. This is expanded on very briefly in the appendix, but I am not convinced that the VLMs will not be biased in some way/unlearning the data. 
* Numbers within Table 1 and Table 4 are not bolded consistently throughout the table (i.e. only Avg and Quiz are bolded in Table 1), it's nto clear why.
* The correlation values given on line 471, it sounds as if these are checking the correlation of human participants' scores on the aesthetics and the quiz scores whereas a separate correlation value could be calculated between the human study scores and the VLM as judge scores within Table 1, was this calculated?
* The discussion regarding human attention and patience is interesting, and could be expanded to include the results presented in table 4. For example, the human made videos score worse for both quiz and completion willingness (the latter by quite a large margin) is this purely because of the length of the human-made video or is there something else going on here?
* There are a lot of spelling and grammar mistakes within the paper and it could benefit from another round of proof-reading and corrections. A incomplete list is given below.


## Corrections
* Line 174: We source from the complete 3Blue1Brown (3B1B) YouTube corpus. This implies that there is a completed corpus and not videos from a channel? If the former there's no reference.
* Figure 2 (1) is quite hard to read at the scale
* consider knowledge conveyance instead of knowledge convey
* Line 241: "are already be learn" -> "have already been learned" or similar
* line 242: missing space between i.e., and answer.
* Line 259: "which is consists of three stages" -> "which consists of three states"

### Questions
1. Has the owner of the 3Blue1Brown been contacted and informed that their videos have been used to construct a dataset for research?
2. Is there a reference or principled explanation for how the knowledge unlearning step can be used to guarantee that the model is in fact not using specific concepts?
3. Has a correlation between the proposed metrics utilising the VLM as judge and the results of the human study been conducted?
4. Are the low scores for the human videos in Table 4 purely because of the length?
5. generation of manim code can exceed two hours, is this using the method as described without parallelising it?

### Soundness
2

### Presentation
1

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
The paper proposes a pipeline for generating educational videos using the Manim software, built on top of VLMs. It also introduces MMMC, a benchmark designed to assess the effectiveness of the proposed pipeline.

### Strengths
- The paper is well organized, with clear and visually rich figures, making it easy to read and follow.
- The paper explores an understudied area in education, which adds meaningful community value to the research.

### Weaknesses
- The paper lacks novelty. It mainly presents a code-generation pipeline without introducing any fundamentally new algorithmic ideas. Most components of the work are combinations of existing (V)LLM techniques, rather than newly developed methods. As such, the contribution resembles more of a workshop or community project rather than a work suitable for a serious machine learning conference.
- The expression is inappropriate. The paper emphasizes that a code-centric approach can produce better videos than generative model–based methods, which is somewhat inappropriate: as is well known, generative models and virtual engines represent fundamentally different paradigms of visual content creation. The paper should not treat generative models as a central motivation and baseline (e.g., Fig. 1 and Tab. 1), nor should it refer to the proposed approach as 'video generation' (e.g., line 99), a term that conventionally refers to generative model-based synthesis.

### Questions
See the weaknesses section.

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
This paper presents Code2Video, a framework for generating educational videos by adopting a "code-centric paradigm" instead of traditional pixel-based synthesis. The authors argue that for structured educational content, generating executable Python code (specifically for the Manim library) provides greater controllability, interpretability, and scalability. The method employs a tri-agent pipeline: a Planner to create a storyboard, a Coder to write and debug Manim code using a "ScopeRefine" strategy, and a Critic to refine the spatial layout using a VLM and a "Visual Anchor" grid system. To evaluate this, the paper introduces the MMMC benchmark, sourced from 3Blue1Brown's YouTube channel , and a novel metric, TeachQuiz, which assesses knowledge transfer by using a prompt-based "unlearning" protocol on a VLM before showing it the generated video.

### Strengths
1. Problem Identification: The paper correctly identifies a significant limitation of current pixel-based video generation models, which struggle with the logical coherence, precise layout, and domain-specific knowledge required for professional educational videos.

2. Metric Conceptualization: The idea behind the TeachQuiz metric is commendable. Attempting to move beyond simple aesthetic scores to measure actual knowledge transfer is a crucial direction for this field. The "unlearn-relearn" concept  is a novel approach to addressing the confound of prior knowledge in VLMs.

3. Modular Architecture: The proposed tri-agent framework (Planner, Coder, Critic) is logical, modular, and clearly structured, with each component addressing a distinct part of the video creation pipeline.

### Weaknesses
1. The paper's central claim of introducing a "new paradigm" is a significant overstatement. Generating educational videos from Manim code is the standard, existing workflow for creators like 3Blue1Brown, the paper's own data source. The paper's actual contribution is the automation of this existing workflow using an LLM-based agentic pipeline. This is more of an application of the well-established "coding agent" paradigm to a specific library (Manim) rather than a fundamentally new paradigm for video generation.

2. TeachQuiz is Unverified: The paper's primary metric, TeachQuiz, lacks experimental robustness. It relies entirely on a prompt-based "unlearning" protocol ($\mathcal{P}_{unlearn}$) applied to a closed-source VLM (Gemini-2.5 Pro). There is no validation that this prompt induces "genuine unlearning" as claimed. It is equally, if not more, likely that the VLM is simply adhering to a complex negative constraint ("do not use knowledge X to answer") rather than truly "forgetting" the concept. The paper's headline claim of a 40% improvement rests entirely on this fragile, unverified, and opaque prompt-engineering artifact.

3. Benchmark is a Monoculture: The experiments are not thorough enough to support the paper's broad claims. The MMMC benchmark is a monoculture, sourced exclusively from 3Blue1Brown. This means the system is only evaluated on a single, niche aesthetic (Manim, dark background, math topics). There is zero evidence of generalization to any other style of educational video (e.g., whiteboard animations, slide-based lectures, software tutorials) or other domains (e.g., history, literature, biology).

4. Baselines are Non-Competitive: The comparison to pixel-based models (e.g., Veo3) is a strawman. It is widely known that these models cannot render coherent text or perform complex, long-form logical reasoning. A more thorough and meaningful comparison would have been against other state-of-the-art agentic coding frameworks, tasked with the same goal of writing Manim code.

5. Prohibitive Cost: The method's feasibility is highly questionable. The paper's own data (Table 1) shows generation times of 15-43 minutes and token consumption up to 49.2K tokens per video. This directly contradicts the claims of being "scalable" and efficient. These costs make the system impractical for "interactive educational settings"

6. VLM-Critic is Flawed: The feasibility of the Critic agent is fundamentally undermined by the paper's own human study. The study explicitly states that humans are "highly sensitive" to layout errors that the VLM-as-a-Judge (and thus the VLM-Critic) "often miss". This means the Critic is optimizing for a flawed, non-human-aligned objective. This misalignment makes it incapable of achieving the "professional" quality it claims to target.

### Questions
1. On Novelty: Given that generating video from Manim code is an existing human workflow, and LLM-based coding agents are an established research area, could the authors please clarify what precisely is the fundamental novelty of this work beyond applying existing agent techniques to the Manim library?

2. On TeachQuiz: How can the authors provide any empirical guarantee that the prompt-based $\mathcal{P}_{unlearn}$ protocol results in "genuine unlearning" rather than simple instruction-following? Without this validation, how can the TeachQuiz scores, and the 40% improvement claim, be considered reliable?

3. On Feasibility: How does the team reconcile the claim of "scalability" with generation times of up to 43 minutes and ~50K token costs for a single short video?

4. On the Critic: If the human study confirms that the VLM-Critic "misses" layout issues that humans find critical, does this not invalidate the Critic as a viable path to achieving "professional" quality? Why should we trust an automated critic that is demonstrably misaligned with human perception?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a code-centric agent framework for generating educational videos via executable Python code.

### Strengths
1. The paper is well organized.
2. This paper introduces a code-centric paradigm for educational video generation, positioning executable code as the unifying medium for temporal sequencing and spatial organization. This seems to be able to generate longer videos than Veo3.

### Weaknesses
1. The paper's task seems to be quite similar to PPT generation. Could you discuss the differences between them?
2. Where did you collect your data from?
3. The generated video is still not long enough; real educational videos are usually over 30 minutes.
4. Large language models suffer from hallucinations, and the knowledge they provide may not necessarily be accurate, which is a fatal weakness in teaching.
5. In Table 1, the author's method is six times slower than Veo3. Generating a 40-minute educational video is estimated to take 5 days, which makes the efficiency unacceptable.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
3
