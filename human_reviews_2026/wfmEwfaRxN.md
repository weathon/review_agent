# Frankentext: Stitching random text fragments into long-form narratives

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 4

## Abstract
We introduce Frankentexts, a long-form narrative generation paradigm that treats a large language model (LLM) as a composer of existing texts rather than as an author. Given a writing prompt and thousands of randomly sampled human-written snippets, the model is asked to produce a narrative under the extreme constraint that most tokens (e.g., 90%) must be copied verbatim from the provided paragraphs. This task is effectively intractable for humans: selecting and ordering snippets yields a combinatorial search space that an LLM implicitly explores, before minimally editing and stitching together selected fragments into a coherent long-form story. Despite the extreme challenge, we observe through extensive automatic and human evaluation that Frankentexts significantly improve over vanilla LLM generations in terms of writing quality, diversity, and originality, while remaining coherent and relevant to the prompt. Furthermore, Frankentexts pose a fundamental challenge to detectors of AI-generated text: 72% of Frankentexts produced by our best Gemini 2.5 Pro configuration are misclassified as human-written by Pangram, a state-of-the-art detector. Human annotators praise Frankentexts for their inventive premises, vivid descriptions, and dry humor; on the other hand, they identify issues with abrupt tonal shifts and uneven grammar across segments, particularly in longer pieces. The emergence of high-quality Frankentexts raises serious questions about authorship and copyright: when humans provide the raw materials and LLMs orchestrate them into new narratives, who truly owns the result?

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents, Frankentext, a method of generating narrative that composes existing texts in response to a prompt instead of generating new text from scratch. Their system retrieves relevant passages from existing stories in response to a writing prompt or assembles sections from randomly chosen story excerpts with minimal transition phrases and editing added. The system is instructed what percentage of generated tokens should be copied verbatim from the existing texts. While this method reduces the coherence of the outputs, the stories are judged to be more diverse and original than typical LLM prompting. The resulting generations are also much harder for AI-generated text detectors to detect.

### Strengths
Originality:
- I haven't seen work like this before and I think it is an interesting task, and useful data for AI vs. human text detection

Quality:
- This paper presents a novel task clearly and enough experiments to demonstrate why this task is interesting (generations are more creative than vanilla LLM prompting).

Clarity:
- The paper is very well-written and clear.

Significance:
- This work will further research in AI text detection and understanding weaknesses in the creativity of AI text generations relative to humans.

### Weaknesses
1. From reading example outputs, they seem significantly less coherent than vanilla LLM generations. I think this could have been discussed or addressed more in the paper. Are the writing scores improving primarily based on including creative wording from the human-written text even though the overall outputs are not as coherent?

2. The ethical concerns with this work could be emphasized more. An obvious use of this work is to copy significant portions of existing work and fly under the radar of AI text detection systems. In particular, I would like to see statistics on the average length of copied span from existing text.

3. Intuitively, it seems MCP should improve over using random snippets. I am confused by the description of why this is not the case given in section 4.4. After retrieving relevant snippets, shouldn't the final task be formatted with the same prompt to the LLM as when there are random snippets? Why does the model suddenly become more verbose in this context?

4. Section 4.6 feels a bit like an afterthought since the issues describe may just be with prompting. I would consider cutting this section or moving the details to the appendix.

5. As much as possible, results discussed in the main body of the paper should be moved into the main body of the paper (e.g., Table 11).

### Questions
I appreciate that you specify in the ethical considerations that you do not endorse using this data for model pretraining based on copyright issues, but at another point in the paper you encourage the use of training AI vs. human text authorship detectors using this data. Given that these detectors could also be for commercial use, can you clarify how you see this use aligning with copyright concerns?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces "Frankentexts," a novel paradigm for long-form narrative generation that reframes the role of a LLM from an author to a composer. The core task involves providing an LLM with a writing prompt and thousands of randomly sampled, human-written text fragments. The model is then constrained to produce a coherent narrative where a vast majority of the text (e.g., 90%) is copied verbatim from these fragments. The process involves the LLM implicitly searching the vast combinatorial space of snippets, creating an initial draft, and then iteratively "polishing" it for coherence while adhering to the copy constraint.

The authors conduct automatic and human evaluation on models such as Gemini 2.5 Pro, GPT-5, and Claude-4-Sonnet and they find that Frankentexts are often preferred by human ratings over unconstrained generation in terms of plot, creativity, and language use. The constrained text shows higher diversity, utility and surprise, but also lower relevance and especially coherence. They also conduct ablations, such as testing retrieval of relevant passages instead of adding thousands of random passages to the model, but this makes the model utilizing the retrieved passages less overall with no improvement in results. Finally, the authors note that this work poses a significant challenge to current AI detectors, since their composed texts cannot be detected as AI generated.

### Strengths
1. The authors have conducted thorough analysis on the benefits and shortcomings of their method via detailed human evaluation measuring different general-purpose and narrative-specific generation aspects. They also present several ablation studies to demonstrate the contribution of the different parts of their approach.
2. The paper poses an interesting question on AI text detection when LLMs are copying big portions from human written text.

### Weaknesses
1. The paper seems incomplete. The authors propose the idea of constraining LLM generation to mostly copying from human text (90% copying). First, this is not clearly motivated to me; what is the hypothesis when constraining generation in this way? More than that, the paper shows that this can produce interesting stories (but with lower coherence and grammatical issues, so there is a big trade off) only when conditioning on thousands of random passages. This is prohibitively expensive in order to be applied as a method for LLM generation, and when constraining the number of passages and applying retrieval of relevant documents instead which could be a more feasible solution, the authors show that this is not improving performance without any further insights.
2. Although the question of AI detectability in copied text is interesting, the authors do not really address this problem or investigate different detection methods. So, this aspect of the paper also is incomplete.

### Questions
1. What is the motivation/idea behind trying out this approach for generation? What is the hypothesis that you are trying to validate?
2. Have you done any analysis on why retrieving relevant passages is not helpful for the task? Does the LLM ignore the retrieved context (apart from not copying it over)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Frankentext, a novel text generation pipeline where large language models (LLMs) assemble long-form narratives primarily by reusing human-written snippets verbatim. The proposed approach encourages models to stitch authentic paragraphs together with minimal connective phrases, aiming to produce outputs that appear more human-like and creative while evading AI-generated text detectors. Experiments with several advanced reasoning models (e.g., Gemini 2.5 Pro, GPT-5) demonstrate consistent improvements in perceived creativity, novelty, and textual quality compared to vanilla generation.

### Strengths
**1. Fascinating idea and topic:**

The work explores a very interesting paradigm, reusing human-authored texts as-is to enhance perceived authenticity and creativity. 

**2. Strong empirical findings:**

The study shows that incorporating verbatim human text leads to outputs that evaluators and automatic metrics consider more creative and high-utility.

**3. Comprehensive evaluation:**

The experiments are extensive, involving multiple LLMs and consistent baseline comparisons.

### Weaknesses
**1. Limited topic relevance and coherence:**

The approach sometimes produces incoherent or context-mismatched passages. Since reward models such as WQRM or LLM-Judge can interpret any unusual text as “creative,” some outputs may be spuriously rewarded for randomness rather than genuine creativity.

**2. Complex and costly generation process:**

The method requires multiple inference rounds (i.e., drafting, revising, and polishing), which increases both computational cost and latency compared to ordinary text generation.

**3. Questionable definition of creativity:**

Although the resulting texts are evaluated as creative, they are not genuinely novel written materials (they primarily reassemble existing human). The framework therefore measures perceived creativity rather than true generative originality.

**4. Privacy and copyright concerns:**

Since the system directly retrieves and reproduces human texts from large corpora, it may expose copyrighted or private content. Ethical and legal implications should be discussed more deeply.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FrankenTexts, texts that are compiled by a LLM from several different human-authored texts. The authors show through human and automated evaluations that the resulting texts are in many respects better than vanilla LLM-authored texts with the same writing prompt and that AI detectors identify them as human-authored.

### Strengths
The paper is well-written and contains many interesting experiments. The human and automated evals are thorough and the figures and diagrams are well-designed.

### Weaknesses
This paper is very interesting, but feels like a FrankenText of multiple different papers. The paper can't seem to make up its mind if it's about a generation methodology for fiction or a critique of AI detection methodologies. Under both interpretations, key experiments are missing, and I feel like the paper would be significantly stronger if it focused on one or the other.

Under the interpretation that the primary interest is in using this to write fiction, the paper does provide evidence that Frankentexts are preferred to vanilla texts, but that's not surprising to me (contra the authors' intuition in line 55). LLMs are generally regarded as worse writers than good humans, and by asking the model to copy as much as possible verbatim from published books the authors ensure a significantly higher writing quality than what a model would be able to do on their own. While I am glad that the authors compare to a vanilla baseline, it seems highly relevant to compare to the baseline of retrieval augmented generation where the model retrieves the same texts but isn't instructed to copy them verbatim. I'm not an expert in LLM-powered fiction writing, but whatever the current state-of-the-art in that area should be compared to as well under this interpretation. Furthermore, regardless of what one thinks of the ethical status of LLM generations in general, using this methodology to generate fiction is unambiguously plagiarism. The authors don't seem to acknowledge this fact anywhere.

Under the interpretation that the primary interest is in critiquing AI detectors, the authors fail to compare with the retrieval baseline again, as well as light human editing or two models editing each others' work. It's also unclear whether this really works as a critique of AI detectors at all given that 90%+ of the text is genuinely human-authored. At no point do the authors engage with the question of whether it is correct to view these FrankenTexts as AI-authors or human-authored. They also fail to engage in discussions related to the reprecusions and implications of the work that I would expect of a critique of AI detectors. Finally, although AI-detectors have had success marketing themselves as commercial tools, I do not know any AI researchers who seriously believe that they are reliable in the face of adversarial generations.

### Questions
1. Do you recommend this methodology as a way to generate fictional texts? If so, what do you say to the contention that this is plagiarism?
2. If 90% of a text is human-authored, why do you think it should be considered an AI-authored text?
3. Is there a meaningful difference in the distributions of texts that result from asking a human to provide the same task with the same sources?
4. What do these results mean, at the end of the day? What should I take from this paper off into my life? It feels like this paper is "just asking questions" when it should be answering them.

### Soundness
3

### Presentation
2

### Contribution
3
