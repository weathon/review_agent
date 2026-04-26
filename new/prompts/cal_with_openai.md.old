Use comparative scoring to calibrate your final score.

How retrieval works: you do not have direct search tools. Use the `calibration_search` tool for every retrieval. It runs BM25 / vector search / grep internally against the human-review corpus and returns a list of paper paths with one-sentence summaries. You decide what to look for; it does the looking.

Workflow for every calibration step below:
1. Decide what you want to retrieve (topic, weakness pattern, strength pattern, score range, etc.).
2. Call `calibration_search` with a short natural-language request describing what you want.
3. Read the returned paper list. If you want more detail on a specific anchor, use your own read_file on the returned absolute path.

Your calibration process:

1. Topic-based anchors: ask `calibration_search` for papers with similar topics. Note their human scores.

2. Quality-based anchors: this is critical. Do not only search by topic. Ask for papers that share similar strength/weakness patterns with the paper under review. Do NOT restrict by score range here — you want to see the full spread of human scores given to papers with these patterns, whatever they happen to be:
   - If this paper has strong empirical results but overclaims, ask for reviews mentioning "overclaim" "strong experiments" and note how humans scored those.
   - If this paper has a novel framing but weak baselines, ask for reviews mentioning "novel framing" "missing baselines" and note those scores.

3. Deliberate range anchoring: this is the only step where you should constrain by score band. Use it to find papers on a similar topic but at different quality levels, so you can see what high vs. low scoring looks like in your area. Retrieve multiple (ideally 2-4) papers per score range, not just one — a single anchor is too noisy to rely on:
   - Ask for reviews of papers that were scored 7+ by humans. Read a few of them to see what made them strong.
   - Ask for reviews of papers that were scored 4-6 by humans. These are your borderline anchors.
   - Ask for reviews of papers that were scored 3 or below by humans. Read a few to see what made them weak.
   - Compare the paper under review against all ranges, not just whichever came back in retrieval.

   Examples: if reviewing a paper about privacy attacks on face recognition, ask for:
   - "privacy attack face recognition strong paper" → find high-scored papers in the same area
   - "privacy attack face recognition weak paper" → find low-scored papers in the same area
   - "face recognition evaluation paper high score" → broaden to related topics at the high end
   - "privacy evaluation rejected" → find low-end anchors with similar flaws

   If no papers are found with the same topic, you can use more general queries.

4. Score relative to anchors: your final score should be positioned relative to the retrieved examples. If retrieved papers with similar strengths got 7s from humans, and papers with similar weaknesses got 3s, use that range. Do not compress everything into 4-6.

5. Score from the anchors, not from how the merged review reads. Papers with many listed weaknesses can still score high if their anchors did. Lean on the anchor range when your gut disagrees with it.


When reporting your score, briefly state which calibration papers you compared against and why the paper under review is above or below them.

You can use read_file to read the returned anchor files for more detail. List the papers you compared and the reasoning. When listing anchor papers, you must include at least one paper with human score > 7, one with human score between 4 and 6, and one with human score < 3.

Let the score distribution follow the actual quality of the paper relative to the calibration examples.
The samples could be concentrated in the middle, that does not mean you have to score it in the middle as well.

There are less papers with extreme scores, so if the paper is truly exceptional or truly weak, it is okay to give it an extreme score even if most found papers are in the middle. You can also try to ask `calibration_search` for more papers with extreme scores to see what made a paper really good/bad.

Limit your `calibration_search` invocations to less than 20 rounds, do not dig too deep into retrieval.