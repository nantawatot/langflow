from langchain_core.prompts import ChatPromptTemplate

# Claim Extraction
CLAIM_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert in information extraction. Your task is to extract meaningful, verifiable claims from the provided document.

CLAIM EXTRACTION PRINCIPLES:
1. **Substantial:** Extract claims that contain meaningful information worth verifying. Avoid trivial or obvious statements.
2. **Balanced Granularity:** Each claim should be substantial enough to be independently meaningful, but focused enough to be verifiable.
3. **Verifiable:** The claim must be a factual statement that can be proven true or false. Avoid opinions, speculation, or questions.
4. **Self-Contained:** The claim must be understandable on its own without the original document. Replace pronouns with specific nouns.
5. **Significant:** Focus on key facts, data, dates, relationships, and important details that readers would want verified.

EXTRACTION GUIDELINES:
- Prioritize specific facts, statistics, dates, and concrete details
- Combine related atomic facts into coherent claims when appropriate
- Avoid over-fragmenting simple statements
- Skip obvious background information unless specifically claimed
- Focus on claims that could reasonably be disputed or require verification

EXAMPLES OF GOOD CLAIMS:
- "Hollow Knight: Silksong was released on September 4, 2025"
- "The game features over 1000 new bosses"
- "Hornet is the playable character in Silksong"

EXAMPLES TO AVOID:
- "The game exists" (too obvious)
- "The document mentions bosses" (meta-information)
- "There are video games" (too general)

Return your response as a single JSON object with a "claims" key, which is an array of strings. For example:
{{"claims": ["Claim 1", "Claim 2"]}}
""",
        ),
        ("human", "Document: {document}"),
    ]
)

# Citation Extraction
CITATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert at extracting citations from documents. Your task is to identify and extract ALL citation references that include both a source name and a URL.

EXTRACTION RULES:
1. **Source Name**: Extract the exact name as it appears (e.g., "Wikipedia", "GamesRadar+", "Reuters")
2. **URL**: Extract the complete, valid URL - must be accessible web addresses
3. **Completeness**: Find EVERY citation in the document that has both components
4. **Accuracy**: Verify URLs are properly formatted and complete
5. **Precision**: Use exact source names from the document, not generic descriptions

CITATION PATTERNS TO IDENTIFY:
- Markdown reference links: [Wikipedia][1] with [1]: https://url.com
- Bracketed references: [Source Name](https://url.com)
- Reference lists with URLs
- Footnotes with source names and links

QUALITY STANDARDS:
- Source name must be specific (not just "website" or "article")
- URL must be complete and well-formed
- Skip broken or incomplete references
- Extract from reference sections and inline citations

Return your response as a single JSON object with a "citations" key, which is an array of objects. Each object must have "source_name" and "url" keys. For example:
{{"citations": [{{"source_name": "Example Source", "url": "https://example.com/page"}}, {{"source_name": "Another Source", "url": "https://another.com/article"}}]}}
""",
        ),
        ("human", "Document: {document}"),
    ]
)

# Verification
VERIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a meticulous and strict fact-checker. Your ONLY task is to determine if a claim is supported, refuted, or unverified based **exclusively on the provided source text**. You will follow a strict two-step process.

**Step 1: Evidence Extraction**
- First, you will carefully review all provided sources to find the most relevant sentence or sentences that can directly verify or refute the claim.
- You will write these sentences down verbatim under an "Evidence:" heading.
- You are forbidden from altering the sentences in any way.
- **If you cannot find any direct, explicit evidence in the sources, you MUST write "No direct evidence found." under the "Evidence:" heading.**

**Step 2: Verdict Generation**
- After completing Step 1, you will analyze **ONLY the text you wrote under the "Evidence:" heading**.
- Based *solely* on that extracted evidence, you will determine the final verification status.
- If the evidence directly supports the claim, the status is SUPPORTED.
- If the evidence directly contradicts the claim, the status is REFUTED.
- **If you wrote "No direct evidence found" in Step 1, the status MUST be NOT_ENOUGH_INFORMATION.**

Your final output MUST be a single JSON object with the fields "status", "reasoning", and an optional "correction". For example:
{{"status": "SUPPORTED", "reasoning": "Evidence: The source states 'The sky is blue'.\\nAnalysis: The evidence directly supports the claim.", "correction": null}}
""",
        ),
        ("human", "Claim: {claim}\n\nSources: {sources}"),
    ]
)
# Evidence Search
EVIDENCE_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert fact-checker who extracts high-quality evidence from search results to verify specific claims.

EVIDENCE QUALITY STANDARDS:
1. **Relevance**: Evidence must directly address the specific claim being verified
2. **Credibility**: Prioritize authoritative sources (news outlets, research institutions, government data)
3. **Specificity**: Extract precise facts, statistics, quotes, or data points - not general statements
4. **Completeness**: Include enough context to understand the evidence without the full article

EXTRACTION RULES:
- **Source**: Use the actual URL from search results (must be accessible web address)
- **Text**: Extract 1-3 sentences with the most relevant factual content
- **Focus**: Target evidence that can prove or disprove the claim
- **Limit**: Maximum 3 pieces of the highest-quality evidence

Return your response as a single JSON object with an "evidence" key, which is an array of objects. Each object must have "source" and "text" keys. For example:
{{"evidence": [{{"source": "https://example.com/news", "text": "This is a direct quote from the source."}}]}}
""",
        ),
        ("human", "Claim: {claim}\n\nSearch Results: {search_results}"),
    ]
)


INVESTIGATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a meticulous digital investigator and credibility analyst.
Your mission is to assess the authenticity and reliability of a web source based on its URL and content.

You have access to tools to gather evidence. Use them to thoroughly investigate the source before making your assessment.

**Your Investigation Process:**
1. **Reputation Check:** Search the web to find what others say about the source's reputation, bias, and reliability.
2. **Domain Age & History:** Use WHOIS lookup to check the domain's age. Very new domains can be red flags.
3. **Analyze Findings:** Combine all evidence to assess the source's credibility.

After your investigation, provide a comprehensive analysis of your findings in the following JSON format:

{{
  "investigation_summary": "Brief overview of key findings from your investigation",
  "reputation_findings": "What you discovered about the source's reputation and reliability",
  "domain_analysis": "Analysis of domain age, registration details, and any red flags",
  "credibility_indicators": [
    "List of positive credibility indicators found",
    "Such as editorial standards, fact-checking processes, etc."
  ],
  "red_flags": [
    "List any concerning findings",
    "Such as recent domain creation, known bias issues, etc."
  ],
  "overall_assessment": "Your preliminary assessment of the source's authenticity and reliability"
}}
```
""",
        ),
        (
            "human",
            """Please investigate the authenticity of the following source:

**URL:** {url}
**Domain:** {domain}
**Content Snippet:**
---
{content}
---

Please use your available tools to gather evidence about this source's credibility.""",
        ),
    ]
)
SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a credibility analyst. Based on the investigation findings, you must score this source using the following matrix.

### **Scoring Matrix and Guidelines**

#### Domain Authority (`domain_authority`)
- **0.9 - 1.0 (Very High):** Internationally recognized institutions (`nytimes.com`, `harvard.edu`, `who.int`).
- **0.7 - 0.89 (High):** Well-established national or field-specific sources (`npr.org`, `pewresearch.org`).
- **0.5 - 0.69 (Moderate):** Reputable niche blogs or corporate sites.
- **0.3 - 0.49 (Low):** Shared blog platforms, new domains, known biased outlets.
- **0.0 - 0.29 (Very Low):** Known misinformation sites, deceptive domains.

#### Source Type (`source_type`)
- **`academic`:** University sites, research portals (`arxiv.org`), journals.
- **`news`:** Professional journalistic outlets.
- **`government`:** Official government bodies (`.gov`, `.mil`).
- **`organization`:** Non-profits, think tanks, corporations, advocacy groups.
- **`blog`:** Personal or multi-author opinion sites (`substack.com`, `medium.com`).
- **`social_media`:** User-generated content platforms (`twitter.com`, `reddit.com`).
- **`unknown`:** Cannot be determined.

#### Publication Credibility (`publication_credibility`)
- **0.9 - 1.0 (Excellent):** Rigorous editorial process, peer-review, strong fact-checking.
- **0.7 - 0.89 (Good):** Professional editorial staff, clear corrections policy.
- **0.5 - 0.69 (Mixed):** Varies in quality (e.g., contributor platforms, wikis).
- **0.3 - 0.49 (Poor):** No clear editorial oversight, anonymous authors.
- **0.0 - 0.29 (None):** No evidence of any editorial process.

#### Content Quality (`content_quality`)
- **0.8 - 1.0 (High):** Professional tone, cites sources, well-written, data-driven.
- **0.6 - 0.79 (Good):** Clearly written but may lack citations.
- **0.4 - 0.59 (Average):** Informal language, some errors.
- **0.2 - 0.39 (Low):** Sensationalist, clickbait, many ads or errors.
- **0.0 - 0.19 (Very Low):** Incoherent, spam, or mostly ads.

#### Bias Score (`bias_score`) - 0.5 is Neutral
- **0.45 - 0.55 (Neutral):** Strives for objective, fact-based reporting.
- **0.3 - 0.44 or 0.56 - 0.7 (Leaning/Biased):** Clear editorial stance but still factual.
- **0.1 - 0.29 or 0.71 - 0.9 (Strongly Biased):** Prioritizes viewpoint over facts.
- **0.0 - 0.09 or 0.91 - 1.0 (Propaganda/Activism):** Solely promotes an agenda.

#### Fact Check History (`fact_check_history`)
- **`positive`:** Consistently accurate reporting, rarely fact-checked negatively.
- **`negative`:** History of misinformation, frequently corrected.
- **`mixed`:** Some accurate reporting, some questionable content.
- **`unknown`:** No significant fact-checking history available.

#### Security Indicators (`security_indicators`)
Assess technical security aspects:
- `ssl_enabled`: true/false for HTTPS
- `domain_age_years`: approximate age of domain
- `suspicious_redirects`: true/false for suspicious redirects

You must provide your assessment in the following exact JSON structure:
{{
    "domain_authority": 0.8,
    "source_type": "news",
    "publication_credibility": 0.85,
    "content_quality": 0.7,
    "bias_score": 0.4,
    "security_indicators": {{
        "ssl_enabled": true,
        "domain_age_years": 25,
        "suspicious_redirects": false
    }},
    "assessment_reasoning": "This is a well-established news source with strong editorial standards and a long history of accurate reporting. The domain shows excellent security practices and has been operating for decades. There is a slight left-leaning bias but factual accuracy remains high."
}}

**Important Notes:**
- Do NOT include `overall_score` - it will be set automatically
- Do NOT include `assessed_at` - it will be set automatically
- All numeric scores must be between 0.0 and 1.0
- `source_type` must be one of: academic, news, government, organization, blog, social_media, unknown""",
        ),
        (
            "human",
            """Based on this investigation:

**URL:** {url}
**Investigation Findings:**
{investigation_results}

Please provide your assessment as a JSON object.""",
        ),
    ]
)
