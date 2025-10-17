import asyncio

from langchain_core.runnables import RunnableConfig

from langflow.omniscien_backend.fact_checker.authenticity import assess_source_authenticity
from langflow.omniscien_backend.fact_checker.configuration import Configuration
from langflow.omniscien_backend.fact_checker.models import (
    AuthenticityScore,
    Citation,
    CitationsOutput,
    Claim,
    ClaimsOutput,
    Source,
    VerificationResult,
    VerificationState,
)
from langflow.omniscien_backend.fact_checker.prompts import (
    CITATION_EXTRACTION_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    VERIFICATION_PROMPT,
)
from langflow.omniscien_backend.fact_checker.utils import retrieve_url_content


async def extract_claims_and_citations(state: VerificationState, config: RunnableConfig) -> VerificationState:
    """Extract claims and citations from the document.

    If state.use_existing_inputs is True and claims/citations already exist,
    skip LLM extraction to ensure deterministic inputs across runs.
    """
    # If deterministic mode is enabled and inputs are present, skip extraction
    if getattr(state, "use_existing_inputs", False) and (state.claims or state.citations):
        return state

    configurable = Configuration.from_runnable_config(config)
    llm = configurable.get_language_model()
    document = state.document_text

    # Extract claims
    claims_chain = CLAIM_EXTRACTION_PROMPT | llm.with_structured_output(ClaimsOutput)
    claims_output = await claims_chain.ainvoke({"document": document})

    # Extract citations
    citations_chain = CITATION_EXTRACTION_PROMPT | llm.with_structured_output(CitationsOutput)
    citations_output = await citations_chain.ainvoke({"document": document})

    # Convert extracted claims to Claim objects
    claims = [Claim(claim_text=claim_text) for claim_text in claims_output.claims]

    state.claims = claims
    state.citations = citations_output.citations
    return state


async def retrieve_citation_content(state: VerificationState) -> VerificationState:
    """Retrieve content from citation URLs."""
    citations = state.citations

    # Process citations concurrently
    async def process_citation(citation: Citation) -> Citation:
        try:
            content = await retrieve_url_content(citation.url)
            citation.retrieved_content = content
            citation.is_accessible = True
        except Exception as e:
            print(f"Failed to retrieve content for {citation.url}: {e}")
            citation.is_accessible = False
            citation.retrieved_content = None
        return citation

    # Process all citations concurrently
    updated_citations = await asyncio.gather(*[process_citation(citation) for citation in citations])

    state.citations = updated_citations
    return state


async def assess_citation_authenticity(state: VerificationState, config: RunnableConfig) -> VerificationState:
    """Assess the authenticity of accessible citations."""
    citations = state.citations

    async def assess_citation(citation: Citation) -> Citation:
        if citation.is_accessible and citation.retrieved_content:
            try:
                authenticity_score = await assess_source_authenticity(citation.url, citation.retrieved_content, config)
                citation.authenticity_score = authenticity_score
            except Exception as e:
                print(f"Failed to assess authenticity for {citation.url}: {e}")
        return citation

    # Process all citations concurrently
    updated_citations = await asyncio.gather(*[assess_citation(citation) for citation in citations])

    state.citations = updated_citations
    return state


async def match_citations_to_claims(state: VerificationState, config: RunnableConfig) -> VerificationState:
    """Match citations to claims and add them as sources."""
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")

    if not llm:
        raise ValueError("LLM not provided in config")

    claims = state.claims
    citations = state.citations
    accessible_citations = [c for c in citations if c.is_accessible]

    if not accessible_citations:
        return state

    # For each claim, find matching citations
    async def process_claim(claim: Claim) -> Claim:
        # Skip if claim already has sources
        if claim.sources:
            return claim

        # Process each citation individually to determine relevance
        existing_urls = {source.url for source in claim.sources}

        for citation in accessible_citations:
            if citation.url in existing_urls:
                continue

            # Check if this specific citation is relevant to this specific claim
            relevance_prompt = f"""
Determine if this citation is relevant to the claim.

CLAIM: "{claim.claim_text}"

CITATION SOURCE: {citation.source_name}
CITATION URL: {citation.url}
CITATION CONTENT: {citation.retrieved_content if citation.retrieved_content else "No content"}

Is this citation relevant to verifying the claim? Consider:
1. Does the citation content mention the same subject/topic as the claim?
2. Could this citation potentially support or refute the claim?
3. Is there topical overlap between the citation and claim?

Respond with only "RELEVANT" or "NOT_RELEVANT" and a brief reason.
"""

            try:
                response = await llm.ainvoke(relevance_prompt)
                content = str(response.content).upper()

                if "RELEVANT" in content and "NOT_RELEVANT" not in content:
                    # Add this citation as a source
                    source = Source(
                        url=citation.url,
                        source_name=citation.source_name,
                        source_type="citation",
                        retrieved_content=citation.retrieved_content,
                        is_accessible=citation.is_accessible,
                        authenticity_score=AuthenticityScore(**citation.authenticity_score.model_dump())
                        if citation.authenticity_score
                        else None,
                    )
                    claim.sources.append(source)
                    existing_urls.add(citation.url)
                    print(f"Matched citation '{citation.source_name}' to claim '{claim.claim_text[:50]}...'")

            except Exception as e:
                print(f"Failed to assess relevance of citation {citation.url} for claim '{claim.claim_text}': {e}")

        return claim

    # Process all claims
    updated_claims = await asyncio.gather(*[process_claim(claim) for claim in claims])

    state.claims = updated_claims
    return state


async def search_external_evidence(state: VerificationState, config: RunnableConfig) -> VerificationState:
    """Search for external evidence for claims that need it.

    In deterministic mode (state.use_existing_inputs=True), skip searching to keep
    external sources fixed across runs.
    """
    if getattr(state, "use_existing_inputs", False):
        return state

    configurable = config.get("configurable", {})
    search_tools = configurable.get("search_tools", [])

    if not search_tools:
        print("No search tools available")
        return state

    search_tool = search_tools[0]  # Use the first available search tool
    claims_needing_evidence = [claim for claim in state.claims if claim.needs_external_search]

    print(f"Found {len(claims_needing_evidence)} claims needing external evidence")

    async def search_for_claim(claim: Claim) -> Claim:
        try:
            # Search for evidence with a focused query
            query = f'"{claim.claim_text}" facts evidence verify'
            print(f"Searching for: {query}")
            search_results = await search_tool.ainvoke({"query": query})

            # Use Tavily search results content directly
            if isinstance(search_results, dict) and "results" in search_results:
                existing_urls = {source.url for source in claim.sources}
                added_count = 0

                # Process up to 2 search results
                for result in search_results["results"][:2]:
                    if added_count >= 2:
                        break

                    url = result.get("url", "")
                    title = result.get("title", "")
                    content = result.get("content", "")

                    if url and url not in existing_urls and content:
                        # Use Tavily's extracted content directly - no need to fetch again
                        source = Source(
                            url=url,
                            source_name=title if title else url.split("/")[2],
                            # Use title or domain as source name
                            source_type="external",
                            retrieved_content=content,  # Use Tavily's extracted content
                            is_accessible=True,
                        )
                        claim.sources.append(source)
                        existing_urls.add(url)
                        added_count += 1
                        print(f"Added external source: {title}... - {url}")

                if added_count == 0:
                    print("No relevant external sources found in search results")
            else:
                print("Unexpected search results format")

        except Exception as e:
            print(f"Failed to search for evidence for claim '{claim.claim_text}': {e}")

        return claim

    # Process claims needing evidence
    updated_claims = []
    for claim in state.claims:
        if claim in claims_needing_evidence:
            updated_claim = await search_for_claim(claim)
            updated_claims.append(updated_claim)
        else:
            updated_claims.append(claim)

    state.claims = updated_claims
    return state


async def assess_external_evidence_authenticity(state: VerificationState, config: RunnableConfig) -> VerificationState:
    """Assess authenticity of external evidence sources."""
    claims = state.claims

    async def process_claim(claim: Claim) -> Claim:
        for source in claim.external_sources:
            if source.is_accessible and source.retrieved_content and not source.authenticity_score:
                try:
                    authenticity_score = await assess_source_authenticity(
                        source.url,
                        source.retrieved_content,
                        config,  # Pass the config object
                    )
                    source.authenticity_score = authenticity_score
                except Exception as e:
                    print(f"Failed to assess authenticity for external source {source.url}: {e}")
        return claim

    # Process all claims
    updated_claims = await asyncio.gather(*[process_claim(claim) for claim in claims])

    state.claims = updated_claims
    return state


async def verify_claims(state: VerificationState, config: RunnableConfig) -> VerificationState:
    """Verify claims against their collected sources."""
    configurable = Configuration.from_runnable_config(config)  # Use the helper method
    llm = configurable.get_language_model()

    if not llm:
        raise ValueError("LLM not provided in config")

    claims = state.claims
    verification_chain = VERIFICATION_PROMPT | llm.with_structured_output(VerificationResult)

    async def verify_claim(claim: Claim) -> Claim:
        if not claim.sources:
            claim.verification_result = VerificationResult(
                status="NOT_ENOUGH_INFORMATION", reasoning="No sources were found to verify this claim."
            )
            return claim

        # Prepare sources context
        sources_context = ""
        for i, source in enumerate(claim.sources, 1):
            if source.retrieved_content:
                sources_context += f"Source {i} ({source.source_type}): {source.url}\n"
                sources_context += f"Content: {source.retrieved_content}\n\n"

        if not sources_context.strip():
            claim.verification_result = VerificationResult(
                status="NOT_ENOUGH_INFORMATION",
                reasoning="Sources were found but no content could be retrieved to verify this claim.",
            )
            return claim

        try:
            # The chain will now directly output a VerificationResult object
            result = await verification_chain.ainvoke({"claim": claim.claim_text, "sources": sources_context})
            claim.verification_result = result

        except Exception as e:
            print(f"Failed to verify claim '{claim.claim_text}': {e}")
            claim.verification_result = VerificationResult(
                status="NOT_ENOUGH_INFORMATION", reasoning=f"Verification failed due to a structural error: {e!s}"
            )

        return claim

    # Process all claims
    updated_claims = await asyncio.gather(*[verify_claim(claim) for claim in claims])

    state.claims = updated_claims
    return state
