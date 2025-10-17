from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AuthenticityScore(BaseModel):
    """Represents the authenticity/credibility assessment of a source."""

    overall_score: float | None = Field(
        description="Overall authenticity score from 0.0 (not credible) to 1.0 (highly credible).",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    domain_authority: float = Field(
        description="Domain authority/reputation score (0.0-1.0).", ge=0.0, le=1.0, default=0.5
    )
    source_type: Literal["academic", "news", "government", "organization", "blog", "social_media", "unknown"] = Field(
        description="Type/category of the source.", default="unknown"
    )
    publication_credibility: float = Field(
        description="Credibility of the publication/website (0.0-1.0).", ge=0.0, le=1.0, default=0.5
    )
    content_quality: float = Field(
        description="Quality indicators of the content (0.0-1.0).", ge=0.0, le=1.0, default=0.5
    )
    bias_score: float = Field(
        description="Bias assessment where 0.5 is neutral, closer to 0 or 1 indicates bias.",
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    security_indicators: dict[str, Any] = Field(
        description="Technical security indicators (SSL, domain age, etc.).", default_factory=dict
    )
    assessment_reasoning: str = Field(description="Explanation of the authenticity assessment.", default="")
    assessed_at: str | None = Field(description="When the authenticity assessment was performed.", default=None)

    def calculate_overall_score(self) -> float:
        """Calculate the overall score based on individual metrics.

        Approach:
        1. Base score from core credibility metrics
        2. Apply bias penalty (deviation from neutral)
        3. Apply fact-check history modifier
        4. Apply security bonus/penalty
        """
        # Core credibility score (60% weight)
        core_score = (
            self.domain_authority * 0.4  # Domain reputation is crucial
            + self.publication_credibility * 0.4  # Editorial standards matter
            + self.content_quality * 0.2  # Content quality supports
        )

        # Bias penalty: farther from 0.5 (neutral) reduces score
        bias_deviation = abs(self.bias_score - 0.5)
        bias_penalty = bias_deviation * 0.3  # Max penalty of 0.15

        # Security indicators modifier
        security_bonus = 0.0
        if self.security_indicators:
            security_score = 0.0
            total_indicators = 0

            # SSL enabled
            if "ssl_enabled" in self.security_indicators:
                security_score += 1 if self.security_indicators["ssl_enabled"] else 0
                total_indicators += 1

            # Domain age (older is generally better)
            if "domain_age_years" in self.security_indicators:
                age = self.security_indicators["domain_age_years"]
                if age >= 10:
                    security_score += 1
                elif age >= 5:
                    security_score += 0.7
                elif age >= 1:
                    security_score += 0.4
                else:
                    security_score += 0  # Very new domains are concerning
                total_indicators += 1

            # No suspicious activity
            suspicious = self.security_indicators.get("suspicious_redirects", False)
            if not suspicious:
                security_score += 1
                total_indicators += 1

            if total_indicators > 0:
                security_bonus = (security_score / total_indicators - 0.5) * 0.1  # Max ±0.05

        # Calculate final score
        final_score = core_score - bias_penalty + security_bonus

        # Ensure score stays within bounds
        return max(0.0, min(1.0, final_score))


class Source(BaseModel):
    """A unified model for any piece of evidence, whether from an original citation or external search."""

    url: str = Field(description="The URL of the evidence source.")

    source_type: Literal["citation", "external"] = Field(
        description="Indicates if the source was a citation in the original document or found via external search."
    )

    # Optional fields that are more relevant to original citations
    source_name: str | None = Field(
        default=None, description="The name of the source (e.g., 'Reuters'), typically for original citations."
    )

    # Content and accessibility fields
    retrieved_content: str | None = Field(default=None, description="The content retrieved from the source URL.")
    is_accessible: bool = Field(default=False, description="Whether the source URL was successfully accessed.")

    # Authenticity
    authenticity_score: AuthenticityScore | None = Field(
        default=None, description="Authenticity/credibility assessment of this source."
    )

    def __hash__(self):
        """Make Source hashable by using url as unique identifier."""
        return hash(self.url)

    def __eq__(self, other):
        """Sources are equal if they have the same URL."""
        return isinstance(other, Source) and self.url == other.url


class VerificationResult(BaseModel):
    """Encapsulates the final verification output for a claim."""

    status: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
    reasoning: str
    correction: str | None = None


class Claim(BaseModel):
    """A self-contained model for an atomic claim and its entire verification lifecycle.
    This is now the central object in our state.
    """

    claim_text: str = Field(description="The individual, verifiable claim.")

    # All evidence for this claim is stored directly within it.
    sources: list[Source] = Field(
        default_factory=list,
        description="A list of all sources (citations and external evidence) relevant to this claim.",
    )

    # The final verification result is also part of the claim's state.
    verification_result: VerificationResult | None = Field(
        default=None, description="The final verification result for this claim."
    )

    # Helper properties for analysis
    @property
    def needs_external_search(self) -> bool:
        """Determines if a claim lacks sufficient supporting evidence from citations."""
        if not self.sources:
            return True

        # Check if we have any citation sources that are accessible and contain relevant content
        citation_sources = [s for s in self.sources if s.source_type == "citation" and s.is_accessible]
        if len(citation_sources) == 0:
            return True

        # For now, assume that if we have accessible citation sources, we don't need external search
        # This can be enhanced later to check content relevance
        return False

    @property
    def citation_sources(self) -> list[Source]:
        """Get all citation-type sources for this claim."""
        return [s for s in self.sources if s.source_type == "citation"]

    @property
    def external_sources(self) -> list[Source]:
        """Get all external-type sources for this claim."""
        return [s for s in self.sources if s.source_type == "external"]

    def __hash__(self):
        """Make Claim hashable by using claim_text as unique identifier."""
        return hash(self.claim_text)

    def __eq__(self, other):
        """Claims are equal if they have the same text."""
        return isinstance(other, Claim) and self.claim_text == other.claim_text


# Keep the original Citation model for initial extraction step
class Citation(BaseModel):
    """A single citation that a document references - used for initial extraction."""

    source_name: str = Field(description="The name of the source (e.g., journal name, website name).")
    url: str = Field(description="The URL of the source.")
    retrieved_content: str | None = Field(
        default=None, description="The original content retrieved from the citation URL."
    )
    is_accessible: bool = Field(default=True, description="Whether the citation URL can be accessed successfully.")
    authenticity_score: AuthenticityScore | None = Field(
        default=None, description="Authenticity/credibility assessment of this citation source."
    )

    def __hash__(self):
        """Make Citation hashable by using url as unique identifier."""
        return hash(self.url)

    def __eq__(self, other):
        """Citations are equal if they have the same URL."""
        return isinstance(other, Citation) and self.url == other.url


# Output models for LLM structured outputs
class ClaimsOutput(BaseModel):
    """The structured output model for claims extracted from a document."""

    claims: list[str] = Field(description="A list of simple, verifiable, and context-independent atomic claims.")


class CitationsOutput(BaseModel):
    """The structured output model for citations extracted from a document."""

    citations: list[Citation] = Field(
        description="A list of citations with source names and URLs found in the document."
    )


class EvidenceOutput(BaseModel):
    """Structured output containing a list of evidence found for a claim."""

    evidence: list[dict] = Field(description="A list of evidence snippets, each with a source URL and content.")


class VerificationState(BaseModel):
    """A simplified, unified state for the verification graph."""

    document_text: str = Field(description="The document text.")

    # Optional: when True, nodes will reuse existing claims/citations/sources and skip
    # LLM extraction and external searches. This enables consistent inputs across runs
    # for comparing different LLM verification outputs.
    use_existing_inputs: bool = Field(
        default=False,
        description="If True, skip extraction/search and use the claims/citations already present in state.",
    )

    # The list of claims is now the *only* primary state variable.
    claims: list[Claim] = Field(
        default_factory=list, description="The list of claims, each containing its own sources and verification status."
    )

    # We still need a temporary place to hold citations right after extraction
    # before they are matched to claims.
    citations: list[Citation] = Field(
        default_factory=list, description="Temporary list of citations extracted from the document."
    )


class WhoisInfo(BaseModel):
    """Structured output for WHOIS domain information."""

    domain_name: str | None = Field(description="The registered domain name.")
    registrar: str | None = Field(description="The registrar of the domain.")
    creation_date: datetime | None = Field(description="The date the domain was created.")
    expiration_date: datetime | None = Field(description="The date the domain will expire.")
    updated_date: datetime | None = Field(description="The date the domain was last updated.")
    name_servers: list[str] | None = Field(description="The name servers for the domain.")
