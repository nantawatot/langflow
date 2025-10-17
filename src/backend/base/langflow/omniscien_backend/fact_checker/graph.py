from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from langflow.omniscien_backend.fact_checker.configuration import Configuration
from langflow.omniscien_backend.fact_checker.models import VerificationState
from langflow.omniscien_backend.fact_checker.nodes import (
    assess_citation_authenticity,
    assess_external_evidence_authenticity,
    extract_claims_and_citations,
    match_citations_to_claims,
    retrieve_citation_content,
    search_external_evidence,
    verify_claims,
)


def create_unified_verification_graph():
    """Create and configure the unified verification graph."""
    verification_graph_builder = StateGraph(
        state_schema=VerificationState,
        config_schema=Configuration,
    )

    # Add all nodes for unified workflow
    verification_graph_builder.add_node("extract_claims_and_citations", extract_claims_and_citations)
    verification_graph_builder.add_node("retrieve_citation_content", retrieve_citation_content)
    verification_graph_builder.add_node("assess_citation_authenticity", assess_citation_authenticity)
    verification_graph_builder.add_node("match_citations_to_claims", match_citations_to_claims)
    verification_graph_builder.add_node("search_external_evidence", search_external_evidence)
    verification_graph_builder.add_node("assess_external_evidence_authenticity", assess_external_evidence_authenticity)
    verification_graph_builder.add_node("verify_claims", verify_claims)

    # Add edges for the unified workflow
    verification_graph_builder.add_edge(START, "extract_claims_and_citations")
    verification_graph_builder.add_edge("extract_claims_and_citations", "retrieve_citation_content")
    verification_graph_builder.add_edge("retrieve_citation_content", "assess_citation_authenticity")
    verification_graph_builder.add_edge("assess_citation_authenticity", "match_citations_to_claims")
    verification_graph_builder.add_edge("match_citations_to_claims", "search_external_evidence")
    verification_graph_builder.add_edge("search_external_evidence", "assess_external_evidence_authenticity")
    verification_graph_builder.add_edge("assess_external_evidence_authenticity", "verify_claims")
    verification_graph_builder.add_edge("verify_claims", END)

    # Compile the graph with memory
    return verification_graph_builder.compile(checkpointer=MemorySaver())


# Create the compiled graph instance
fact_check_graph = create_unified_verification_graph()
