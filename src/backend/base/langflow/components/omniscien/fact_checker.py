from langflow.custom.custom_component.component import Component
from langflow.inputs import HandleInput, MultilineInput
from langflow.io import Output
from langflow.omniscien_backend.fact_checker.graph import fact_check_graph
from langflow.omniscien_backend.fact_checker.models import VerificationState
from langflow.schema.data import Data


class FactCheckerComponent(Component):
    display_name = "Fact Checker"
    description = "This component checks facts using an LLM and search tools."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "FactCheckerComponent"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Input Value",
            info="This is a custom component Input",
            value="""The CEO of Twitter at the time of writing this answer is Jack Dorsey.
                He co-founded Twitter in 2006 and served as the CEO until 2008, and returned as CEO in 2015.
                Dorsey is also the CEO and co-founder of Square, a financial services and mobile payment company.
                He has been recognized as one of Time magazine's 100 most influential people in the world and
                has also been awarded the Innovator of the Year Award by Wall Street Journal.""",
            tool_mode=True,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="The LLM used to run the summarization chain.",
            required=True,
        ),
        HandleInput(
            name="search_tools",
            display_name="Search Tools",
            input_types=["Tool"],
            required=True,
            info="These are the search tools that the agent can use to help with tasks.",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    async def build_output(self) -> Data:
        initial_state = VerificationState(document_text=self.input_value)

        # Configuration
        config = {
            "configurable": {
                "thread_id": "test-verification-thread",
                "llm": self.llm,
                "search_tools": self.search_tools,
            }
        }
        result = {}
        try:
            # Run the graph
            print("\nRunning verification graph...")
            result = await fact_check_graph.ainvoke(input=initial_state, config=config)

            # Print results
            print("\n" + "=" * 80)
            print("VERIFICATION RESULTS")
            print("=" * 80)

            final_state = VerificationState(**result)

            for i, claim in enumerate(final_state.claims, 1):
                print(f"\nClaim {i}: {claim.claim_text}")
                print("-" * 60)

                print(f"Sources found: {len(claim.sources)}")
                for j, source in enumerate(claim.sources, 1):
                    print(f"  Source {j} ({source.source_type}): {source.url}")
                    if source.source_name:
                        print(f"    Name: {source.source_name}")
                    print(f"    Accessible: {source.is_accessible}")
                    if source.authenticity_score:
                        print(f"    Authenticity: {source.authenticity_score.overall_score:.2f}")

                if claim.verification_result:
                    print(f"\nVerification Status: {claim.verification_result.status}")
                    print(f"Reasoning: {claim.verification_result.reasoning}")
                    if claim.verification_result.correction:
                        print(f"Correction: {claim.verification_result.correction}")
                else:
                    print("\nVerification: Not completed")

                print("\n" + "=" * 80)

        except Exception as e:
            print(f"\nError during verification: {e}")
            import traceback

            traceback.print_exc()
        return result
