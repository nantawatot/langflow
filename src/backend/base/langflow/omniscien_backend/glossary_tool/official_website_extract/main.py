import argparse
import os
import sys

# set working directory to the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.web_wiki_extract import main as wiki_extract_main


def parser_initial():
    """Check variable pass in."""
    parser = argparse.ArgumentParser(
        description="This script processes two arguments.",
        epilog="Example usage: python main.py -q 'wikipedia_tool_demo'",
    )
    parser.add_argument("-q", "--query", type=str, help="The query to search for.", required=True, nargs="+")
    args = parser.parse_args()
    return args


def main():
    """Main function to execute the web score script."""
    try:
        args = parser_initial()
        result = wiki_extract_main(query=args.query)
    except Exception as e:
        result = {"error": str(e), "message": "An error occurred while processing the request."}
    sys.stderr.flush()
    if result.get("error"):
        print(result)
        sys.exit(1)

    print(result)
    sys.exit(0)


if __name__ == "__main__":
    main()
