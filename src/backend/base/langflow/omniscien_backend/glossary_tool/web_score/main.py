import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dotenv import load_dotenv
from src.score_web import main as web_score_main

load_dotenv()


def parser_initial():
    """Check variable pass in."""
    parser = argparse.ArgumentParser(
        description="This script processes arguments for web scoring.",
        epilog="Example usage: python main.py -q 'wikipedia_tool_demo' -l 'http://example.com' 'http://example.org'"
        "additionally, set SERPAPI_API_KEY in environment variables or pass --serpapi_key '<your_key>'",
    )
    parser.add_argument("-q", "--query", type=str, help="The query to search for.", required=True, nargs="+")
    parser.add_argument("-l", "--list", type=str, help="List of candidate URLs to score.", required=False, nargs="*")
    parser.add_argument(
        "--serpapi_key", type=str, help="SerpAPI key for search engine queries.", required=False, default=None
    )
    args = parser.parse_args()
    return args


def check_api_key(key):
    """Check if SERPAPI_API_KEY is set in environment variables."""
    if key.serpapi_key is not None:
        return key.serpapi_key
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        pass
    return api_key


def main():
    """Main function to execute the web score script."""
    try:
        args = parser_initial()
        check_api_key(args)
        result = web_score_main(query=args.query, list_url=args.list if args.list else [])
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
