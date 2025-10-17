import argparse
import os
import sys

# set working directory to the src folder
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.map_route.map_utility import main as map_route_main


def parser_initial():
    """Check variable pass in."""
    parser = argparse.ArgumentParser(
        description="This script processes two arguments.",
        epilog="Example usage: python main.py -q 'query1' 'query2'",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="The query to search for.",
        # required=True,
        nargs="+",
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to execute the web score script."""
    try:
        args = parser_initial()
        result = map_route_main(map_query=args.query)
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
