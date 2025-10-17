import argparse
import asyncio

from app.agent.manus import Manus
from app.logger import logger


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Manus agent with a prompt")
    parser.add_argument("--prompt", type=str, required=False, help="Input prompt for the agent")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum steps for the agent")
    args = parser.parse_args()
    max_steps = args.max_steps if args.prompt else 5

    # Create and initialize Manus agent
    agent = await Manus.create(max_steps=max_steps)
    try:
        # Use command line prompt if provided, otherwise ask for input
        prompt = args.prompt if args.prompt else ""

        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")

    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
