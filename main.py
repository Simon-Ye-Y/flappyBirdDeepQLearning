import argparse
import asyncio

from src.flappy import Flappy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the Flappy Bird DQN agent.")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Whether to train the agent or run evaluation episodes.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run. Defaults to the class configured value for training and 1 for evaluation.",
    )
    parser.add_argument(
        "--save-best-model",
        type=str,
        default=None,
        help="Path to store the best performing model checkpoint during training.",
    )
    parser.add_argument(
        "--best-average-window",
        type=int,
        default=10,
        help="Window size for the moving average used to determine the best model.",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Checkpoint to load before running. Mandatory for evaluation mode.",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=0,
        help="If provided during training, automatically run this many evaluation episodes using the saved best model.",
    )
    return parser


async def async_main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    flappy = Flappy()

    if args.mode == "train":
        if args.load_model:
            flappy.agent.load_model(args.load_model)

        await flappy.start(
            mode="train",
            episodes=args.episodes,
            save_best_model_path=args.save_best_model,
            best_average_window=args.best_average_window,
        )

        if args.evaluation_episodes > 0 and args.save_best_model:
            await flappy.start(
                mode="evaluate",
                evaluation_episodes=args.evaluation_episodes,
                model_path=args.save_best_model,
            )
    else:
        model_path = args.load_model or args.save_best_model
        if not model_path:
            parser.error("Evaluation mode requires --load-model or --save-best-model to specify the checkpoint to use.")
        await flappy.start(
            mode="evaluate",
            episodes=args.episodes or args.evaluation_episodes,
            model_path=model_path,
        )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
