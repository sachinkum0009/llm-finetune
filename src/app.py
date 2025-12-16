"""
script for cli application to fine tune the llm model
"""

import click


@click.command()
@click.option(
    "--name",
    prompt="Model name to fine tune",
    help="Provide the name of the model which needs to be fine tuned.",
)
def main(name):
    """
    App that fine tunes the model
    """
    print(f"Fine tuning the model: {name}")


if __name__ == "__main__":
    main()
