import click
from ci.ray_ci.automation.upgrade_version_lib import (
    get_current_version,
    upgrade_file_version,
)
import os

bazel_workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")


@click.command()
@click.option("--new_version", required=True, type=str)
def main(new_version: str):
    """
    Update the version in the files to the specified version.
    """
    main_version, java_version = get_current_version(bazel_workspace_dir)

    upgrade_file_version(
        main_version,
        java_version,
        new_version,
        bazel_workspace_dir,
    )


if __name__ == "__main__":
    main()
