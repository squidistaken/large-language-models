import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

SLURM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SLURM_DIR.parent
PROFILE_DIR = SLURM_DIR / "conf"
JOB_SCRIPT = SLURM_DIR / "run_job.sh"
RTX_GPU = "rtx_pro_6000"
RTX_ENV = ".venv-rtx"
NO_GPU = "none"

SBATCH_OPTIONS = {
    "job_name": "--job-name",
    "time": "--time",
    "cpus_per_task": "--cpus-per-task",
    "mem": "--mem",
    "partition": "--partition",
    "account": "--account",
    "qos": "--qos",
    "constraint": "--constraint",
    "output": "--output",
    "error": "--error",
}
REQUIRED_SLURM_OPTIONS = {
    "job_name",
    "time",
    "cpus_per_task",
    "mem",
    "partition",
    "gpu_type",
}


def profile_path(profile: str) -> Path:
    """Get the path to a Slurm profile, checking that it exists.

    Args:
        profile (str): The profile name.

    Returns:
        Path: A path to the profile file.

    Raises:
        FileNotFoundError: If the profile does not exist.
    """
    path = PROFILE_DIR / Path(profile).with_suffix(".yaml").name

    if not path.is_file():
        raise FileNotFoundError(f"No profile {path}; choose one of slurm/conf/*.yaml")

    return path


def load_profile(path: Path) -> dict[str, Any]:
    """Load a Slurm profile.

    Args:
        path (Path): The profile file.

    Returns:
        dict[str, Any]: A mapping of the profile.

    Raises:
        TypeError: If the profile is not a mapping.
    """
    profile = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    if not isinstance(profile, dict):
        raise TypeError(f"{path} must contain a mapping")

    return cast(dict[str, Any], profile)


def absolute_log_path(value: str) -> str:
    """Get an absolute path for a Slurm log file.

    Args:
        value (str): The path to a log file.

    Returns:
        str: An absolute path.
    """
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


def setup_command(environment: str) -> str:
    """Set up an environment.

    Args:
        environment (str): The environment name.

    Returns:
        str: A shell command.
    """
    return f"./slurm/setup_uv.sh{' --use-rtx' if environment == RTX_ENV else ''}"


def uv_environment(gpu_type: str) -> str:
    """Get the uv environment name for a Slurm job.

    Args:
        gpu_type (str): The GPU type of the profile.

    Returns:
        str: An environment name.

    Raises:
        FileNotFoundError: If the environment is missing.
        RuntimeError: If the environment cannot import the project.
    """
    name = RTX_ENV if gpu_type == RTX_GPU else ".venv"
    python = PROJECT_ROOT / name / "bin" / "python"

    if not python.is_file():
        raise FileNotFoundError(f"Environment {name} is missing; run: {setup_command(name)}")

    result = subprocess.run(
        [str(python), "-c", "import src.cli.menu"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        detail = output.splitlines()[-1] if output else "import failed"

        raise RuntimeError(
            f"Environment {name} is not ready ({detail}); run: {setup_command(name)}"
        )

    return name


def require_action(overrides: list[str]) -> str:
    """Get the action from Hydra overrides.

    Args:
        overrides (list[str]): The Hydra overrides.

    Returns:
        str: An action.

    Raises:
        ValueError: If no action override is given.
    """
    for override in overrides:
        key, _, value = override.partition("=")

        if key == "action" and value:
            return value

    raise ValueError("Pass an action, e.g. action=finetune; a job cannot use the menu")


def build_command(
    profile_file: Path,
    profile: dict[str, Any],
    overrides: list[str],
    dependency: str | None,
    parsable: bool,
) -> list[str]:
    """Build a Slurm sbatch command from a profile and Hydra overrides.

    Args:
        profile_file (Path): The profile file.
        profile (dict[str, Any]): The loaded profile.
        overrides (list[str]): The Hydra overrides passed to main.py.
        dependency (str | None): The Slurm dependency.
        parsable (bool): Whether sbatch should print only the job id.

    Returns:
        list[str]: A list of command-line arguments for sbatch.

    Raises:
        TypeError: If the profile has no slurm mapping.
        ValueError: If the profile has unknown, missing or unsupported options.
    """
    slurm = profile.get("slurm")

    if not isinstance(slurm, dict):
        raise TypeError("A Slurm profile needs a top-level 'slurm' mapping")

    action = require_action(overrides)

    unknown = set(slurm) - set(SBATCH_OPTIONS) - {"gpu_type"}
    missing = REQUIRED_SLURM_OPTIONS - set(slurm)

    if unknown:
        raise ValueError(f"Unsupported slurm options: {', '.join(sorted(unknown))}")

    if missing:
        raise ValueError(f"Missing slurm options: {', '.join(sorted(missing))}")

    gpu_type = str(slurm["gpu_type"])
    environment = uv_environment(gpu_type)
    command = ["sbatch", f"--chdir={PROJECT_ROOT}", "--nodes=1", "--ntasks=1"]

    if parsable:
        command.append("--parsable")

    if dependency:
        command.append(f"--dependency={dependency}")

    for key, option in SBATCH_OPTIONS.items():
        value = slurm.get(key)

        if value is None:
            continue

        if key == "job_name":
            value = f"{value}-{action}"

        if key in {"output", "error"}:
            value = absolute_log_path(str(value))

        command.append(f"{option}={value}")

    if gpu_type != NO_GPU:
        command.append(f"--gpus-per-node={gpu_type}:1")

    command += [
        str(JOB_SCRIPT),
        profile_file.name,
        environment,
        *overrides,
    ]

    return command


def main() -> None:
    """Run the program."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--slurm-config", required=True, help="The Slurm profile to use.")
    parser.add_argument("--dependency", help="The Slurm dependency.")
    parser.add_argument(
        "--parsable", action="store_true", help="Whether sbatch should print only the job id."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Whether to print the command without running it."
    )
    parser.add_argument("overrides", nargs="*", help="The Hydra overrides to pass to main.py.")

    args = parser.parse_intermixed_args()

    try:
        profile_file = profile_path(args.slurm_config)
        command = build_command(
            profile_file,
            load_profile(profile_file),
            args.overrides,
            args.dependency,
            args.parsable,
        )

    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)

        raise SystemExit(2) from None

    # With --parsable, stdout carries only the job id so scripts can chain jobs.
    print(shlex.join(command), file=sys.stderr if args.parsable else sys.stdout)

    if args.dry_run:
        if args.parsable:
            print("DRYRUN")

        return

    if shutil.which("sbatch") is None:
        print("ERROR: sbatch is not available; run this on a Slurm login node", file=sys.stderr)

        raise SystemExit(2)

    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from None


if __name__ == "__main__":
    main()
