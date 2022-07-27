import os
import os.path as op
from functools import partial
import click
import yaml
from ta_lib.pyspark.context import create_context

from ta_lib.pyspark import (
    job_planner,
    job_runner)
from ta_lib.pyspark.processors import (
    list_jobs,
    load_job_processors
)


HERE = op.dirname(op.abspath(__file__))

@click.group()
@click.option("-c", "--cfg", required=False)
@click.pass_context
def cli(cli_ctx, cfg):
    """Invoke various ML workflows."""
    if cfg is None:
        cfg = os.environ.get("TA_LIB_APP_CONFIG_PATH")
        # by default look for a config.yml in conf folder relative to the script
        if cfg is None:
            cfg = os.path.join(HERE, "conf", "config.yml")

    cli_ctx.ensure_object(dict)
    proj_ctxt = create_context(cfg)
    cli_ctx.obj["project_context"] = proj_ctxt



# --------------
# command groups
# --------------
@cli.group()
@click.pass_context
def job(cli_ctx):
    """Command group for Job management tasks."""
    pass


# -------------
# Job commands
# -------------
@job.command("list")
@click.pass_context
def _list_jobs(cli_ctx):

    # -------------
    # Data commands
    # -------------
    # FIXME: move this into a function/context manager
    # use tabulate ?
    header = "Available jobs"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for key in list_jobs():
        print(key)


@job.command("run")
@click.option(
    "-j", "--job-id", default="all", help="Id of the job to run",
)

@click.pass_context
def _run_job(cli_ctx, job_id):

    proj_ctxt = cli_ctx.obj["project_context"]
    job_catalog = proj_ctxt.job_catalog

    _completed = False
    for job_spec in job_catalog["jobs"]:
        spec_job_id = job_spec["name"]
        if (job_id != "all") and (spec_job_id != job_id):
            continue

        # FIXME: if needed, add decorators for job_planners and task_runners
        # and associate with job_id
        planner = job_planner.create_job_plan
        job_runner.main(
            proj_ctxt,
            planner,
            job_spec,
        )
        _completed = True

    if not _completed:
        print(
            f"Invalid job-id : {job_id}. \n\n"
            "Use list sub-command to see available tasks."
        )
        if not _completed:
            print(
                f"Invalid job-id : {job_id}. \n\n"
                "Use list sub-command to see available tasks."
            )


# ------------------
# Initialize the CLI
# ------------------
def main():
    # load the processing jobs defined in the current folder
    load_job_processors(op.dirname(op.abspath(__file__)))
    cli()


if __name__ == "__main__":
    main()
