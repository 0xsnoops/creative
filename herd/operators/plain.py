"""plain: the task prompt as-is at temperature 1.0."""


def jobs(ctx):
    while True:
        yield ctx.job(ctx.task.prompt, temperature=1.0)
