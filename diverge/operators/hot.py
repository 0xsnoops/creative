"""hot: the task prompt as-is at the highest temperature that still yields working code.

Temperatures are tried from the highest down. One passes if at least half of its
calibration samples are valid per the evaluator. Calibration samples are real candidates
and are billed to this operator."""


def jobs(ctx):
    bench = ctx.cfg["bench"]
    n = bench["hot_calibration_samples"]
    chosen, calibration = 1.0, {}

    def note():  # recorded before each yield: the run may stop before the generator resumes
        ctx.notes.update(temperature=chosen, calibration_n=n,
                         calibration_valid_of_n={str(t): v for t, v in calibration.items()})

    note()
    for temperature in bench["hot_temperatures"]:
        batch = ctx.run_now([ctx.job(ctx.task.prompt, temperature) for _ in range(n)])
        calibration[temperature] = sum(c["valid"] for c in batch)
        passed = calibration[temperature] * 2 >= n
        if passed:
            chosen = temperature
        note()
        yield from batch
        if passed:
            break
    while True:
        yield ctx.job(ctx.task.prompt, chosen)
