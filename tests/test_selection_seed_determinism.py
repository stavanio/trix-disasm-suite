import os
import subprocess
import sys

from benchmark import selection as S


EXPECTED = {
    "BATTERY": 29,
    "CRANK": 86,
    "PCB": 65,
    "PRY": 19,
    "SCREW": 99,
    "SNAP": 72,
}


def test_frozen_task_offsets():
    assert S.TASK_SEED_OFFSET == EXPECTED


def test_expected_seed_bases():
    for task, offset in EXPECTED.items():
        assert S.validation_seeds(task, 1) == [
            500_000 + 1000 * offset
        ]
        assert S.test_seeds(task, 1) == [
            800_000 + 1000 * offset
        ]


def test_independent_of_python_hash_seed():
    code = (
        "from benchmark.selection import "
        "validation_seeds,test_seeds;"
        "print(validation_seeds('PCB',3));"
        "print(test_seeds('PCB',3))"
    )

    outputs = []

    for hash_seed in ("1", "777", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed

        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                env=env,
                text=True,
            )
        )

    assert outputs[0] == outputs[1] == outputs[2]
