from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from utils25 import flood_data


def check_failure(value, expected_exception, expected_text):
    with TemporaryDirectory() as tmp_dir:
        pt_path = Path(tmp_dir) / "case_000.pt"
        torch.save(value, pt_path)

        dataset = flood_data(
            path_root=tmp_dir,
            T_in=1,
            T_out=24,
            train=False,
            strategy="oneshot",
        )

        try:
            dataset[0]
        except expected_exception as exc:
            message = str(exc)
            assert "case_000.pt" in message, (
                f"Expected filename 'case_000.pt' in error message, got: {message}"
            )
            assert expected_text in message, (
                f"Expected text {expected_text!r} in error message, got: {message}"
            )
        else:
            raise AssertionError(
                f"Expected {expected_exception.__name__} for invalid tensor payload"
            )


def main():
    check_failure(
        {"not": "a tensor"},
        TypeError,
        "Expected a torch.Tensor",
    )
    print("Passed invalid-case TypeError check")

    check_failure(
        torch.zeros(8, 8, 25),
        ValueError,
        "Expected oneshot tensor shape [X, Y, T, C]",
    )
    print("Passed invalid-case shape check")

    check_failure(
        torch.zeros(8, 8, 24, 5),
        ValueError,
        "at least 25 are required",
    )
    print("Passed invalid-case timestep check")

    check_failure(
        torch.zeros(8, 8, 25, 4),
        ValueError,
        "at least 5 are required",
    )
    print("Passed invalid-case channel check")

    with TemporaryDirectory() as tmp_dir:
        pt_path = Path(tmp_dir) / "case_000.pt"
        torch.save(torch.zeros(8, 8, 25, 5), pt_path)

        dataset = flood_data(
            path_root=tmp_dir,
            T_in=1,
            T_out=24,
            train=False,
            strategy="oneshot",
        )
        x, y, mask = dataset[0]

        assert x.shape == (8, 8, 24, 1, 5), x.shape
        assert y.shape == (8, 8, 24, 3), y.shape
        assert mask.shape == (8, 8, 24, 3), mask.shape
        assert torch.isfinite(x).all(), "x contains non-finite values"
        assert torch.isfinite(y).all(), "y contains non-finite values"

    print("Passed valid-case shape and finiteness checks")
    print("DNO oneshot tensor-validation smoke test completed successfully.")


if __name__ == "__main__":
    main()
