from govec_bench.tui.runlog import Progress, last_segments, parse_progress, styled_line


def test_last_segments__progress_bar_redraws__keeps_what_the_terminal_would_show() -> None:
    # Arrange
    text = "batch insert...\nbatch:  10%\rbatch:  50%\rbatch: 100%\n  -> done\npartial"

    # Act
    lines = last_segments(text)

    # Assert
    assert lines == ["batch insert...", "batch: 100%", "  -> done", "partial"]


def test_parse_progress__tqdm_line__counts_times_and_rate() -> None:
    # Arrange
    line = "batch inserts:  64%|######4   | 640/1000 [03:09<18:30,  3.08s/batch]"

    # Act
    progress = parse_progress(line)

    # Assert
    assert progress == Progress("batch inserts", 640, 1000, "03:09", "18:30", "3.08 s/batch")


def test_parse_progress__first_update__no_rate_yet() -> None:
    # Arrange
    line = "queries k=10:   0%|          | 0/1000 [00:00<?, ?it/s]"

    # Act
    progress = parse_progress(line)

    # Assert
    assert progress is not None
    assert (progress.desc, progress.done, progress.left) == ("queries k=10", 0, "?")


def test_parse_progress__a_result_line__none() -> None:
    # Arrange
    line = "      -> k=10: recall 0.982; mean 5.5 ms, p50 5.4"

    # Act
    progress = parse_progress(line)

    # Assert
    assert progress is None


def test_styled_line__session_lines__time_dimmed_and_the_rest_by_what_it_says() -> None:
    # Arrange
    arms = ["main", "no-hint"]
    lines = [
        "[2026-10-30 21:49:58]   no-hint failed partway (GoVecTimeoutError: timed out) -> r1-no-hint.json",
        "[2026-10-30 21:45:07]   main done -> r1-main.json",
        "[2026-10-30 21:45:07]   no-hint:",
    ]

    # Act
    styled = [styled_line(line, arms) for line in lines]

    # Assert
    assert [text.plain[:9] for text in styled] == ["21:49:58 ", "21:45:07 ", "21:45:07 "]
    assert "#f7768e" in str(styled[0].spans[-1].style)  # failed: red
    assert styled[1].plain.endswith("  ✓")
    assert "#e0af68" in str(styled[2].spans[-1].style)  # no-hint: the second arm's colour
