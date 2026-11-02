import pytest

from govec_bench.tui.theme import DEFAULT_THEME, THEMES, palette, rgb, use


def test_palette_arm__arms_past_the_palette__colours_wrap_and_unknown_is_text() -> None:
    # Arrange
    colors = THEMES[DEFAULT_THEME]
    arms = [f"arm{index}" for index in range(len(colors.arms) + 1)]

    # Act
    first, wrapped, unknown = colors.arm(arms, "arm0"), colors.arm(arms, arms[-1]), colors.arm(arms, "nope")

    # Assert
    assert first == wrapped == colors.arms[0]
    assert unknown == colors.text


def test_use__unknown_theme__names_the_themes_there_are() -> None:
    # Arrange
    name = "nope"

    # Act
    with pytest.raises(ValueError, match="the themes are") as raised:
        use(name)

    # Assert
    assert DEFAULT_THEME in str(raised.value)
    assert palette() is THEMES[DEFAULT_THEME]


def test_rgb__hex_colour__plotext_tuple() -> None:
    # Arrange
    color = "#1a1b26"

    # Act
    value = rgb(color)

    # Assert
    assert value == (26, 27, 38)
