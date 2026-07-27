"""Regression tests for the product's single Streamlit light theme."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest

from src.ui.styles import ENTERPRISE_STYLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THEME_CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


class StreamlitThemeConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tomllib.loads(THEME_CONFIG.read_text(encoding="utf-8"))

    def test_only_one_explicit_light_theme_is_configured(self) -> None:
        theme = self.config["theme"]
        self.assertEqual(
            {key: value for key, value in theme.items() if key != "sidebar"},
            {
                "base": "light",
                "primaryColor": "#0071E3",
                "backgroundColor": "#F5F5F7",
                "secondaryBackgroundColor": "#FFFFFF",
                "textColor": "#1D1D1F",
                "linkColor": "#0066CC",
                "baseRadius": "large",
                "buttonRadius": "full",
            },
        )
        self.assertNotIn("light", theme)
        self.assertNotIn("dark", theme)

    def test_light_sidebar_and_minimal_toolbar_are_configured(self) -> None:
        self.assertEqual(
            self.config["theme"]["sidebar"],
            {
                "backgroundColor": "#FFFFFF",
                "secondaryBackgroundColor": "#F5F5F7",
                "textColor": "#1D1D1F",
                "primaryColor": "#0071E3",
            },
        )
        self.assertEqual(self.config["client"]["toolbarMode"], "minimal")
        self.assertFalse(self.config["browser"]["gatherUsageStats"])

    def test_supported_streamlit_version_is_pinned(self) -> None:
        requirements = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        self.assertIn("streamlit==1.56.0", requirements)
        self.assertNotIn("streamlit==1.44.1", requirements)

    def test_required_light_semantic_tokens_exist(self) -> None:
        expected = {
            "--app-bg": "#F5F5F7",
            "--app-surface": "#FFFFFF",
            "--app-surface-subtle": "#FAFAFC",
            "--app-text-primary": "#1D1D1F",
            "--app-text-secondary": "#6E6E73",
            "--app-text-tertiary": "#86868B",
            "--app-border": "rgba(0, 0, 0, 0.10)",
            "--app-border-strong": "rgba(0, 0, 0, 0.16)",
            "--app-accent": "#0071E3",
            "--app-accent-hover": "#0077ED",
            "--app-accent-text": "#FFFFFF",
            "--app-warning-bg": "#FFF8E6",
            "--app-warning-text": "#6E5200",
        }
        for token, value in expected.items():
            with self.subTest(token=token):
                self.assertIn(f"{token}: {value};", ENTERPRISE_STYLES)

    def test_no_dark_mode_or_runtime_theme_detection_remains(self) -> None:
        source = (PROJECT_ROOT / "src" / "ui" / "styles.py").read_text(
            encoding="utf-8"
        )
        forbidden = (
            "theme.dark",
            "DARK_THEME",
            "prefers-color-scheme",
            "data-theme",
            "st.context.theme",
            "color-mix",
            "#0B0B0D",
            "#1C1C1E",
            "#2997FF",
        )
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)

    def test_custom_css_does_not_override_streamlit_root_surfaces(self) -> None:
        self.assertNotRegex(ENTERPRISE_STYLES, r"\.stApp\s*\{")
        sidebar_rule = re.search(
            r'\[data-testid="stSidebar"\]\s*\{(?P<body>.*?)\}',
            ENTERPRISE_STYLES,
            re.DOTALL,
        )
        self.assertIsNotNone(sidebar_rule)
        self.assertNotIn("background", sidebar_rule.group("body"))

    def test_custom_rules_do_not_hardcode_white_text_on_light_surfaces(self) -> None:
        rules_without_tokens = ENTERPRISE_STYLES.split("}", maxsplit=1)[1]
        fixed_text = re.compile(
            r"color\s*:\s*(?:white|#fff(?:fff)?|rgb\(255)",
            re.IGNORECASE,
        )
        self.assertIsNone(fixed_text.search(rules_without_tokens))
        self.assertIn("color: var(--app-text-primary)", ENTERPRISE_STYLES)
        self.assertIn("color: var(--app-text-secondary)", ENTERPRISE_STYLES)

    def test_html_components_do_not_embed_fixed_text_colors(self) -> None:
        source_paths = [
            PROJECT_ROOT / "assessment_app.py",
            PROJECT_ROOT / "src" / "ui" / "components" / "common.py",
            PROJECT_ROOT / "src" / "ui" / "components" / "cards.py",
        ]
        forbidden = re.compile(
            r"color\s*:\s*(?:black|white|#[0-9a-fA-F]|rgba?\()",
            re.IGNORECASE,
        )
        for source_path in source_paths:
            with self.subTest(source=source_path.name):
                source = source_path.read_text(encoding="utf-8")
                self.assertIsNone(forbidden.search(source))


if __name__ == "__main__":
    unittest.main()
