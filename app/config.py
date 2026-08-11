import os
import yaml

CONFIG_PATH_DEFAULT = "config.yml"


class Config:
    _config_data = None
    _config_path = CONFIG_PATH_DEFAULT

    @classmethod
    def load_config(cls, path: str = None):
        """Load YAML configuration file. Use path or default."""
        if path is not None:
            cls._config_path = path
        if not os.path.exists(cls._config_path):
            raise FileNotFoundError(f"Configuration file not found: {cls._config_path}")
        with open(cls._config_path, "r", encoding="utf-8") as file:
            cls._config_data = yaml.safe_load(file)
        cls._apply_attributes()

    @classmethod
    def _apply_attributes(cls):
        """Assign config keys as class attributes."""
        if cls._config_data is None:
            return
        for key, value in cls._config_data.items():
            setattr(Config, key, value)

    @classmethod
    def reload(cls, path: str = None):
        """Reload configuration from file (e.g. after config was updated)."""
        cls.load_config(path or cls._config_path)

    @classmethod
    def get(cls, key: str, default=None):
        """Retrieve a configuration value by key."""
        if cls._config_data is None:
            cls.load_config()
        return cls._config_data.get(key, default)

# Load configuration on import
Config.load_config()

# Provide sensible defaults if not supplied in config.yml
if not hasattr(Config, "THRESHOLDS"):
    setattr(
        Config,
        "THRESHOLDS",
        {
            "retriever": 0.6,
            "reasoner": 0.7,
        },
    )

_DEFAULT_PII_FILTERS = {
    "email": True,
    "phone": True,
    "ip": True,
    "date": True,
    "id": True,
    "name": True,
}
if not hasattr(Config, "PII_FILTERS"):
    setattr(Config, "PII_FILTERS", _DEFAULT_PII_FILTERS)

# Reasoning runtime mode ("ollama" or "mock").
if not hasattr(Config, "REASONING_MODE"):
    setattr(Config, "REASONING_MODE", "ollama")

# Master switch for the governance agent (ablation baseline when False).
if not hasattr(Config, "GOVERNANCE_ENABLED"):
    setattr(Config, "GOVERNANCE_ENABLED", True)

# Policy-aware answerability governance defaults.
_DEFAULT_GOVERNANCE = {
    "answerability_enabled": True,
    "retriever_abstain_below": 0.2,
    "reasoner_abstain_below": 0.3,
    "clarify_band": [0.3, 0.5],
}
if not hasattr(Config, "GOVERNANCE"):
    setattr(Config, "GOVERNANCE", _DEFAULT_GOVERNANCE)
