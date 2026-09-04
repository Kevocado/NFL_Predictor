from pathlib import Path

from nfl_predictor import config


def test_paths_are_absolute_and_created():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.CACHE_DIR.is_dir()
    assert config.MODELS_DIR.is_dir()
    assert config.SCHEDULES_CACHE_DIR.is_dir()
    assert config.PLAYER_STATS_CACHE_DIR.is_dir()
    assert config.INJURIES_CACHE_DIR.is_dir()
    assert config.ODDS_CACHE_DIR.is_dir()


def test_current_season_is_reasonable():
    assert 2020 <= config.CURRENT_SEASON <= 2100


def test_odds_api_sport_key():
    assert config.ODDS_API_SPORT_KEY == "americanfootball_nfl"
