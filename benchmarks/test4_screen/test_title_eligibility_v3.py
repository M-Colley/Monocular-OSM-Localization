import importlib.util
from pathlib import Path
import re

P = Path(__file__).with_name("title_eligibility_v3.py")
S = importlib.util.spec_from_file_location("test4_title_v3_under_test", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def test_motorvehicle_aliases_close_metadata_language_gap():
    for title in ["Rennes en Camion", "FLIXBUS Rennes Paris", "Autofahrt Dresden",
                  "Dresden mit dem LKW", "Trondheim med bil", "Tampere autolla"]:
        assert not re.search(M.ORIGINAL_ROAD_PATTERN, title, re.I)
        assert M.road_word(title)


def test_existing_english_predicate_is_retained():
    for title in ["Trondheim Dashcam", "Driving in Tampere", "Rennes RoadTrip",
                  "Garmin DriveAssist Bucharest", "Kansas City highway truck"]:
        assert re.search(M.ORIGINAL_ROAD_PATTERN, title, re.I)
        assert M.road_word(title)


def test_aliases_are_vehicle_words_not_generic_city_or_travel_words():
    for title in ["Rennes city tour", "Trondheim 180km", "Dresden Bahnhof", "Airbus Rennes"]:
        assert not M.road_word(title)


def test_motor_words_do_not_override_existing_non_realtime_criterion():
    for title in ["FLIXBUS Rennes TIMELAPS", "Dresden Autofahrt Zeitraffer",
                  "Camion Rennes accéléré", "Rennes truck timelapse"]:
        assert M.road_word(title)
        assert M.EXCLUDE.search(title)
