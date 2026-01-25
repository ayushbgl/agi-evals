"""
Word lists for Codenames game.

Contains the standard 400-word list from the original game,
plus additional themed word packs.
"""

from typing import List, Optional
import random


# Standard Codenames word list (400 words from the original game)
STANDARD_WORDS = [
    "AFRICA", "AGENT", "AIR", "ALIEN", "ALPS", "AMAZON", "AMBULANCE", "AMERICA",
    "ANGEL", "ANTARCTICA", "APPLE", "ARM", "ATLANTIS", "AUSTRALIA", "AZTEC",
    "BACK", "BALL", "BAND", "BANK", "BAR", "BARK", "BAT", "BATTERY", "BEACH",
    "BEAR", "BEAT", "BED", "BEIJING", "BELL", "BELT", "BERLIN", "BERMUDA",
    "BERRY", "BILL", "BLOCK", "BOARD", "BOLT", "BOMB", "BOND", "BOOM", "BOOT",
    "BOTTLE", "BOW", "BOX", "BRIDGE", "BRUSH", "BUCK", "BUFFALO", "BUG",
    "BUGLE", "BUTTON", "CALF", "CANADA", "CAP", "CAPITAL", "CAR", "CARD",
    "CARROT", "CASINO", "CAST", "CAT", "CELL", "CENTAUR", "CENTER", "CHAIR",
    "CHANGE", "CHARGE", "CHECK", "CHEST", "CHICK", "CHINA", "CHOCOLATE",
    "CHURCH", "CIRCLE", "CLIFF", "CLOAK", "CLUB", "CODE", "COLD", "COMIC",
    "COMPOUND", "CONCERT", "CONDUCTOR", "CONTRACT", "COOK", "COPPER", "COTTON",
    "COURT", "COVER", "CRANE", "CRASH", "CRICKET", "CROSS", "CROWN", "CYCLE",
    "CZECH", "DANCE", "DATE", "DAY", "DEATH", "DECK", "DEGREE", "DIAMOND",
    "DICE", "DINOSAUR", "DISEASE", "DOCTOR", "DOG", "DRAFT", "DRAGON", "DRESS",
    "DRILL", "DROP", "DUCK", "DWARF", "EAGLE", "EGYPT", "EMBASSY", "ENGINE",
    "ENGLAND", "EUROPE", "EYE", "FACE", "FAIR", "FALL", "FAN", "FENCE", "FIELD",
    "FIGHTER", "FIGURE", "FILE", "FILM", "FIRE", "FISH", "FLUTE", "FLY",
    "FOOT", "FORCE", "FOREST", "FORK", "FRANCE", "GAME", "GAS", "GENIUS",
    "GERMANY", "GHOST", "GIANT", "GLASS", "GLOVE", "GOLD", "GRACE", "GRASS",
    "GREECE", "GREEN", "GROUND", "HAM", "HAND", "HAWK", "HEAD", "HEART",
    "HELICOPTER", "HIMALAYAS", "HOLE", "HOLLYWOOD", "HONEY", "HOOD", "HOOK",
    "HORN", "HORSE", "HOSPITAL", "HOTEL", "ICE", "ICE CREAM", "INDIA", "IRON",
    "IVORY", "JACK", "JAM", "JET", "JUPITER", "KANGAROO", "KETCHUP", "KEY",
    "KID", "KING", "KIWI", "KNIFE", "KNIGHT", "LAB", "LAP", "LASER", "LAWYER",
    "LEAD", "LEMON", "LEPRECHAUN", "LIFE", "LIGHT", "LIMOUSINE", "LINE", "LINK",
    "LION", "LITTER", "LOCH NESS", "LOCK", "LOG", "LONDON", "LUCK", "MAIL",
    "MAMMOTH", "MAPLE", "MARBLE", "MARCH", "MASS", "MATCH", "MERCURY", "MEXICO",
    "MICROSCOPE", "MILLIONAIRE", "MINE", "MINT", "MISSILE", "MODEL", "MOLE",
    "MOON", "MOSCOW", "MOUNT", "MOUSE", "MOUTH", "MUG", "NAIL", "NEEDLE", "NET",
    "NEW YORK", "NIGHT", "NINJA", "NOTE", "NOVEL", "NURSE", "NUT", "OCTOPUS",
    "OIL", "OLIVE", "OLYMPUS", "OPERA", "ORANGE", "ORGAN", "PALM", "PAN",
    "PANTS", "PAPER", "PARACHUTE", "PARK", "PART", "PASS", "PASTE", "PENGUIN",
    "PHOENIX", "PIANO", "PIE", "PILOT", "PIN", "PIPE", "PIRATE", "PISTOL",
    "PIT", "PITCH", "PLANE", "PLASTIC", "PLATE", "PLATYPUS", "PLAY", "PLOT",
    "POINT", "POISON", "POLE", "POLICE", "POOL", "PORT", "POST", "POUND",
    "PRESS", "PRINCESS", "PUMPKIN", "PUPIL", "PYRAMID", "QUEEN", "RABBIT",
    "RACKET", "RAY", "REVOLUTION", "RING", "ROBIN", "ROBOT", "ROCK", "ROME",
    "ROOT", "ROSE", "ROULETTE", "ROUND", "ROW", "RULER", "SATELLITE", "SATURN",
    "SCALE", "SCHOOL", "SCIENTIST", "SCORPION", "SCREEN", "SCUBA DIVER", "SEAL",
    "SERVER", "SHADOW", "SHAKESPEARE", "SHARK", "SHIP", "SHOE", "SHOP", "SHOT",
    "SHOULDER", "SILK", "SINK", "SKYSCRAPER", "SLIP", "SLUG", "SMUGGLER",
    "SNOW", "SNOWMAN", "SOCK", "SOLDIER", "SOUL", "SOUND", "SPACE", "SPELL",
    "SPIDER", "SPIKE", "SPINE", "SPOT", "SPRING", "SPY", "SQUARE", "STADIUM",
    "STAFF", "STAR", "STATE", "STICK", "STOCK", "STRAW", "STREAM", "STRIKE",
    "STRING", "SUB", "SUIT", "SUPER HERO", "SWING", "SWITCH", "TABLE", "TABLET",
    "TAG", "TAIL", "TAP", "TEACHER", "TELESCOPE", "TEMPLE", "THIEF", "THUMB",
    "TICK", "TIE", "TIME", "TOKYO", "TOOTH", "TORCH", "TOWER", "TRACK", "TRAIN",
    "TRIANGLE", "TRIP", "TRUNK", "TUBE", "TURKEY", "UNDERTAKER", "UNICORN",
    "VACUUM", "VAN", "VET", "WAKE", "WALL", "WAR", "WASHER", "WASHINGTON",
    "WATCH", "WATER", "WAVE", "WEB", "WELL", "WHALE", "WHIP", "WIND", "WITCH",
    "WORM", "YARD",
]

# Simple/easy words for beginners
EASY_WORDS = [
    "DOG", "CAT", "BIRD", "FISH", "TREE", "HOUSE", "CAR", "BALL", "BOOK", "CHAIR",
    "TABLE", "BED", "DOOR", "WINDOW", "FLOOR", "WALL", "ROOF", "GARDEN", "FLOWER",
    "GRASS", "SUN", "MOON", "STAR", "RAIN", "SNOW", "WIND", "FIRE", "WATER", "EARTH",
    "SKY", "MOUNTAIN", "RIVER", "OCEAN", "BEACH", "FOREST", "DESERT", "ISLAND",
    "BRIDGE", "ROAD", "PATH", "KING", "QUEEN", "PRINCE", "KNIGHT", "CASTLE", "CROWN",
    "SWORD", "SHIELD", "ARROW", "BOW", "HORSE", "DRAGON", "MAGIC", "SPELL", "WITCH",
    "WIZARD", "GHOST", "MONSTER", "ANGEL", "DEVIL", "HEAVEN", "HELL", "CHURCH",
    "TEMPLE", "SCHOOL", "HOSPITAL", "POLICE", "DOCTOR", "NURSE", "TEACHER", "STUDENT",
    "FARMER", "SOLDIER", "PILOT", "SAILOR", "COOK", "BAKER", "ARTIST", "SINGER",
    "DANCER", "ACTOR", "WRITER", "POET", "PAINTER", "MUSICIAN", "ATHLETE", "COACH",
    "REFEREE", "CHAMPION", "WINNER", "LOSER", "GAME", "SPORT", "TEAM", "PLAYER",
    "BALL", "BAT", "GOAL", "NET", "COURT", "FIELD", "TRACK", "POOL", "GYM",
]

# Tech/Science words
TECH_WORDS = [
    "ALGORITHM", "BINARY", "BYTE", "CACHE", "CLOUD", "CODE", "COMPILER", "CPU",
    "DATA", "DEBUG", "DISK", "DOWNLOAD", "EMAIL", "ENCRYPT", "FILE", "FIREWALL",
    "FOLDER", "FUNCTION", "GIGABYTE", "HACK", "HARDWARE", "HTML", "HTTP", "ICON",
    "INPUT", "INTERNET", "JAVA", "KERNEL", "KEYBOARD", "LAPTOP", "LINK", "LINUX",
    "LOOP", "MALWARE", "MEMORY", "MODEM", "MONITOR", "MOUSE", "NETWORK", "NODE",
    "OFFLINE", "ONLINE", "OUTPUT", "PASSWORD", "PATCH", "PIXEL", "PLATFORM",
    "PLUGIN", "PORT", "PRINTER", "PROCESS", "PROGRAM", "PROTOCOL", "PYTHON",
    "QUERY", "RAM", "REBOOT", "ROUTER", "SCRIPT", "SERVER", "SOFTWARE", "SPAM",
    "STACK", "STORAGE", "SYNTAX", "TABLET", "THREAD", "TOKEN", "TROJAN", "UPLOAD",
    "URL", "USB", "USER", "VARIABLE", "VIRTUAL", "VIRUS", "WEB", "WIFI", "WINDOW",
    "WIRELESS", "ATOM", "BACTERIA", "CELL", "CHEMICAL", "DNA", "ELECTRON", "ELEMENT",
    "ENERGY", "ENZYME", "EXPERIMENT", "FORMULA", "GENE", "GRAVITY", "HYPOTHESIS",
    "ION", "LASER", "MAGNET", "MOLECULE", "NEUTRON", "NUCLEUS", "ORBIT", "PARTICLE",
    "PHOTON", "PLASMA", "PROTON", "QUANTUM", "RADIATION", "REACTOR", "SPECTRUM",
    "THEORY", "VACCINE", "VOLTAGE", "WAVELENGTH",
]


def get_word_list(list_type: str = "standard") -> List[str]:
    """
    Get a word list by type.

    Args:
        list_type: One of "standard", "easy", "tech", or "combined"

    Returns:
        List of words for the game
    """
    if list_type == "standard":
        return STANDARD_WORDS.copy()
    elif list_type == "easy":
        return EASY_WORDS.copy()
    elif list_type == "tech":
        return TECH_WORDS.copy()
    elif list_type == "combined":
        # Combine all lists, remove duplicates
        combined = set(STANDARD_WORDS + EASY_WORDS + TECH_WORDS)
        return list(combined)
    else:
        raise ValueError(f"Unknown word list type: {list_type}")


def select_game_words(
    word_list: Optional[List[str]] = None,
    count: int = 25,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Select random words for a game.

    Args:
        word_list: List of words to choose from (default: standard)
        count: Number of words to select (default: 25 for 5x5 grid)
        seed: Random seed for reproducibility

    Returns:
        List of selected words
    """
    if word_list is None:
        word_list = STANDARD_WORDS

    if len(word_list) < count:
        raise ValueError(f"Word list has {len(word_list)} words, need at least {count}")

    rng = random.Random(seed)
    return rng.sample(word_list, count)
