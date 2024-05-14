from enum import Enum

class Label_Type(Enum):
    COLOR = 1, 
    SIZE = 2,
    SIZE_NOT_TRAINED = 3,
    ANGLE = 4,
    CLASS = 5,
    PIXEL_AMOUNT = 6,
    SKELETON_LENGTH = 7,
    THICKNESS = 8,
    SPIKINESS = 9, 
    IRREGULARITY = 10