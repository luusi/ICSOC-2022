
from icsoc_2022.declare_utils import exactly_once, absence_2, alt_succession, alt_precedence, \
    not_coexistence, build_declare_assumption

COMPOSITION_MDP_UNDEFINED_ACTION = "failure"
DEFAULT_GAMMA = 0.9
GAMMA = 0.99

BUILD_RETRIEVE_STATOR = "build_retrieve_stator"
BUILD_RETRIEVE_ROTOR = "build_retrieve_rotor"
BUILD_RETRIEVE_INVERTER = "build_retrieve_inverter"
ASSEMBLE_MOTOR = "assemble_motor"
PAINTING = "painting"
RUNNING_IN = "running_in"
ELECTRIC_TEST = "electric_test"
STATIC_TEST = "static_test"

ALL_SYMBOLS = {
    BUILD_RETRIEVE_STATOR,
    BUILD_RETRIEVE_ROTOR,
    BUILD_RETRIEVE_INVERTER,
    ASSEMBLE_MOTOR,
    PAINTING,
    RUNNING_IN,
    ELECTRIC_TEST,
    STATIC_TEST,
}

declare_constraints = [
            exactly_once(BUILD_RETRIEVE_STATOR),
            exactly_once(BUILD_RETRIEVE_ROTOR),
            exactly_once(BUILD_RETRIEVE_INVERTER),
            exactly_once(RUNNING_IN),
            exactly_once(ASSEMBLE_MOTOR),
            absence_2(ELECTRIC_TEST),
            absence_2(PAINTING),
            absence_2(STATIC_TEST),
            alt_succession(BUILD_RETRIEVE_STATOR, ASSEMBLE_MOTOR),
            alt_succession(BUILD_RETRIEVE_ROTOR, ASSEMBLE_MOTOR),
            alt_succession(BUILD_RETRIEVE_INVERTER, ASSEMBLE_MOTOR),
            alt_succession(ASSEMBLE_MOTOR, RUNNING_IN),
            alt_precedence(ASSEMBLE_MOTOR, PAINTING),
            alt_precedence(ASSEMBLE_MOTOR, ELECTRIC_TEST),
            alt_precedence(ASSEMBLE_MOTOR, STATIC_TEST),
            not_coexistence(ELECTRIC_TEST, STATIC_TEST),
            build_declare_assumption(ALL_SYMBOLS),
        ]