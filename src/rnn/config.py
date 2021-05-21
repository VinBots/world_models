from dataclasses import dataclass


@dataclass
class LSTMParams:
    hidden_units: int #256


@dataclass
class GaussianMixt:
    number: int #5


@dataclass
class Configuration:
    version: str
    LSTMParams: LSTMParams
    GaussianMixt: GaussianMixt
    tr_epochs: int #20

