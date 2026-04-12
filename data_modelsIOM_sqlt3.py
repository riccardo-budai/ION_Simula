# 'data_modelsIOM_sqlt3.py'

from dataclasses import dataclass
from datetime import datetime

# definition of subject fields ----------------------------------------------------------------------------------------
@dataclass
class Subject:
    """
    list of subjects recorded (uniques): identifying information used for the purpose of identifying
    the same operator across different and subsequent sessions.
    The dbaseclass field can later be used to group different types of users:
    learners, instructors, operators authorized to insert new scenarios or anomalies, etc.
    """
    idcode: int         # subject code
    firstname: str      # first name (or surname, depending on convention)
    lastname: str       # last name (or first name, depending on convention)
    height: int         # not used
    weight: int         # not used
    age: int            # age
    gender: str         # gender
    fname: str          # unique identifier of file prefix for results, reports, AI, etc...
    dbaseclass: str     # working group
    level: str          # user level definition: ENTRY, ADVANCED, TUTOR
    datetime: datetime  # insertion date and time

# definition of simulation work session fields -------------------------------------------------------------------------
@dataclass
class Session:
    """
    sessions (multiples but unique indexed) recorded for each operator: multiple sessions for each operator.
    The dbasename field is updated based on the dbaseclass field chosen from the subjects table.
    """
    idcode: int         # subject code
    idsession: int      # session code
    session: str        # type of simulation session
    taskname: str       # type of task used in the simulation
    simulacode: str     # renamed field: to store code of simulation for this session
    sesnote: str        # annotations related to the executed simulation session
    fname: str          # filename to save all operational simulation data
    dbasename: str      # defined by the dbaseclass field in the subjects table
    datetime: datetime  # execution date and time

# definition of simulation scenarios_setup fields ----------------------------------------------------------------------
@dataclass
class Scenari:
    """
    describes all scenarios present in the simulation models that can be selected
    by the learner for learning or in-depth study.
    Currently, there are 17 unimodal and one (IOM) multimodal scenarios. Several multimodal scenarios
    will be arranged depending on user demand.
    """
    scenario_id: int
    label_id: str
    nome_scenario: str  # scenario name
    descrizione: str    # description
    trigger_tipo: str   # trigger type
    trigger_delay: int  # trigger delay
    data_creazione: datetime # creation date

# definition of scenari_anomaly fields, detail of the scenari_setup table ---------------------------------------------
@dataclass
class Anomaly:
    """
    describes the types of anomalies that can occur for each scenario outlined in the scenari_setup table.
    All data relating to the settings of each anomaly and the operating methods are described
    in a .json file referenced by the table and saved in the "anomaly" directory.
    """
    anomaly_id: int
    scenario_id: int
    label_id: str
    descrizione: str    # description
    json_anomaly_id: str
    data_creazione: datetime # creation date
    azione_corretta: str

@dataclass
class Results:
    """
    describes the results for each simulation session: one-to-one relationship
    """
    decision_id: int
    idsession: int      # session code
    anomaly_id: int
    timestamp_decision: datetime
    azione_presa: str   # action taken
    valore_parametro: float # parameter value
    esito_corretto: int # correct outcome (1 for correct, 0 for incorrect)
    punti_ottenuti: int # points obtained
    tempo_risposta_sec: float # response time in seconds