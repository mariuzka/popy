from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Set
from typing import Tuple

from faker.providers import BaseProvider
import numpy as np

from . import constants as c

class SOEPProvider(BaseProvider):
    """
    Helps to create Fake Data that is as close to the SOEP as possible.
    Parameters of the random distributions are taken from the original dataset.
    """

    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self.rng = np.random.default_rng(self.generator._global_seed)

        # For hhid
        self.hh_type_dist = lambda: self._multinomial_select(c.HH_TYPE)
        self.hh_size_dist = lambda: self.rng.poisson(lam=c.HH_SIZE["lam"])
        self.hh_n_children_dist = lambda: self._multinomial_select(c.HH_N_CHILDREN)
        self.hh_cur = 0
        self.used_hhids: Set[int] = set()

        # For age
        self.age_child_dist = lambda: self.rng.normal(
            loc=c.AGE_DIST["child"]["loc"],
            scale=c.AGE_DIST["child"]["loc"],
        )
        self.age_adult_dist = lambda: self.rng.normal(
            loc=c.AGE_DIST["adult"]["loc"],
            scale=c.AGE_DIST["adult"]["loc"],
        )
        self.age_senior_dist = lambda: self.rng.normal(
            loc=c.AGE_DIST["senior"]["loc"],
            scale=c.AGE_DIST["senior"]["loc"],
        )

        # For gender
        self.gender_dist = lambda: self._multinomial_select(c.GENDER_DIST)

        # For work hours per day
        self.work_at_all_dist = lambda: self.rng.binomial(n=1, p=c.WORKING_HOURS["at_all"])
        self.work_full_day_dist = lambda: self.rng.binomial(n=1, p=c.WORKING_HOURS["8_hours"])
        self.work_dist = lambda: self.rng.normal(
            loc=c.WORKING_HOURS["other"]["loc"],  # type: ignore
            scale=c.WORKING_HOURS["other"]["scale"],  # type: ignore
        )

        # type of work
        self.nace2_dist = lambda: self._multinomial_select(c.NACE2_DIVISIONS)

    def _multinomial_select(self, dist: Dict):
        pvals = list(dist.values())
        labels = {i: label for i, label in enumerate(dist.keys())}

        index = int(np.argmax(self.rng.multinomial(n=1, pvals=pvals)))

        return labels[index]

    def _hhid(self) -> int:
        counter = 0
        while True:
            hhid = self.generator.random_int()
            if hhid not in self.used_hhids:
                return hhid
            counter += 1
            if counter > 1000:
                raise ValueError("Could not find new hhid after 1000 iterations!")

    def household(self) -> Tuple[int, str, int, int]:
        """Create correlated fake numbers on household level

        Returns:
            Tuple[int, str, int, int]: [hh_id, hh_type, hh_size, n_children]
        """

        hh_type = self.hh_type_dist()

        if hh_type == "children":
            min_hh_size = 2
        else:
            min_hh_size = 1

        while (hh_size := self.hh_size_dist()) < min_hh_size:
            pass

        if hh_type != "children":
            n_children = 0
        else:
            while (n_children := self.hh_n_children_dist()) >= hh_size:
                pass

        return (self._hhid(), hh_type, hh_size, n_children)

    def person(self, person: Literal["child", "adult", "senior"]):

        work_hours = self.work_hours_per_day(person=person)
        nace2 = self.nace2_division(work_hours)
        case = {
            "age": self.age(person=person),
            "gender": self.gender(),
            "work_hours_day": work_hours,
            "nace2_division": nace2,
        }
        return case

    def household_persons(self) -> List[Dict]:
        hh_id, hh_type, hh_size, n_children = self.household()

        if hh_type == "no_children":
            persons = [self.person("adult") for _ in range(hh_size)]
        elif hh_type == "senior":
            persons = [self.person("senior") for _ in range(hh_size)]
        else:
            n_adults = hh_size - n_children
            persons = [self.person("adult") for _ in range(n_adults)]
            persons.extend([self.person("child") for _ in range(n_children)])

        for person in persons:
            person["hid"] = hh_id

        return persons

    def age(self, person: Literal["child", "adult", "senior"] = "adult") -> float:

        if person == "child":
            min_age = 0.0
            max_age = 18
            dist = self.age_child_dist
        elif person == "adult":
            min_age = 18
            max_age = 100
            dist = self.age_adult_dist
        elif person == "senior":
            min_age = 40
            max_age = 100
            dist = self.age_senior_dist

        # iterate until a positive number
        while not min_age <= (age := dist()) <= max_age:
            pass

        return round(age, 0)

    def work_hours_per_day(self, person: Literal["child", "adult", "senior"]) -> float:

        # children do not work
        if person != "adult":
            return 0.0

        # about half of people work at all
        if self.work_at_all_dist() == 0:
            return 0.0

        # of the remaining, about 19% have a full work day (8 hours)
        if self.work_full_day_dist() == 1:
            return 8.0

        # rest is drawn for normal (more like bimodal in SOEP but who cares..)
        while not 0.0 < (hours := self.work_dist()) < 16.0:
            pass

        return round(hours, 6)

    def gender(self) -> str:
        return self.gender_dist()

    def nace2_division(self, work_hours: float):
        if work_hours > 0:
            return self.nace2_dist()
        else:
            return -2
