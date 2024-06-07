import popy
import pytest

@pytest.fixture()
def model():
    return popy.Model()


@pytest.fixture()
def two_locations(model):
    return popy.LocationList(
        model,
        [
            popy.Location(model),
            popy.Location(model),
        ],
    )


def test_agent_creation(model):
    agent = popy.Agent(model)
    assert agent.model == model
    assert list(agent.locations) == []


def test_agentlist_broadcasting(model):
    agents = popy.AgentList(model, [popy.Agent(model), popy.Agent(model)])
    agents.x = 1
    assert sum(agents.x) == 2  # type: ignore


def test_agent_locations(model):

    agent = popy.Agent(model)

    location1 = popy.Location(model)
    location2 = popy.Location(model)

    agent.enter_location(location1)
    exp = popy.LocationList(model, [location1])
    assert agent.locations == exp

    agent.enter_location(location2)
    exp = popy.LocationList(model, [location1, location2])
    assert agent.locations == exp

    assert len(agent.locations) == 2


def test_agent_located_at_single_location(model):
    class Model(popy.Model):
        def setup(self):
            self.agents = popy.AgentList(self, 1, popy.Agent)
            self.locations = popy.LocationList(self, 2, popy.Location)
            self.agents[0].enter_location(self.locations[0])

    model = Model(parameters={"steps": 1})
    model.run()

    assert [len(loc.agents) for loc in model.locations] == [1, 0]

def test_agent_visits_two_locations(model):
    class Model(popy.Model):
        def setup(self):
            self.agents = popy.AgentList(self, 1, popy.Agent)
            self.locations = popy.LocationList(self, 2, popy.Location)
            self.agents[0].enter_location(self.locations[0])
            self.agents[0].enter_location(self.locations[1])

    model = Model(parameters={"steps": 1})
    model.run()

    assert [len(loc.agents) for loc in model.locations] == [1, 1]
