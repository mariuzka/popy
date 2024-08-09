# %%
import popy


def test_1():
    model = popy.Model()
    creator = popy.Creator(model)

    class Max(popy.Agent):
        pass

    class Marius(popy.Agent):
        pass

    class Lukas(popy.Agent):
        pass

    class Meeting1(popy.MagicLocation):
        def filter(self, agent):
            return agent.type in ["Max", "Marius"]

    class Meeting2(popy.MagicLocation):
        def filter(self, agent):
            return agent.type in ["Marius", "Lukas"]

    _max = creator.create_agents(agent_class=Max, n=1)[0]
    _marius = creator.create_agents(agent_class=Marius, n=1)[0]
    _lukas = creator.create_agents(agent_class=Lukas, n=1)[0]
    creator.create_locations(location_classes=[Meeting1, Meeting2])

    assert len(model.locations) == 2
    assert len(model.agents) == 3
    assert _max.shared_locations(agent=_marius)[0].type == "Meeting1"
    # TODO offiziellster Weg leere Liste zu checken?
    assert not bool(_max.shared_locations(agent=_lukas))
    assert _marius.shared_locations(agent=_max)[0].type == "Meeting1"
    assert _marius.shared_locations(agent=_lukas)[0].type == "Meeting2"
    assert _lukas.shared_locations(agent=_marius)[0].type == "Meeting2"
    assert not bool(_lukas.shared_locations(agent=_max))
