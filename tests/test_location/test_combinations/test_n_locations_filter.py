# %%
import pandas as pd

import popy
from popy.creator import Creator


# %%
def test_1():
    df = pd.DataFrame(
        {
            "status": ["A", "A", "B", "B", "A", "A"],
        },
    )

    model = popy.Model()
    creator = Creator(model)

    class TestLocationA(popy.MagicLocation):
        n_locations = 2

        def filter(self, agent):
            return agent.status == "A"

    class TestLocationB(popy.MagicLocation):
        n_locations = 1
        def filter(self, agent):
            return agent.status == "B"

    creator.create_agents(df=df)
    creator.create_locations(location_classes=[TestLocationA, TestLocationB])
    inspector = popy.NetworkInspector(model=model)
    inspector.plot_bipartite_network()
    inspector.plot_agent_network(node_attrs=["status"])

    assert len(model.locations) == 3
    assert len(model.agents) == 6
    assert all(len(location.agents) == 2 for location in model.locations)
    assert all(agent.status == "A" for agent in model.locations[0].agents)
    assert all(agent.status == "A" for agent in model.locations[1].agents)
    assert all(agent.status == "B" for agent in model.locations[2].agents)

test_1()
# %%