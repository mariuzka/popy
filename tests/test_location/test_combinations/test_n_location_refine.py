# %%
import pandas as pd

import popy

# %%


# TODO Gefundener Nebeneffekt:
# Die Aufteilung der Agenten ist anders als Erwartet
# Hätte erwartet loc 1 mit 3x "pupil" und loc 2 mit 2x "teacher"
# Ist es faslch diese Aufteilung zu erwarten?
def test_1():
    df = pd.DataFrame({"status": ["pupil", "pupil", "pupil", "teacher", "teacher"]})

    class TestLocation(popy.MagicLocation):
        n_locations = 2

        def refine(self):
            if len(self.agents) % 2 == 0:
                new_agent = popy.Agent(model)
                new_agent.status = "pupil"
                self.add_agent(new_agent)

    model = popy.Model()

    creator = popy.Creator(model=model)
    creator.create(df=df, location_classes=[TestLocation])

    inspector = popy.NetworkInspector(model)
    inspector.plot_bipartite_network()
    inspector.plot_agent_network(node_attrs=df.columns, node_color="status")

    assert len(model.locations) == 2
    assert len(model.agents) == 6
    assert len(model.locations[0].agents) == 3
    assert len(model.locations[1].agents) == 3
    assert sum(agent.status == "pupil" for agent in model.locations[0].agents) == 2
    assert sum(agent.status == "teacher" for agent in model.locations[0].agents) == 1
    assert sum(agent.status == "pupil" for agent in model.locations[1].agents) == 2
    assert sum(agent.status == "teacher" for agent in model.locations[1].agents) == 1


test_1()
# %%