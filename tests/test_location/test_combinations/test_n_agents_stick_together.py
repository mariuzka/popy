# %%

import pandas as pd

import popy


def test_1():
    df = pd.DataFrame({"status": ["pupil", "pupil", "pupil", "pupil"], "class_id": [1, 2, 1, 2]})

    class Classroom(popy.MagicLocation):
        n_agents = 2

        def stick_together(self, agent):
            return agent.class_id

    model = popy.Model()
    creator = popy.Creator(model=model)
    creator.create(df=df, location_classes=[Classroom])

    assert len(model.agents) == 4
    assert len(model.locations) == 2

    for location in model.locations:
        assert len(location.agents) == 2

    for agent in model.agents:
        assert agent.neighbors(location_classes=[Classroom])[0].class_id == agent.class_id

    inspector = popy.NetworkInspector(model)
    inspector.plot_bipartite_network()
    inspector.plot_agent_network(node_attrs=df.columns, node_color="class_id")


test_1()

# %%


def test_2():
    df = pd.DataFrame({"status": ["pupil", "pupil", "pupil", "pupil"], "class_id": [1, 1, 1, 1]})

    class Classroom(popy.MagicLocation):
        n_agents = 1

        def stick_together(self, agent):
            return agent.class_id

    model = popy.Model()

    creator = popy.Creator(model=model)
    creator.create(df=df, location_classes=[Classroom])

    inspector = popy.NetworkInspector(model)
    inspector.plot_bipartite_network()
    inspector.plot_agent_network(node_attrs=df.columns, node_color="status")

    assert len(model.agents) == 4
    assert len(model.locations) == 1


test_2()


# %%


def test_3():
    df = pd.DataFrame(
        {
            "status": ["pupil", "pupil", "pupil", "pupil", "pupil", "pupil", "pupil"],
            "class_id": [1, 1, 1, 1, 2, 2, 3],
        }
    )

    class Classroom(popy.MagicLocation):
        n_agents = 1

        def stick_together(self, agent):
            return agent.class_id

    model = popy.Model()

    creator = popy.Creator(model=model)
    creator.create(df=df, location_classes=[Classroom])

    inspector = popy.NetworkInspector(model)
    inspector.plot_bipartite_network()
    inspector.plot_agent_network(node_attrs=df.columns, node_color="status")

    assert len(model.agents) == 7
    assert len(model.locations) == 3

    counter = 0
    for location in model.locations:
        if location.agents[0].class_id == 1:
            assert len(location.agents) == 4
            counter += 1

        if location.agents[0].class_id == 2:
            assert len(location.agents) == 2
            counter += 1

        if location.agents[0].class_id == 3:
            assert len(location.agents) == 1
            counter += 1

    assert counter == 3


test_3()

# %%