import pop2net as p2n


def test_1():
    class City(p2n.MagicLocation):
        n_agents = 4

    class Group(p2n.MagicLocation):
        n_agents = 2

        def split(self, agent):
            return agent.group

    model = p2n.Model()
    creator = p2n.Creator(model=model)

    for i in range(8):
        agent = p2n.Agent(model=model)
        agent.group = i % 2

    creator.create_locations(location_classes=[City, Group])

    for location in model.locations.select(model.locations.type == "City"):
        assert int(location.agents[0].group) == 0
        assert int(location.agents[1].group) == 1
        assert int(location.agents[2].group) == 0
        assert int(location.agents[3].group) == 1

    for location in model.locations.select(model.locations.type == "Group"):
        assert location.agents[0].group == location.agents[1].group

    # not all members of the same group are also in the same city (which is not desired)
    assert not all(
        location.agents[0].City == location.agents[1].City for location in model.locations
    )

    class GroupNestedInCity(Group):
        def nest(self):
            return City

    model = p2n.Model()
    creator = p2n.Creator(model=model)
    creator.create_locations(location_classes=[City, GroupNestedInCity])

    for location in model.locations.select(model.locations.type == "City"):
        assert int(location.agents[0].group) == 0
        assert int(location.agents[1].group) == 1
        assert int(location.agents[2].group) == 0
        assert int(location.agents[3].group) == 1

    for location in model.locations.select(model.locations.type == "Group"):
        assert location.agents[0].group == location.agents[1].group

    # all members of a group are in the same city
    assert all(location.agents[0].City == location.agents[1].City for location in model.locations)
