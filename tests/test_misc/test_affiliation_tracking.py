import pop2net as p2n

# Laufzeit mit ca. 4.3 GHz: ca. 29-30s


def test_affiliation_tracking():
    class Agent(p2n.Agent):
        def setup(self):
            self.n_locations = 0

        def count_locations(self):
            self.n_locations = len(self.locations)

    class Location(p2n.Location):
        def setup(self):
            self.n_agents = 0

        def count_agents(self):
            self.n_agents = len(self.agents)

    class Model(p2n.Model):
        def setup(self):
            n_agents = 1000
            self.add_agents(p2n.AgentList(self, n_agents, Agent))
            self.add_locations(p2n.LocationList(self, n_agents, Location))

            for i, location in enumerate(self.locations):
                location.add_agent(self.agents[i])

        def step(self):
            self.agents.count_locations()  # type: ignore
            self.locations.count_agents()  # type: ignore

    model = Model(parameters={"steps": 5})
    model.run()

    for agent in model.agents:
        assert agent.n_locations == 1

    for location in model.locations:
        assert location.n_agents == 1


if __name__ == "__main__":
    test_affiliation_tracking()
