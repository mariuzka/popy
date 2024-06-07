import numpy as np
import popy

# Test that examples run without errors

def test_wealth_model():
    def gini(x):

        """ Calculate Gini Coefficient """
        # By Warren Weckesser https://stackoverflow.com/a/39513799

        x = np.array(x)
        mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
        rmad = mad / np.mean(x)  # Relative mean absolute difference
        return 0.5 * rmad


    class WealthAgent(popy.Agent):

        """ An agent with wealth """

        def setup(self):

            self.wealth = 1

        def wealth_transfer(self):

            if self.wealth > 0:

                partner = self.model.agents.random()
                partner.wealth += 1
                self.wealth -= 1


    class WealthModel(popy.Model):

        """
        Demonstration model of random wealth transfers.

        See Also:
            Notebook in the model library: :doc:`agentpy_wealth_transfer`

        Arguments:
            parameters (dict):

                - agents (int): Number of agents.
                - steps (int, optional): Number of time-steps.
        """

        def setup(self):
            self.agents = popy.AgentList(self, self.p.agents, WealthAgent)
            self.locations = popy.LocationList(self, [popy.Location(self)])
            for agent in self.agents:
                self.locations[0].add_agent(agent)

        def step(self):
            self.agents.wealth_transfer()

        def update(self):
            self.gini = gini(self.agents.wealth)
            self.record("gini")

        def end(self):
            self.report("gini")


    parameters = {
        "seed": 42,
        "agents": 1000,
        "steps": 100,
    }

    model = WealthModel(parameters)
    _ = model.run(display=False)
    assert model.reporters["gini"] == 0.627486
