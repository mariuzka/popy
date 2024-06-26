"""Create a population for the simulation."""
from __future__ import annotations

import random
import warnings

from bokehgraph import BokehGraph
import networkx as nx
import pandas as pd

import popy
import popy.utils as utils

from .exceptions import PopyException

class PopMaker:
    """Creates and connects agents and locations."""

    def __init__(
        self,
        model: popy.Model | None = None,
        seed: int = 999,
    ) -> None:
        """Instantiate a population maker for a specific model.

        Args:
            model (popy.Model): Model, for which a population should be created
            seed (int, optional): A seed for reproducibility. Defaults to 999.
        """
        # TODO: Seed should default to None.

        self.model = model if model is not None else popy.Model()
        self.seed = seed
        self.rng = random.Random(seed)
        self.agents: popy.AgentList | None = None
        self.locations: popy.LocationList | None = None

    def _create_dummy_location(self, location_cls) -> popy.Location:
        location = location_cls(model=self._dummy_model)
        location.setup()
        return location

    def draw_sample(
        self,
        df: pd.DataFrame,
        n: int | None = None,
        sample_level: str | None = None,
        sample_weight: str | None = None,
        replace_sample_level_column: bool = True,
    ) -> pd.DataFrame:
        """Draw a sample from a dataframe.

        Args:
            df (:class:`pandas.DataFrame`): a pandas DataFrame
            n (int): Target size of the final sample. If this is higher the size of the input
                DataFrame, the sampling will occure with replacement. Without otherwise. If `n` is
                set to `None`, df is returned as it is.
            sample_level (str, optional): A variable the specifies sample units,
                i.e. rows that should always
            be sampled together. For instance, a household ID to sample by households.
                Defaults to None.
            sample_weight (str, optional): The column of df which should be used as
                probability weight. Defaults to None.
            replace_sample_level_column (bool): Should the original values of the sample level be
                overwritten by unique values after sampling to avoid duplicates?

        Returns:
            A pandas DataFrame.
        """
        df = df.copy()
        df = df.sample(frac=1)

        if n is None:
            return df

        elif sample_level is None:
            df_sample = df.sample(
                n=n,
                replace=not (n <= len(df) and sample_weight is None),
                weights=sample_weight,
                random_state=self.seed,
            )

        else:
            sample_level_ids = list(df[sample_level].drop_duplicates())

            if sample_weight is not None:
                weights = list(df.drop_duplicates(subset=sample_level)[sample_weight])

            samples = []
            counter = 0
            sample_cluster_id = 1
            while counter < n:
                if sample_weight is None:
                    random_id = self.rng.choice(sample_level_ids)
                else:
                    random_id = self.rng.choices(sample_level_ids, weights=weights, k=1)[0]

                sample = df.loc[df[sample_level] == random_id, :].copy()

                # create new unique ids for sample level variable
                if replace_sample_level_column:
                    sample.loc[:, sample_level + "_original"] = sample.loc[:, sample_level]
                    sample.loc[:, sample_level] = sample_cluster_id

                samples.append(sample)

                counter += len(sample)
                sample_cluster_id += 1

            df_sample = pd.concat(samples).reset_index(drop=True)

        return df_sample

    def create_agents(
            self,
            agent_class = popy.Agent,
            agent_class_attr: None | str = None,
            agent_class_dict: None | dict = None,
            df: pd.DataFrame | None = None,
            n: int | None = None,
            clear_agents: bool = False,
    ) -> popy.AgentList:
        """Creates agents from a pandas DataFrame.

        Creates one agent-instance of the given agent-class for each row of the given df,
        if df is not None.
        All columns of the df are added as instance attributes containing the row-specific values
        of the specific column.
        If df is None and n is not None, n default agents without any additional attributes are
        created.

        Args:
            agent_class: A class to instantiate all agents with. Every column in the DataFrame will
                result in an attribute of the agents.
            df: The DataFrame from which the agents should be created from.
            n: The number of agents that should be created. Defaults to None.
            clear_agents: Should existing agents be removed before creating new ones?

        Returns:
            A list of agents.
        """
        # remove agents from environment (network)
        if clear_agents:
            agent_nodes = [node for node in self.model.env.g.nodes if self.model.env.g.nodes[node]["bipartite"] == 0]
            for agent_node in agent_nodes:
                self.model.env.g.remove_node(agent_node)
            self.agents = None

        if df is not None:
            df = df.copy()

            # create one agent for each row in df
            agents = []
            for _, row in df.iterrows():
                if agent_class_dict is None:
                    agent = agent_class(model=self.model)
                else:
                    agent = agent_class_dict[row[agent_class_attr]](model=self.model)

                for col_name in df.columns:
                    if col_name == "id":
                        msg = "You are not allowed to set an agent attribute called `id`."
                        raise Exception(msg)
                    else:
                        setattr(agent, col_name, row[col_name])
                agents.append(agent)

        else:
            if n is not None:
                agents = [agent_class(model=self.model) for _ in range(n)]
            else:
                msg = "Either `df` or `n` must be not None."
                raise Exception(msg)


        agents = popy.AgentList(model=self.model, objs=agents)
        
        # Save agents 
        if self.agents is None:
            self.agents = agents.copy()
        else:
            self.agents.extend(agents)
            self.agents = popy.AgentList(model=self.model, objs=self.agents)

        return agents


    def _get_affiliated_agents(self, agents, dummy_location) -> list:
        return [agent for agent in agents if dummy_location.filter(agent)]

    def _get_mother_group_id(self, agent, dummy_location) -> str:
        if dummy_location.nest() is None:
            return "None"

        else:
            #mother_location = None

            # search for mother location assigned to this agent
            n_mother_locations_found = 0
            for location in agent.locations:
                if isinstance(location, dummy_location.nest()):
                    mother_location = location
                    n_mother_locations_found += 1

            # Check if the number of mother locations is not 1
            if n_mother_locations_found > 1:
                warnings.warn(
                    f"""For agent {agent},
                    {n_mother_locations_found} locations of class
                    {dummy_location.nest()} were found as potential mothers of
                    {dummy_location} in the list of locations.""",
                    stacklevel=2,
                )
            elif n_mother_locations_found == 0:
                return "None"

            return "-".join([str(mother_location.group_value), str(mother_location.group_id)])


    def _get_split_values(
            self,
            agents: list,
            dummy_location,
            allow_nesting: bool = False,
    ) -> set[int | str]:

        all_values = []
        for agent in agents:
            agent_values = utils.make_it_a_list_if_it_is_no_list(dummy_location.split(agent))

            if allow_nesting:
                # Add mother location's value to the value of the lower level location
                for i, value in enumerate(agent_values):
                    agent_values[i] = "-".join(
                        [self._get_mother_group_id(agent, dummy_location), str(value)],
                    )

            # Temporarely store group values as agent attribute
            # to assign them to the corresponding location group later
            agent._TEMP_group_values = agent_values
            all_values.extend(agent_values)

        return set(all_values)

    def _get_stick_value(self, agent, dummy_location):
        stick_value = dummy_location.stick_together(agent)
        if stick_value is None:
            return "None" + str(agent.id)
        else:
            return stick_value


    def _get_groups(self, agents, location_cls) -> list[list]:
        dummy_location = self._create_dummy_location(location_cls)

        # determine the number of groups needed
        if dummy_location.n_locations is None:
            n_location_groups = (
                1
                if dummy_location.size is None
                else max(dummy_location.round_function(len(agents) / dummy_location.size), 1)
            )
        else:
            n_location_groups = dummy_location.n_locations


        stick_values = {self._get_stick_value(agent, dummy_location) for agent in agents}
        groups: list[list] = [[]]
        dummy_location = self._create_dummy_location(location_cls)

        # for each group of sticky agents
        for stick_value in stick_values:
            sticky_agents = [
                agent for agent in agents
                if self._get_stick_value(agent, dummy_location) == stick_value
            ]

            assigned = False

            for _i, group in enumerate(groups):
                # if there are still enough free places available
                if (
                    dummy_location.size is None
                    or (dummy_location.size - len(group)) >= len(sticky_agents)
                ):
                    if sum(
                        [dummy_location.find(agent) for agent in sticky_agents],
                    ) == len(sticky_agents):
                        # assign agents
                        for agent in sticky_agents:
                            group.append(agent)
                            dummy_location.add_agent(agent)

                        assigned = True
                        break

            if not assigned:
                if dummy_location.allow_overcrowding and len(groups) >= n_location_groups:

                    # sort by the number of assigned agents
                    groups.sort(key=lambda x: len(x))

                    # assign agents to the group_list with the fewest members
                    for agent in sticky_agents:
                        groups[0].append(agent)

                    random.shuffle(groups)
                else:
                    new_group = []
                    dummy_location = self._create_dummy_location(location_cls)
                    # assign agents
                    for agent in sticky_agents:
                        new_group.append(agent)
                        dummy_location.add_agent(agent)

                    groups.append(new_group)

        random.shuffle(groups)
        return groups


    def _get_group_value_affiliated_agents(
            self,
            agents: list,
            group_value: int | str,
    ) -> list:
        group_affiliated_agents = [
            agent for agent in agents
            if group_value in agent._TEMP_group_values
        ]
        random.shuffle(group_affiliated_agents)
        return group_affiliated_agents


    def _get_melted_groups(self, agents: list, location_cls) -> list[list]:

        dummy_location = self._create_dummy_location(location_cls)

        # get all mother locations the agents are nested in
        all_mother_group_ids = {
            self._get_mother_group_id(agent, dummy_location) for agent in agents
        }

        # for each mother location
        for mother_group_id in all_mother_group_ids:

            # get agents that are part of this location
            nested_agents = [
                agent for agent in agents
                if self._get_mother_group_id(agent, dummy_location) == mother_group_id
            ]

            # a list that stores a list of groups for each location
            #[
            #[[_agent], [_agent], [_agent]], # groups of location 1
            #[[_agent], [_agent]],           # groups of location 2
            #]
            groups_to_melt_by_location: list[list[list]] = []


            # for each location that shall be melted
            for location_cls in dummy_location.melt():

                # create dummy location
                melt_dummy_location = self._create_dummy_location(location_cls)

                # get all agents that should be assigned to this location
                melt_location_affiliated_agents = self._get_affiliated_agents(
                    agents=nested_agents,
                    dummy_location=melt_dummy_location,
                )

                # get all values for which seperated groups/locations should be created
                melt_group_values = self._get_split_values(
                    agents=melt_location_affiliated_agents,
                    dummy_location=melt_dummy_location,
                    allow_nesting=False,
                )

                # for each split value: get groups and collect them in one list for all values
                location_groups_to_melt: list[list] = []
                for melt_group_value in melt_group_values:
                    melt_group_value_affiliated_agents = self._get_group_value_affiliated_agents(
                        agents=melt_location_affiliated_agents,
                        group_value=melt_group_value,
                    )
                    location_groups_to_melt.extend(
                        self._get_groups(
                            agents=melt_group_value_affiliated_agents,
                            location_cls=location_cls,
                        ),
                    )
                random.shuffle(location_groups_to_melt)
                groups_to_melt_by_location.append(location_groups_to_melt)

            # Melt groups
            all_melted_groups: list[list] = []
            z = sorted(
                [len(groups_to_melt) for groups_to_melt in groups_to_melt_by_location],
                reverse=True if dummy_location.multi_melt else False,
            )[0]
            for i in range(z):
                melted_group = []
                for groups_to_melt in groups_to_melt_by_location:
                    if len(groups_to_melt) > 0:
                        if dummy_location.multi_melt:
                            melted_group.extend(groups_to_melt[i % len(groups_to_melt)])
                        else:
                            try:
                                melted_group.extend(groups_to_melt[i])
                            except IndexError:
                                pass

                all_melted_groups.append(melted_group)

        return all_melted_groups

    def create_locations(
        self,
        location_classes: list,
        agents: list | popy.AgentList | None = None,
        clear_locations: bool = False,
    ) -> popy.LocationList:
        """Creates location instances and connects them with the given agent population.

        Args:
            location_classes (list): A list of location classes.
            agents (list | popy.AgentList): A list of agents.
            clear_locations: Should existing locations be removed before creating new ones?

        Returns:
            popy.LocationList: A list of locations.
        """

        if agents is None:
            agents = self.agents

        # remove locations from environment (network)
        if clear_locations:
            location_nodes = [node for node in self.model.env.g.nodes if self.model.env.g.nodes[node]["bipartite"] == 1]
            for location_node in location_nodes:
                self.model.env.g.remove_node(location_node)
            self.locations = None

        self._dummy_model = popy.Model()
        self._dummy_model.agents = agents
        for agent in agents:
            self._dummy_model.env.add_agent(agent)

        for location_cls in location_classes:

            str_location_cls = utils._get_cls_as_str(location_cls)
            for agent in agents:
                setattr(agent, str_location_cls, None)

        locations = []

        # for each location class
        for location_cls in location_classes:
            str_location_cls = utils._get_cls_as_str(location_cls)

            #WaRUM GEHT DAS NICHT???????????????
            random.shuffle(agents)

            # create location dummy in order to use the location's methods
            dummy_location = self._create_dummy_location(location_cls)

            if not dummy_location.melt():
                # get all agents that could be assigned to locations of this class
                affiliated_agents = self._get_affiliated_agents(
                    agents=agents,
                    dummy_location=dummy_location,
                )

            else:
                affiliated_agents = []
                for melt_location_cls in dummy_location.melt():
                    melt_dummy_location = self._create_dummy_location(melt_location_cls)
                    affiliated_agents.extend(
                        self._get_affiliated_agents(
                            agents=agents,
                            dummy_location=melt_dummy_location,
                        ),
                    )

            # get all values that are used to split the agents into groups
            group_values = self._get_split_values(
                agents=affiliated_agents,
                dummy_location=dummy_location,
                allow_nesting=True,
            )

            # for each group split value
            for group_value in group_values:
                # get all agents with that value
                group_value_affiliated_agents = self._get_group_value_affiliated_agents(
                    agents=affiliated_agents,
                    group_value=group_value,
                )

                # if this location does not glue together other locations
                if not dummy_location.melt():
                    group_lists: list[list] = self._get_groups(
                        agents=group_value_affiliated_agents,
                        location_cls=location_cls,
                    )
                else:
                    group_lists = self._get_melted_groups(
                        agents=group_value_affiliated_agents,
                        location_cls=location_cls,
                    )

                # for each group of agents
                for i, group_list in enumerate(group_lists):
                    dummy_location = self._create_dummy_location(location_cls)

                    dummy_location.group_agents = group_list

                    # get all subgroub values
                    subgroup_values = {
                        agent_subgroup_value
                        for agent in group_list
                        for agent_subgroup_value
                        in utils.make_it_a_list_if_it_is_no_list(dummy_location.subsplit(agent))
                    }

                    # for each group of agents assigned to a specific sublocation
                    for j, subgroup_value in enumerate(subgroup_values):

                        # get all subgroup affiliated agents
                        subgroup_affiliated_agents = []

                        #for agent in group_affiliated_agents:
                        for agent in group_list:
                            agent_subgroup_value = utils.make_it_a_list_if_it_is_no_list(
                                dummy_location.subsplit(agent),
                            )
                            if subgroup_value in agent_subgroup_value:
                                subgroup_affiliated_agents.append(agent)

                        # Build the final location
                        subgroup_location = location_cls(model=self.model)
                        subgroup_location.setup()
                        subgroup_location.group_value = group_value
                        subgroup_location.subgroup_value = subgroup_value
                        subgroup_location.group_id = i
                        subgroup_location.subgroup_id = j
                        subgroup_location.group_agents = group_list # maybe delete later

                        # Assigning process:
                        for agent in subgroup_affiliated_agents:
                            subgroup_location.add_agent(agent)

                            group_info_str = (
                                f"gv={subgroup_location.group_value}, \
                                    gid={subgroup_location.group_id}"
                            )

                            setattr(agent, str_location_cls, group_info_str)

                        locations.append(subgroup_location)

        # TODO:
        # Warum gibt es keinen Fehler, wenn man ein Argument falsch schreibt? Habe gerade ewig
        # nach einem Bug gesucht und letzt hatte ich nur das "j" in "objs" vergessen
        locations = popy.LocationList(
            model=self.model,
            objs=locations,
        )

        # save locations
        if self.locations is None:
            self.locations = locations
        else:
            self.locations.extend(locations)

        self._dummy_model.locations = locations

        # execute an action after all locations have been created
        for location in self.locations:
            location.do_this_after_creation()

        # delete temporary agent attributes
        for agent in self._dummy_model.agents:
            if hasattr(agent, "_TEMP_group_values"):
                del(agent._TEMP_group_values)

        # delete temporary location attributes
        for location in locations:
            del(location.group_agents)

        return self.locations


    def make(
        self,
        df: pd.DataFrame,
        location_classes: list,
        agent_class: type[popy.Agent]=popy.Agent,
        agent_class_attr: None | str = None,
        agent_class_dict: None | dict = None,
        n_agents: int | None = None,
        sample_level: str | None = None,
        sample_weight: str | None = None,
        replace_sample_level_column: bool = True,
    ) -> tuple:
        """Creates agents and locations based on a given dataset.

        Combines the PopMaker-methods `draw_sample()`, `create_agents()` and `create_locations()`.

        Args:
            df (pd.DataFrame): A data set with individual data that forms the basis for
                the creation of agents. Each row is (potentially) translated into one agent.
                Each column is translated into one agent attribute.
            agent_class (type[popy.Agent]): The class from which the agent instances are created.
            location_classes (list): A list of classes from which the location instances are
                created.
            n_agents (Optional[int], optional): The number of agents that will be created.
                If `n_agents` is set to None, each row of `df` is translated into exactly one agent.
                Otherwise, rows are randomly drawn (with replacement,
                if `n_agents > len(df)`) from `df` until the number of agents created
                equals `n_agents`.
            sample_level (Optional[str], optional): If `sample_level` is None,
                the rows are sampled individually.
                Otherwise the rows are sampled as groups. `sample_level` defines
                which column of `df` contains the group id.
            sample_weight (Optional[str]): The column of df in which should be used as probability
                weight during sampling.
            replace_sample_level_column (bool): Should the original values of the sample level be
                overwritten by unique values after sampling to avoid duplicates?

        Returns:
            tuple: A list of agents and a list of locations.
        """
        # draw a sample from dataset
        df_sample = self.draw_sample(
            df=df,
            n=n_agents,
            sample_level=sample_level,
            sample_weight=sample_weight,
            replace_sample_level_column=replace_sample_level_column,
        )

        # create agents
        agents = self.create_agents(
            df=df_sample, 
            agent_class=agent_class, 
            agent_class_attr=agent_class_attr,
            agent_class_dict=agent_class_dict,
            )

        # create locations
        locations = self.create_locations(agents=agents, location_classes=location_classes)

        return agents, locations


    def get_df_agents(
            self,
            columns: None | list[str] = None,
            drop_agentpy_columns: bool =True,
    ) -> pd.DataFrame:
        """Returns the latest created population of agents as a dataframe.

        Args:
            columns (list | None): A list of column names that sould be kept.
                All other columns are deleted.
            drop_agentpy_columns (bool): Deletes some columns created by AgentPy.

        Raises:
            PopyException: _description_

        Returns:
            pd.DataFrame: A dataframe which contains one row for each
            agent and one column for each agent attribute.
        """
        if self.agents is None:
            msg = "There are no agents."
            raise PopyException(msg)

        df = pd.DataFrame([vars(agent) for agent in self.agents])

        if drop_agentpy_columns:
            df = df.drop(
            columns=[
                "_var_ignore",
                "id",
                "type",
                "log",
                "model",
                "p",
            ],
            )

        if columns is not None:
            df = df.loc[:,columns]

        return df


    def _plot_network(
            self,
            network_type,
            node_color: str | None,
            node_attrs: list | None,
            edge_alpha: str,
            edge_color: str,
            include_0_weights: bool,
    ):

        if network_type == "bipartite":
            graph = self.model.env.g.copy()
            for i in graph:
                if node_attrs is not None:
                    for node_attr in node_attrs:
                        graph.nodes[i][node_attr] = graph.nodes[i]["_obj"][node_attr]
                del graph.nodes[i]["_obj"]
            node_color = "cls" if node_color is None else node_color

        elif network_type == "agent":
            graph = utils.create_agent_graph(
                agents=self.agents,
                node_attrs=node_attrs,
                include_0_weights=include_0_weights,
            )
            node_color = "firebrick" if node_color is None else node_color

        graph_layout = nx.drawing.spring_layout(graph)
        plot = BokehGraph(graph, width=500, height=500, hover_edges=True)
        plot.layout(layout=graph_layout)
        plot.draw(
            node_color=node_color,
            edge_alpha=edge_alpha,
            edge_color=edge_color,
        )

    def plot_bipartite_network(
            self,
            node_color: str | None = None,
            node_attrs: list | None = None,
            edge_alpha: str = "weight",
            edge_color: str = "black",
            include_0_weights: bool = True,
    ) -> None:
        """Plots the two-mode network of agents and locations.

        Args:
            node_color (str, optional): The node attribute that determines the
                color of the nodes. If None, the node color represents whether
                it is a location or an agent instance.
            node_attrs (list | None, optional): A list of agent and location attributes that
                should be shown as node attributes in the network graph. Defaults to None.
            edge_alpha (str, optional): The node attribute that determines the edges' transparency.
                Defaults to "weight".
            edge_color (str, optional): The node attribute that determines the edges' color.
                Defaults to "black".
            include_0_weights (bool, optional): Should edges with a weight of zero be included in
                the plot? Defaults to True.
        """
        if node_attrs is None:
            node_attrs = ["cls"]
        elif isinstance(node_attrs, list):
            if "cls" not in node_attrs:
                node_attrs.append(node_attrs)
        else:
            raise Exception

        self._plot_network(
            network_type="bipartite",
            node_color=node_color,
            node_attrs=node_attrs,
            edge_alpha=edge_alpha,
            edge_color=edge_color,
            include_0_weights=include_0_weights,
        )

    def plot_agent_network(
            self,
            node_color: str | None = "firebrick",
            node_attrs: list | None = None,
            edge_alpha: str = "weight",
            edge_color: str = "black",
            include_0_weights: bool = True,
    ) -> None:
        """Plots the agent network.

        Args:
            node_color (str, optional): The node attribute that determines the
                color of the nodes. If None, the node color represents whether
                it is a location or an agent instance.
            node_attrs (list | None, optional): A list of agent and location attributes that
                should be shown as node attributes in the network graph. Defaults to None.
            edge_alpha (str, optional): The node attribute that determines the edges' transparency.
                Defaults to "weight".
            edge_color (str, optional): The node attribute that determines the edges' color.
                Defaults to "black".
            include_0_weights (bool, optional): Should edges with a weight of zero be included in
                the plot? Defaults to True.
        """
        self._plot_network(
            network_type="agent",
            node_color=node_color,
            node_attrs=node_attrs,
            edge_alpha=edge_alpha,
            edge_color=edge_color,
            include_0_weights=include_0_weights,
        )
